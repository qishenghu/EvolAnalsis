#!/bin/bash
# ============================================================================
# NeurIPS 2026 rebuttal experiment queue — A100 server, GPUs 0,1,2,4
# (paper-identical infra: 4xA100, so numbers are directly comparable)
#
# MINIMAL phase (finish first -> enough to write an effective rebuttal):
#   M1. webshop duet seed2025     — WS multi-seed pt.1 (y9x6 Q1); validates WS stack
#   M2. alfworld obsnoise_soft    — soft matching under obs noise (y9x6 W1 / bDeY W2)
#   M3. webshop duet seed2027     — WS multi-seed pt.2
#   M4. alfworld obsnoise_hash    — exact matching under obs noise (premise side)
#
# EXTENDED phase (continues automatically unless logs/A100_STOP_AFTER_MINIMAL exists):
#   E1. webshop sft (rerun)       — WS SFT ckpt + SFT curve (bDeY Q2)
#   E2. alfworld soft_clean       — soft matching drop-in check on clean obs
#   E3. webshop sft_rl seed2025   — baseline multi-seed (needs E1 ckpt)
#   E4. webshop sft_rl seed2027
#   E5. alfworld teacher14b       — weak-teacher DUET (UyKJ Q1 / y9x6);
#                                   requires 14B cache from run_a100_teacher_sampling.sh
#                                   (runs on spare GPU 6 in parallel; E5 skipped if missing)
#   E6. alfworld teacher32b       — optional 2nd teacher-quality point (skipped if no cache)
#
# Reference numbers (paper, seed 2026): WS DUET 36.0% | WS SFT+GRPO 18.5%
#   AF DUET 47.5% | AF -SC 31.0% | AF SFT+GRPO 30.0%
#
# Results appended to NeurIPS_2026_Latex/data/a100_rebuttal_results.md
# Stop gracefully:      touch logs/A100_REBUTTAL_STOP        (between runs)
# Stop after minimal:   touch logs/A100_STOP_AFTER_MINIMAL
# Launch:               nohup bash run_a100_rebuttal_queue.sh > logs/a100_queue.log 2>&1 &
# ============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Single-instance guard. Two concurrent queues fight over the same GPUs and env
# service ports and silently corrupt each other's runs. A PID file is used rather
# than flock because the env services this script starts inherit the lock file
# descriptor and would keep an flock alive after the queue itself exits.
mkdir -p logs
QUEUE_PIDFILE=logs/.a100_queue.pid
if [ -f "$QUEUE_PIDFILE" ] && kill -0 "$(cat "$QUEUE_PIDFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: another A100 rebuttal queue is already running (PID $(cat "$QUEUE_PIDFILE"))."
    echo "       Check: pgrep -fa 'run_a100_rebuttal_queue.sh'"
    exit 1
fi
echo $$ > "$QUEUE_PIDFILE"
trap 'rm -f "$QUEUE_PIDFILE"' EXIT

source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

export GPUS="0,1,2,4"
CFG_ROOT="config/duet_paper_experiments_configs/rebuttal_neurips"
RESULTS_LOG="NeurIPS_2026_Latex/data/a100_rebuttal_results.md"
STOP_FILE="logs/A100_REBUTTAL_STOP"
STOP_AFTER_MINIMAL="logs/A100_STOP_AFTER_MINIMAL"
mkdir -p logs NeurIPS_2026_Latex/data

if [ ! -f "$RESULTS_LOG" ]; then
    cat > "$RESULTS_LOG" <<'EOF'
# A100 rebuttal results (GPUs 0,1,2,4 — paper-identical infra)

Reference (paper, seed 2026): WS DUET 36.0% | WS SFT+GRPO 18.5% | AF DUET 47.5% | AF -SC 31.0% | AF SFT+GRPO 30.0%
Metric: val@100 strict SR (score >= 1.0), 200 tasks.

| finished | host | experiment | strict SR | lenient SR | mean reward | n | status |
|---|---|---|---|---|---|---|---|
EOF
fi

# ---------------------- helpers ----------------------

kill_ray_stragglers() {
    # Scoped kill: only Ray worker processes. NEVER broaden these patterns.
    local n
    n=$(ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | wc -l)
    if [ "$n" -gt 0 ]; then
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
        sleep 5
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
        sleep 3
    fi
}

wait_for_gpu_clean() {
    local _i
    for _i in {1..60}; do
        local used
        used=$(nvidia-smi --id=0,1,2,4 --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>30000' | wc -l)
        if [ "$used" = "0" ]; then echo "  GPUs 0,1,2,4 clean"; return 0; fi
        echo "  Waiting for GPUs 0,1,2,4 ($used busy)... ${_i}/60"; sleep 30
    done
    echo "  WARN: timed out waiting; proceeding anyway"
}

restart_env() {
    # $1: alfworld | webshop. Touches ONLY main ports (36001/8081, 36003/8083);
    # the AUX sampling stack (18011/18091) is never affected. kill_port inside the
    # env scripts refuses to kill ray/vLLM processes, since 36001/36003 fall inside
    # this host's ephemeral port range (32768-60999) and can collide with vLLM.
    bash start_env_alfworld.sh stop 2>/dev/null || true
    bash start_env_webshop.sh stop 2>/dev/null || true
    sleep 8
    if [ "$1" = "webshop" ]; then
        bash start_env_webshop.sh
    else
        bash start_env_alfworld.sh
    fi
    sleep 12
}

env_of() {
    case "$1" in
        webshop_*) echo webshop ;;
        *)         echo alfworld ;;
    esac
}

parse_and_log() {
    local name=$1
    local env=$(env_of "$name")
    local val_log="experiments/${env}/${name}/validation_log/100.jsonl"
    python - <<PYEOF
import json, os, datetime
val_log = "${val_log}"; name = "${name}"
results_log = "${RESULTS_LOG}"
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
if not os.path.exists(val_log):
    line = f"| {ts} | 4xA100(0124) | {name} | - | - | - | - | val@100 MISSING |"
else:
    n = 0; sr_strict = 0; sr_lenient = 0; rw = 0.0
    with open(val_log) as fh:
        for ln in fh:
            try: x = json.loads(ln)
            except Exception: continue
            n += 1
            s = float(x.get("score", x.get("reward", 0.0)))
            rw += s
            if s >= 1.0: sr_strict += 1
            if s >= 0.9: sr_lenient += 1
    if n == 0:
        line = f"| {ts} | 4xA100(0124) | {name} | - | - | - | 0 | EMPTY |"
    else:
        line = (f"| {ts} | 4xA100(0124) | {name} | "
                f"{sr_strict/n*100:.1f}% | {sr_lenient/n*100:.1f}% | "
                f"{rw/n:.4f} | {n} | OK |")
print(line)
with open(results_log, "a") as fh: fh.write(line + "\n")
PYEOF
}

run_one() {
    local name=$1
    local env=$(env_of "$name")
    local config="${CFG_ROOT}/${env}/${name}.yaml"
    local short_hash=$(echo "$name" | md5sum | head -c 8)
    local ray_tmp="${RAY_TMPDIR}/r${short_hash}"

    if [ -f "$STOP_FILE" ]; then
        echo "[$(date '+%m-%d %H:%M')] STOP_FILE detected — exiting before $name"
        return 99
    fi
    if [ ! -f "$config" ]; then
        echo "  MISSING config: $config — skipping"
        return 1
    fi

    echo ""
    echo "=================================================="
    echo "[$(date '+%m-%d %H:%M')] PREP: $name (env=$env)"
    echo "=================================================="
    kill_ray_stragglers
    wait_for_gpu_clean
    restart_env "$env"
    mkdir -p "$ray_tmp"
    rm -rf "$ray_tmp"/session_* 2>/dev/null || true

    echo "[$(date '+%m-%d %H:%M')] RUN: $name"
    (CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
        python launcher.py --conf "$config" > "logs/${name}.log" 2>&1)
    local rc=$?
    echo "[$(date '+%m-%d %H:%M')] rc=$rc: $name"

    parse_and_log "$name"
    kill_ray_stragglers
    sleep 5
}

# ---------------------- main ----------------------

echo "=================================================="
echo " A100 REBUTTAL QUEUE  $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPUs: $GPUS"
echo " Stop switches: $STOP_FILE | $STOP_AFTER_MINIMAL"
echo "=================================================="

MINIMAL_QUEUE=(
    webshop_qwen1.5b_duet_a100_seed2025
    alfworld_qwen1.5b_duet_a100_obsnoise_soft
    webshop_qwen1.5b_duet_a100_seed2027
    alfworld_qwen1.5b_duet_a100_obsnoise_hash
)

EXTENDED_QUEUE=(
    webshop_qwen1.5b_sft_a100
    alfworld_qwen1.5b_duet_a100_soft_clean
    webshop_qwen1.5b_sft_rl_a100_seed2025
    webshop_qwen1.5b_sft_rl_a100_seed2027
    alfworld_qwen1.5b_duet_a100_teacher14b
    alfworld_qwen1.5b_duet_a100_teacher32b
)

for name in "${MINIMAL_QUEUE[@]}"; do
    run_one "$name"
    if [ "$?" = "99" ]; then echo "Queue stopped by STOP_FILE."; exit 0; fi
done

echo "" | tee -a "$RESULTS_LOG"
echo "=== MINIMAL SET COMPLETE $(date '+%Y-%m-%d %H:%M') — rebuttal writing can start ===" | tee -a "$RESULTS_LOG"

if [ -f "$STOP_AFTER_MINIMAL" ]; then
    echo "STOP_AFTER_MINIMAL set — not running extended phase."
    bash start_env_alfworld.sh stop 2>/dev/null || true
    bash start_env_webshop.sh stop 2>/dev/null || true
    exit 0
fi

for name in "${EXTENDED_QUEUE[@]}"; do
    # teacher runs need their cache (produced by run_a100_teacher_sampling.sh on GPU 6)
    case "$name" in
        *teacher14b) cache="data/teacher_trajectories/qwen14b/alfworld_qwen14b_filtered_react_tags.pkl" ;;
        *teacher32b) cache="data/teacher_trajectories/qwen32b/alfworld_qwen32b_filtered_react_tags.pkl" ;;
        *) cache="" ;;
    esac
    if [ -n "$cache" ] && [ ! -f "$cache" ]; then
        echo "SKIP $name: teacher cache not ready ($cache)"
        continue
    fi
    run_one "$name"
    if [ "$?" = "99" ]; then echo "Queue stopped by STOP_FILE."; exit 0; fi
    if [ "$name" = "webshop_qwen1.5b_sft_a100" ]; then
        if [ ! -d "checkpoints/agentevolver/webshop_qwen1.5b_sft_a100/global_step_50/actor_hf" ]; then
            echo "WARN: WS SFT ckpt missing — sft_rl runs will fail. Check logs/webshop_qwen1.5b_sft_a100.log"
        fi
    fi
done

bash start_env_alfworld.sh stop 2>/dev/null || true
bash start_env_webshop.sh stop 2>/dev/null || true
echo ""
echo "=================================================="
echo " A100 REBUTTAL QUEUE COMPLETE  $(date '+%Y-%m-%d %H:%M:%S')"
echo " Results: $RESULTS_LOG"
echo "=================================================="
