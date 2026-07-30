#!/bin/bash
# ============================================================================
# NeurIPS 2026 rebuttal experiment queue — H200 server (2 GPUs)
#
# Runs the ALFWorld-1.5B rebuttal experiments sequentially on GPUs 0,1.
#
# MINIMAL phase (finish first -> enough for an effective rebuttal):
#   M1. duet_h200_seed2026   — cross-infra replication of the paper run (47.5%)
#       >>> ALSO the infra smoke test: if this lands far from 47.5%, STOP
#           and investigate before burning time on the rest of the queue.
#   M2. duet_h200_seed2025   — DUET seed replicate
#   M3. duet_h200_seed2027   — DUET seed replicate (=> AF DUET 3-seed complete)
#   M4. duet_h200_cache10    — teacher cache 10% ablation (headline cache cell)
#
# EXTENDED phase (continues unless logs/H200_STOP_AFTER_MINIMAL exists):
#   E1. sft_h200             — regenerate SFT ckpt (step 50) + SFT curve (bDeY)
#   E2. sft_rl_h200_seed2025 — SFT+GRPO seed replicate (needs ckpt from E1)
#   E3. sft_rl_h200_seed2027 — SFT+GRPO seed replicate
#   E4. duet_h200_cache1     — teacher cache 1% ablation
#   E5. duet_h200_ntch2      — OPTIONAL mixing-ratio (2 teacher/group); last
#
# Results appended to NeurIPS_2026_Latex/data/h200_rebuttal_results.md
# Stop gracefully:      touch logs/H200_REBUTTAL_STOP   (takes effect between runs)
# Stop after minimal:   touch logs/H200_STOP_AFTER_MINIMAL
# Launch:               nohup bash run_h200_rebuttal_queue.sh > logs/h200_queue.log 2>&1 &
# ============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Single-instance guard. Two concurrent queues fight over the same GPUs and env
# service ports and silently corrupt each other's runs. A PID file is used rather
# than flock because the env services this script starts inherit the lock file
# descriptor and would keep an flock alive after the queue itself exits.
mkdir -p logs
QUEUE_PIDFILE=logs/.h200_queue.pid
if [ -f "$QUEUE_PIDFILE" ] && kill -0 "$(cat "$QUEUE_PIDFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: another H200 rebuttal queue is already running (PID $(cat "$QUEUE_PIDFILE"))."
    echo "       Check: pgrep -fa 'run_h200_rebuttal_queue.sh'"
    exit 1
fi
echo $$ > "$QUEUE_PIDFILE"
trap 'rm -f "$QUEUE_PIDFILE"' EXIT

source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="0,1"
N_GPUS=2
CFG_DIR="config/duet_paper_experiments_configs/rebuttal_neurips/alfworld"
RESULTS_LOG="NeurIPS_2026_Latex/data/h200_rebuttal_results.md"
STOP_FILE="logs/H200_REBUTTAL_STOP"
mkdir -p logs NeurIPS_2026_Latex/data

if [ ! -f "$RESULTS_LOG" ]; then
    cat > "$RESULTS_LOG" <<'EOF'
# H200 rebuttal results (ALFWorld 1.5B)

Reference numbers (paper, 4xA100, seed 2026): DUET 47.5% | SFT+GRPO 30.0% | CHORD 27.0% | LUFFY 5.5% | GRPO 1.0%
Metric: val@100 strict SR (score >= 1.0), 200 tasks.

| finished | host | experiment | strict SR | lenient SR | mean reward | n | status |
|---|---|---|---|---|---|---|---|
EOF
fi

# ---------------------- helpers ----------------------

kill_ray_stragglers() {
    # Scoped kill: only Ray worker processes from our own trainings.
    # NEVER use broad pkill patterns here.
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
        used=$(nvidia-smi --id=0,1 --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>20000' | wc -l)
        if [ "$used" = "0" ]; then echo "  GPUs 0,1 clean"; return 0; fi
        echo "  Waiting for GPUs 0,1 ($used busy)... ${_i}/60"; sleep 30
    done
    echo "  WARN: timed out waiting; proceeding anyway"
}

restart_env() {
    bash start_env_alfworld.sh stop 2>/dev/null || true
    sleep 8
    bash start_env_alfworld.sh
    sleep 12
}

parse_and_log() {
    local name=$1
    local val_log="experiments/alfworld/${name}/validation_log/100.jsonl"
    python - <<PYEOF
import json, os, datetime
val_log = "${val_log}"; name = "${name}"
results_log = "${RESULTS_LOG}"
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
if not os.path.exists(val_log):
    line = f"| {ts} | 2xH200 | {name} | - | - | - | - | val@100 MISSING |"
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
        line = f"| {ts} | 2xH200 | {name} | - | - | - | 0 | EMPTY |"
    else:
        line = (f"| {ts} | 2xH200 | {name} | "
                f"{sr_strict/n*100:.1f}% | {sr_lenient/n*100:.1f}% | "
                f"{rw/n:.4f} | {n} | OK |")
print(line)
with open(results_log, "a") as fh: fh.write(line + "\n")
PYEOF
}

run_one() {
    local name=$1
    local config="${CFG_DIR}/${name}.yaml"
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
    echo "[$(date '+%m-%d %H:%M')] PREP: $name"
    echo "=================================================="
    kill_ray_stragglers
    wait_for_gpu_clean
    restart_env
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

STOP_AFTER_MINIMAL="logs/H200_STOP_AFTER_MINIMAL"

MINIMAL_QUEUE=(
    alfworld_qwen1.5b_duet_h200_seed2026
    alfworld_qwen1.5b_duet_h200_seed2025
    alfworld_qwen1.5b_duet_h200_seed2027
    alfworld_qwen1.5b_duet_h200_cache10
)

EXTENDED_QUEUE=(
    alfworld_qwen1.5b_sft_h200
    alfworld_qwen1.5b_sft_rl_h200_seed2025
    alfworld_qwen1.5b_sft_rl_h200_seed2027
    alfworld_qwen1.5b_duet_h200_cache1
    alfworld_qwen1.5b_duet_h200_ntch2
)

echo "=================================================="
echo " H200 REBUTTAL QUEUE  $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPUs: $GPUS | minimal=${#MINIMAL_QUEUE[@]} + extended=${#EXTENDED_QUEUE[@]}"
echo " Stop switches: $STOP_FILE | $STOP_AFTER_MINIMAL"
echo "=================================================="

for name in "${MINIMAL_QUEUE[@]}"; do
    run_one "$name"
    if [ "$?" = "99" ]; then echo "Queue stopped by STOP_FILE."; exit 0; fi
done

echo "" | tee -a "$RESULTS_LOG"
echo "=== MINIMAL SET COMPLETE $(date '+%Y-%m-%d %H:%M') — report these numbers now ===" | tee -a "$RESULTS_LOG"

if [ -f "$STOP_AFTER_MINIMAL" ]; then
    echo "STOP_AFTER_MINIMAL set — not running extended phase."
    bash start_env_alfworld.sh stop 2>/dev/null || true
    exit 0
fi

for name in "${EXTENDED_QUEUE[@]}"; do
    run_one "$name"
    if [ "$?" = "99" ]; then echo "Queue stopped by STOP_FILE."; exit 0; fi
    # Guard: SFT+GRPO runs need the SFT checkpoint from alfworld_qwen1.5b_sft_h200
    if [ "$name" = "alfworld_qwen1.5b_sft_h200" ]; then
        if [ ! -d "checkpoints/agentevolver/alfworld_qwen1.5b_sft_h200/global_step_50/actor_hf" ]; then
            echo "WARN: SFT checkpoint missing after SFT run — sft_rl runs will fail."
            echo "      Check logs/alfworld_qwen1.5b_sft_h200.log before continuing."
        fi
    fi
done

bash start_env_alfworld.sh stop 2>/dev/null || true
echo ""
echo "=================================================="
echo " H200 REBUTTAL QUEUE COMPLETE  $(date '+%Y-%m-%d %H:%M:%S')"
echo " Results: $RESULTS_LOG"
echo "=================================================="
