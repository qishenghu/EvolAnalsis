#!/bin/bash
# ==============================================================================
# DUET 3B Ablation Queue — A100 server takes over from L20X
#
# Runs 4 cells: AF/WS × {-SC, -BC}
# Priority: -SC first (L20X cannot finish), then -BC (backup).
#
# This script waits for the 1.5B ablation queue to finish before starting.
# Safe-stop switch: `touch logs/ABLATION_3B_STOP`
#
# Started: 2026-05-06
# ==============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

source "/data/home/qisheng/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
mkdir -p logs NeurIPS_2026_Latex/data
RESULTS_LOG="NeurIPS_2026_Latex/data/ablation_results.md"
STOP_FILE="logs/ABLATION_3B_STOP"

# ---------------------- helper functions ----------------------
# (same as 1.5B orchestrator)

wait_for_gpu_clean() {
    local _i; local our_gpus="${CUDA_GPUS:-0,1,2,3}"
    for _i in {1..30}; do
        local used
        used=$(nvidia-smi --id="$our_gpus" --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>30000' | wc -l)
        if [ "$used" = "0" ]; then echo "  GPU clean"; return 0; fi
        echo "  Waiting ($used busy)... ${_i}/30"; sleep 10
    done
    echo "  WARN: proceeding despite busy GPU"
}

kill_ray_stragglers() {
    local n
    n=$(ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | wc -l)
    if [ "$n" -gt 0 ]; then
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
        sleep 5
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
        sleep 3
    fi
}

restart_env() {
    local env=$1
    bash start_env_alfworld.sh stop 2>/dev/null || true
    bash start_env_webshop.sh stop 2>/dev/null || true
    sleep 8
    bash start_env_${env}.sh
    sleep 12
}

parse_and_log() {
    local name=$1; local env=$2; local mech=$3; local size=$4
    local val_log="experiments/${env}/${name}/validation_log/100.jsonl"
    python - <<PYEOF
import json, os, datetime
val_log = "${val_log}"; name = "${name}"; env = "${env}"
mech = "${mech}"; size = "${size}"
results_log = "${RESULTS_LOG}"
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

setting = f"{size}-{'AF' if env=='alfworld' else 'WS'}"

if not os.path.exists(val_log):
    line = f"| {ts} | 8xA100 | {setting} | -{mech} | — | — | — | — | val@100 MISSING — run failed or killed |"
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
        line = f"| {ts} | 8xA100 | {setting} | -{mech} | — | — | — | 0 | val@100 EMPTY |"
    else:
        line = (f"| {ts} | 8xA100 | {setting} | -{mech} | "
                f"{sr_strict/n*100:.1f}% | {sr_lenient/n*100:.1f}% | "
                f"{rw/n:.4f} | {n} | OK |")
print(line)
with open(results_log, "a") as fh: fh.write(line + "\n")
PYEOF
}

run_one() {
    local config=$1; local name=$2; local env=$3; local mech=$4; local size=$5
    local idx=$6; local total=$7

    if [ -f "$STOP_FILE" ]; then
        echo "[$(date '+%m-%d %H:%M')] STOP_FILE detected — exiting cleanly before [$idx/$total] $name"
        return 99
    fi

    local short_hash=$(echo "$name" | md5sum | head -c 8)
    local ray_tmp="${RAY_TMPDIR}/r${short_hash}"

    echo ""
    echo "=================================================="
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] PREP: $name"
    echo "  env=$env mech=$mech size=$size"
    echo "  config=$config"
    echo "  ray_tmp=$ray_tmp (hash=$short_hash)"
    echo "=================================================="

    kill_ray_stragglers
    wait_for_gpu_clean
    restart_env "$env"
    mkdir -p "$ray_tmp"
    rm -rf "$ray_tmp"/session_* 2>/dev/null || true

    if [ ! -f "$config" ]; then
        echo "  MISSING config: $config — skipping"
        return 1
    fi

    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] RUN: $name"
    (CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
        python launcher.py --conf "$config" > "logs/${name}.log" 2>&1)
    local rc=$?

    if [ $rc -eq 0 ]; then
        echo "[$(date '+%m-%d %H:%M')] [$idx/$total] DONE rc=0: $name"
    else
        echo "[$(date '+%m-%d %H:%M')] [$idx/$total] FAILED rc=$rc: $name"
    fi

    parse_and_log "$name" "$env" "$mech" "$size"
    kill_ray_stragglers
    sleep 5
}

# ---------------------- wait for 1.5B queue ----------------------

echo "=================================================="
echo " 3B ABLATION QUEUE — waiting for 1.5B to finish"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

while true; do
    if [ -f "$STOP_FILE" ]; then
        echo "STOP_FILE detected before start. Exiting."
        exit 0
    fi
    # Check if 1.5B orchestrator is still running
    if ps -ef | grep "run_ablation_neurips_2026_05_05" | grep -v grep | grep -v "$0" > /dev/null 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] 1.5B queue still running. Waiting 5 min..."
        sleep 300
    # Also check if main_ppo is running (the 1.5B training itself)
    elif ps -ef | grep "agentevolver.main_ppo" | grep -v grep > /dev/null 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] main_ppo still running. Waiting 5 min..."
        sleep 300
    else
        echo "[$(date '+%m-%d %H:%M')] 1.5B queue finished. Starting 3B ablations."
        sleep 30  # grace period for cleanup
        break
    fi
done

# ---------------------- queue ----------------------

QUEUE=(
    "config/duet_paper_experiments_configs/ablations_neurips/alfworld/alfworld_qwen3b_duet_minus_sc.yaml|alfworld_qwen3b_duet_minus_sc|alfworld|SC|3B"
    "config/duet_paper_experiments_configs/ablations_neurips/webshop/webshop_qwen3b_duet_minus_sc.yaml|webshop_qwen3b_duet_minus_sc|webshop|SC|3B"
    "config/duet_paper_experiments_configs/ablations_neurips/alfworld/alfworld_qwen3b_duet_minus_bc.yaml|alfworld_qwen3b_duet_minus_bc|alfworld|BC|3B"
    "config/duet_paper_experiments_configs/ablations_neurips/webshop/webshop_qwen3b_duet_minus_bc.yaml|webshop_qwen3b_duet_minus_bc|webshop|BC|3B"
)
TOTAL=${#QUEUE[@]}

echo ""
echo "=================================================="
echo " 8xA100 3B ABLATION QUEUE  $(date '+%Y-%m-%d %H:%M:%S')"
echo " Total: $TOTAL runs"
echo " Order: -SC(AF,WS) → -BC(AF,WS)"
echo " Stop switch: touch $STOP_FILE"
echo "=================================================="

idx=0
for entry in "${QUEUE[@]}"; do
    idx=$((idx+1))
    IFS='|' read -r cfg name env mech size <<< "$entry"
    run_one "$cfg" "$name" "$env" "$mech" "$size" "$idx" "$TOTAL"
    rc=$?
    if [ $rc -eq 99 ]; then
        echo "Queue stopped by STOP_FILE."
        break
    fi
done

# Final cleanup
bash start_env_webshop.sh stop 2>/dev/null || true
bash start_env_alfworld.sh stop 2>/dev/null || true
kill_ray_stragglers

echo ""
echo "=================================================="
echo " 3B ABLATION QUEUE COMPLETE  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
echo " Results: $RESULTS_LOG"
echo " Per-run logs: logs/<run_name>.log"
