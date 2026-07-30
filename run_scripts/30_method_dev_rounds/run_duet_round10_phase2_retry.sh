#!/bin/bash
# ==============================================================================
# DUET Round 10 — Phase 2 Retry with OOM Prevention
#
# Previous Round 9 crashed:
#   v39b ALFWorld — step 99/100 OK (Val@50=40.0%), crashed at Val@100 (vLLM OOM)
#   v39c WebShop — never ran (orchestrator set -e aborted after v39b crash)
#
# This round's fixes:
#   1. gpu_memory_utilization: 0.75 → 0.70 (buffer for vLLM wake-up)
#   2. Clean Ray state before each run
#   3. NO `set -e` — one failure doesn't abort subsequent experiments
#   4. Wait for GPU to be clean before each launch
#
# Sequential runs on GPU 0-3:
#   1. v39c WebShop 1.5B (~3h) — d_floor=0.4 param sweep, push past 20%
#   2. v39b ALFWorld 1.5B (~5h) — rerun to capture Val@100
#
# Success criteria:
#   v39c WebShop: success@100 ≥ 20% (push past v39b's 19.0%)
#   v39b ALFWorld: success@100 ≥ 32.5% (DUET-v1 baseline); ideally ≥ 42% (v39)
# ==============================================================================

# NO set -e — we want to continue even if one experiment fails
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
mkdir -p logs

wait_for_gpu_clean() {
    # Wait until all GPUs show < 200 MiB used (no lingering processes)
    for i in {1..30}; do
        local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>200' | wc -l)
        if [ "$used" = "0" ]; then
            echo "  GPU clean"
            return 0
        fi
        echo "  Waiting for GPU to clear ($used GPUs still busy)... ${i}/30"
        sleep 10
    done
    echo "  WARN: GPU not fully clean after 300s, proceeding anyway"
}

kill_ray_stragglers() {
    # Kill any stale ray workers from previous runs
    local n=$(ps -ef | grep -E "ray::|raylet" | grep -v grep | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "  Killing $n stale Ray processes..."
        ps -ef | grep -E "ray::|raylet" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
        sleep 5
        ps -ef | grep -E "ray::|raylet" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
        sleep 3
    fi
}

run_experiment() {
    local config=$1
    local name=$2
    local idx=$3
    local total=$4
    local ray_tmp="${RAY_TMPDIR}/${name}"

    echo ""
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] PREP: $name"

    # Pre-flight: clean state
    kill_ray_stragglers
    wait_for_gpu_clean

    # Fresh Ray temp dir
    rm -rf "$ray_tmp" 2>/dev/null || true
    mkdir -p "$ray_tmp"

    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] RUN: $name"
    # `set +e` scope: don't abort orchestrator on experiment failure
    (
        CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
            python launcher.py --conf "$config" \
            > "logs/${name}.log" 2>&1
    )
    local rc=$?
    if [ "$rc" = "0" ]; then
        echo "[$(date '+%m-%d %H:%M')] [$idx/$total] DONE: $name"
    else
        echo "[$(date '+%m-%d %H:%M')] [$idx/$total] FAILED (rc=$rc): $name — continuing orchestrator"
    fi

    # Post-flight: kill any stragglers before next experiment
    kill_ray_stragglers
    sleep 5
}

echo "============================================"
echo " DUET Round 10: Phase 2 retry"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPU: $GPUS"
echo " Changes: gpu_mem=0.70, no set -e, pre-flight GPU clean"
echo "============================================"

TOTAL=2

# Experiment 1: v39c on WebShop 1.5B (~3h) — d_floor=0.4 sweep
if ! curl -s http://127.0.0.1:8083 >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] Starting WebShop environment..."
    bash start_env_webshop.sh
fi
run_experiment "config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v39c.yaml" "webshop_qwen1.5b_duet_v39c" 1 $TOTAL

# Experiment 2: v39b on ALFWorld 1.5B (~5h) — rerun for Val@100
echo ""
echo "[$(date '+%m-%d %H:%M')] Switching env: WebShop -> ALFWorld"
bash start_env_webshop.sh stop 2>/dev/null || true
sleep 10

if ! curl -s http://127.0.0.1:8081 >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] Starting ALFWorld environment..."
    bash start_env_alfworld.sh
fi
run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39b.yaml" "alfworld_qwen1.5b_duet_v39b" 2 $TOTAL

bash start_env_alfworld.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " Round 10 complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
