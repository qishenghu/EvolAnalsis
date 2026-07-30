#!/bin/bash
# ==============================================================================
# Round 11 follow-up: experiments 2-6 after v39b webshop test passes.
#
# v39b webshop is already running as a smoke-test of the env_service shared-actor
# fix (2026-04-25). When it exits, this script picks up the remaining 5:
#   2. v39_postfix  WebShop  (α=0.2)
#   3. v39c_postfix WebShop  (d_floor=0.4)
#   4. v39b_postfix ALFWorld
#   5. v39_postfix  ALFWorld
#   6. v39c_postfix ALFWorld
#
# Resilience: same orchestrator pattern as run_duet_round11_postfix_full.sh.
# ==============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
mkdir -p logs

wait_for_gpu_clean() {
    for i in {1..30}; do
        local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>200' | wc -l)
        if [ "$used" = "0" ]; then echo "  GPU clean"; return 0; fi
        echo "  Waiting for GPU ($used busy)... ${i}/30"; sleep 10
    done
    echo "  WARN: GPU not fully clean after 300s, proceeding anyway"
}

kill_ray_stragglers() {
    local n=$(ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "  Killing $n training Ray actors (env_service raylet preserved)..."
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
        sleep 5
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
        sleep 3
    fi
}

run_experiment() {
    local config=$1
    local name=$2
    local idx=$3
    local total=$4
    local ray_tmp="${RAY_TMPDIR}"

    echo ""
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] PREP: $name"
    kill_ray_stragglers
    wait_for_gpu_clean
    mkdir -p "$ray_tmp"

    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] RUN: $name"
    (
        CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
            python launcher.py --conf "$config" \
            > "logs/${name}.log" 2>&1
    )
    local rc=$?
    if [ "$rc" = "0" ]; then
        echo "[$(date '+%m-%d %H:%M')] [$idx/$total] DONE: $name"
    else
        echo "[$(date '+%m-%d %H:%M')] [$idx/$total] FAILED (rc=$rc): $name — continuing"
    fi
    kill_ray_stragglers
    sleep 5
}

echo "============================================"
echo " Round 11 follow-up (5 experiments after v39b smoke-test)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# Wait for v39b smoke-test launcher to finish
echo ""
echo "[$(date '+%m-%d %H:%M')] Waiting for v39b WebShop smoke-test launcher to exit..."
while pgrep -f "launcher.py.*webshop_qwen1.5b_duet_v39b_postfix" > /dev/null; do
    sleep 60
done
echo "[$(date '+%m-%d %H:%M')] v39b smoke-test exited. Starting follow-up sweep."

TOTAL=5

# ---- WebShop continued ----
if ! curl -s http://127.0.0.1:8083 >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] WebShop env_service down — restarting..."
    bash start_env_webshop.sh
    sleep 10
fi

run_experiment "config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v39_postfix.yaml" \
    "webshop_qwen1.5b_duet_v39_postfix" 1 $TOTAL
run_experiment "config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v39c_postfix.yaml" \
    "webshop_qwen1.5b_duet_v39c_postfix" 2 $TOTAL

# ---- Switch env: WebShop → ALFWorld ----
echo ""
echo "[$(date '+%m-%d %H:%M')] Switching env: WebShop -> ALFWorld"
bash start_env_webshop.sh stop 2>/dev/null || true
sleep 10
if ! curl -s http://127.0.0.1:8081 >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] Starting ALFWorld environment..."
    bash start_env_alfworld.sh
fi
sleep 5

run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39b_postfix.yaml" \
    "alfworld_qwen1.5b_duet_v39b_postfix" 3 $TOTAL
run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39_postfix.yaml" \
    "alfworld_qwen1.5b_duet_v39_postfix" 4 $TOTAL
run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39c_postfix.yaml" \
    "alfworld_qwen1.5b_duet_v39c_postfix" 5 $TOTAL

bash start_env_alfworld.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " Round 11 follow-up complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
