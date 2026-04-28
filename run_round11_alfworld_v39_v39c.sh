#!/bin/bash
# ==============================================================================
# Round 11 — ALFWorld retry for v39 + v39c only.
# v39b ALFWorld already done (42% val@50, 36% val@100).
# v39_postfix got stuck 47h on broken agentgym (resolved 04-28).
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

# Restart alfworld env between experiments to avoid agentgym degradation
restart_alfworld_env() {
    echo "  Restarting alfworld env (preventive)..."
    bash start_env_alfworld.sh stop 2>/dev/null || true
    sleep 8
    bash start_env_alfworld.sh
    sleep 10
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
echo " Round 11 ALFWorld v39 + v39c retry"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

if ! curl -s http://127.0.0.1:8081/healthz >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] env_service not up — starting..."
    bash start_env_alfworld.sh
    sleep 10
fi

TOTAL=2
run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39_postfix.yaml" \
    "alfworld_qwen1.5b_duet_v39_postfix" 1 $TOTAL

# Restart agentgym between experiments to avoid the 47h hang
restart_alfworld_env

run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39c_postfix.yaml" \
    "alfworld_qwen1.5b_duet_v39c_postfix" 2 $TOTAL

bash start_env_alfworld.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " Round 11 v39+v39c retry complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
