#!/bin/bash
# ==============================================================================
# 3B WebShop Sweep — port 1.5B winning recipe
#
# 1.5B SOTA: swC_02 (peak=0.3, valley=0.10, d_floor=0.6, ema=0.2) → 36.0%
# 3B target: beat LUFFY 49.5% on val@100 success_rate.
# 3B handoff said v_no_bc_ws=1% / DUET v1=53% — high variance on WS-3B.
#
# 4 cells, ~5h each → ~20h total. Per-run env restart (memory leak prevention).
# ==============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
mkdir -p logs

wait_for_gpu_clean() {
    # Only check the GPUs we'll actually use (CUDA_GPUS), not 4-7 which may be
    # in use by another tenant on this multi-tenant box.
    local _i
    local our_gpus="${CUDA_GPUS:-0,1,2,3}"
    for _i in {1..30}; do
        local used
        used=$(nvidia-smi --id="$our_gpus" --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>200' | wc -l)
        if [ "$used" = "0" ]; then echo "  GPU clean (checked: $our_gpus)"; return 0; fi
        echo "  Waiting for GPU $our_gpus ($used busy)... ${_i}/30"; sleep 10
    done
    echo "  WARN: GPU not fully clean after 300s, proceeding anyway"
}

kill_ray_stragglers() {
    local n
    n=$(ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "  Killing $n training Ray actors (env_service raylet preserved)..."
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
        sleep 5
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
        sleep 3
    fi
}

restart_env() {
    echo "  Restarting webshop env service (memory leak prevention)..."
    bash start_env_webshop.sh stop 2>/dev/null || true
    sleep 8
    bash start_env_webshop.sh
    sleep 12
}

run_one() {
    local config=$1
    local name=$2
    local idx=$3
    local total=$4
    local ray_tmp="${RAY_TMPDIR}"

    echo ""
    echo "============================================"
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] PREP: $name"
    echo "============================================"
    kill_ray_stragglers
    wait_for_gpu_clean
    restart_env
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
echo " 3B WebShop sweep — port 1.5B SOTA recipe"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# Make sure no AF env is up
bash start_env_alfworld.sh stop 2>/dev/null || true
sleep 3

# Order: 1.5B SOTA exact port FIRST, then variations
WS_RUNS=(
    "swD_01_pk03_v10_floor06"   # = 1.5B swC_02 (36%)
    "swD_02_pk03_v10_floor07"   # push d_floor higher
    "swD_03_pk03_v10_ema02"     # = 1.5B swB_01 (21.5%)
    "swD_04_pk03_v15_floor06"   # more BC valley
)
WS_TOTAL=${#WS_RUNS[@]}

idx=0
for tag in "${WS_RUNS[@]}"; do
    idx=$((idx+1))
    name="webshop_qwen3b_duet_${tag}"
    config="config/duet_paper_experiments_configs/webshop/sweep_3b/${name}.yaml"
    if [ ! -f "$config" ]; then
        echo "[$(date '+%m-%d %H:%M')] MISSING config: $config — skipping"
        continue
    fi
    run_one "$config" "$name" "$idx" "$WS_TOTAL"
done

bash start_env_webshop.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " 3B WebShop sweep complete"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
