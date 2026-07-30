#!/bin/bash
# ==============================================================================
# 3B WS Sweep v2 — Plan E (untested BC corners)
#
# Decision rationale (2026-05-02 ~01:15):
#   swD_01 (= 1.5B SOTA recipe, peak=0.3, valley=0.10, d_floor=0.6, ema=0.2)
#   gave only 29.5% on 3B — 1.5B winning direction did not transfer.
#   3B server's 12-cell sweep + our swD_01 cover almost all of the
#   (peak, valley, d_floor, ema) space — none broke 49%.
#
#   Two combos remain untested:
#     swE_01: peak=0.3, valley=0.05, d_floor=0.5, ema=0.1 (very slow EMA)
#     swE_02: peak=0.2, valley=0.10, d_floor=0.5, ema=0.5 (low peak + raised valley)
#
#   8h window → 2 cells × ~3:40 = 7:20. Fits.
# ==============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
mkdir -p logs

wait_for_gpu_clean() {
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
        echo "  Killing $n training Ray actors..."
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
echo " 3B WebShop sweep — Plan E (untested corners)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

bash start_env_alfworld.sh stop 2>/dev/null || true
sleep 3

WS_RUNS=(
    "swE_01_pk03_v05_ema01"   # very slow EMA
    "swE_02_pk02_v10"         # low peak + raised valley
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
echo " 3B WS Plan E complete"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
