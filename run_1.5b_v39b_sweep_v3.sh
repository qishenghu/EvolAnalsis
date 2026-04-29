#!/bin/bash
# ==============================================================================
# 1.5B v39b Sweep — v3 (Plan C: precision search around swB_01 = 21.5%)
#
# Context: swB_01 (peak=0.3, valley=0.10, d_floor=0.5, ema=0.2) hit 21.5% on
# val@100, just 0.5pp shy of v24 22% SOTA. swB_02 (valley=0.15) and swB_03
# (ema=0.1) collapsed (2-4%), so the sweet spot is narrow. Probe nearest
# neighbors of swB_01.
#
# v3 runs 3 swC cells (~2.5h each = ~7.5h total). After v3 finishes, also runs
# the remaining 5 swA ablation cells (paper material, not SOTA candidates).
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
    for _i in {1..30}; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>200' | wc -l)
        if [ "$used" = "0" ]; then echo "  GPU clean"; return 0; fi
        echo "  Waiting for GPU ($used busy)... ${_i}/30"; sleep 10
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
    local env=$1
    echo "  Restarting $env env service (memory leak prevention)..."
    bash "start_env_${env}.sh" stop 2>/dev/null || true
    sleep 8
    bash "start_env_${env}.sh"
    sleep 12
}

run_one() {
    local env=$1
    local config=$2
    local name=$3
    local idx=$4
    local total=$5
    local ray_tmp="${RAY_TMPDIR}"

    echo ""
    echo "============================================"
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] PREP: $name"
    echo "============================================"
    kill_ray_stragglers
    wait_for_gpu_clean
    restart_env "$env"
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
echo " 1.5B v39b Sweep — v3 (Plan C precision search)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# Plan C precision search FIRST (predicted highest-lift over swB_01),
# then the remaining swA ablation cells (paper material).
WS_RUNS=(
    "swC_01_pk03_v10_floor04"
    "swC_02_pk03_v10_floor06"
    "swC_03_pk03_v12_ema02"
    "swA_03_peak04"
    "swA_02_peak02"
    "swA_05_peak06"
    "swA_06_peak07"
    "swA_08_ema08"
)
WS_TOTAL=${#WS_RUNS[@]}

idx=0
for tag in "${WS_RUNS[@]}"; do
    idx=$((idx+1))
    name="webshop_qwen1.5b_duet_${tag}"
    config="config/duet_paper_experiments_configs/webshop/sweep_1.5b/${name}.yaml"
    if [ ! -f "$config" ]; then
        echo "[$(date '+%m-%d %H:%M')] MISSING config: $config — skipping"
        continue
    fi
    run_one "webshop" "$config" "$name" "$idx" "$WS_TOTAL"
done

bash start_env_webshop.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " 1.5B v39b Sweep v3 (Plan C) complete"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
