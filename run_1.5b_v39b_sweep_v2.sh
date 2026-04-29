#!/bin/bash
# ==============================================================================
# 1.5B v39b Sweep — v2 (WebShop SOTA hunt only)
#
# AF SOTA already locked in (v39_postfix=47.0%, v39c_postfix=47.5% beat
# DUET v1 32.5% by +14.5/+15pp). User decision (2026-04-29): drop AF sweep,
# focus all GPU on WS until WS SOTA breaks 22% (DUET v24).
#
# Plan B priority over remaining swA: swB cells are the predicted winners
# (valley=0.10/0.15 per analysis_reports/v39_vs_v24_webshop_diagnosis.md).
# If swB hits SOTA early, we can re-evaluate before chewing through the
# remaining swA cells (which are paper-ablation only, not SOTA candidates).
#
# Picks up from where v1 master was killed:
#   v1 master had completed: swA_10, swA_12, swA_11, swA_04 (4/9)
#   v1 master was running: swA_03_peak04 (21/100, killed)
#   This script: 3 swB Plan B (priority) + 5 remaining swA (ablation).
# ==============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
mkdir -p logs

# ---------- helpers (use locals to avoid the v1 i-pollution bug) ----------
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
    local env=$1   # alfworld | webshop
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
echo " 1.5B v39b Sweep — v2 (WS SOTA hunt)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# Make sure no AF env is up at the start
bash start_env_alfworld.sh stop 2>/dev/null || true
sleep 3

# Plan B (predicted winners) FIRST, then swA remaining (ablation).
WS_RUNS=(
    "sweep_1.5b/webshop_qwen1.5b_duet_swB_01_pk03_v10_ema02|swB_01_pk03_v10_ema02"
    "sweep_1.5b/webshop_qwen1.5b_duet_swB_02_pk03_v15_ema02|swB_02_pk03_v15_ema02"
    "sweep_1.5b/webshop_qwen1.5b_duet_swB_03_pk03_v10_ema01|swB_03_pk03_v10_ema01"
    "sweep_1.5b/webshop_qwen1.5b_duet_swA_03_peak04|swA_03_peak04"
    "sweep_1.5b/webshop_qwen1.5b_duet_swA_02_peak02|swA_02_peak02"
    "sweep_1.5b/webshop_qwen1.5b_duet_swA_05_peak06|swA_05_peak06"
    "sweep_1.5b/webshop_qwen1.5b_duet_swA_06_peak07|swA_06_peak07"
    "sweep_1.5b/webshop_qwen1.5b_duet_swA_08_ema08|swA_08_ema08"
)
WS_TOTAL=${#WS_RUNS[@]}

echo ""
echo "============================================"
echo " WebShop sweep v2: ${WS_TOTAL} runs"
echo "   3 swB Plan B (predicted winners) -> 5 swA remaining (ablation)"
echo "============================================"

idx=0
for entry in "${WS_RUNS[@]}"; do
    idx=$((idx+1))
    yaml_rel="${entry%|*}"
    name="webshop_qwen1.5b_duet_${entry##*|}"
    config="config/duet_paper_experiments_configs/webshop/${yaml_rel}.yaml"
    if [ ! -f "$config" ]; then
        echo "[$(date '+%m-%d %H:%M')] MISSING config: $config — skipping"
        continue
    fi
    run_one "webshop" "$config" "$name" "$idx" "$WS_TOTAL"
done

# Final cleanup
bash start_env_webshop.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " 1.5B v39b Sweep v2 (WS-only) complete"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
