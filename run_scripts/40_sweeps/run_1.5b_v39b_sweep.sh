#!/bin/bash
# ==============================================================================
# 1.5B v39b Sweep — Phase A
#
# Plan (per analysis_reports/handoff/HANDOFF_1.5B_SERVER_2026-04-28.md, v2):
#   1. Wait for current v39+v39c AF retry (PID-marker: launcher.py running an
#      alfworld_qwen1.5b_duet_v39*_postfix config) to exit.
#      That run fills sweep cells #07 (ema02) and #09 (floor04) for AF.
#   2. WebShop sweep: 9 new cells (#02 #03 #04 #05 #06 #08 #10 #11 #12),
#      ordered by predicted win likelihood. ~3h × 9 = 27h.
#   3. ALFWorld sweep: 9 new cells, same order. ~5h × 9 = 45h.
#
# Per-run:
#   - Stop env service (alfworld:8081 / webshop:8083), wait, restart fresh.
#   - Wait for GPU to be clean before launching.
#   - Kill straggler training Ray actors before each run.
#   - Run launcher.py with config; log to logs/<exp_name>.log.
# ==============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
mkdir -p logs

# ---------- helpers ----------
wait_for_gpu_clean() {
    for i in {1..30}; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>200' | wc -l)
        if [ "$used" = "0" ]; then echo "  GPU clean"; return 0; fi
        echo "  Waiting for GPU ($used busy)... ${i}/30"; sleep 10
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
    local env=$1   # alfworld | webshop
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

# ---------- Step 0: wait for current v39+v39c AF retry to finish ----------
echo "============================================"
echo " 1.5B v39b Sweep — Phase A start"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

wait_for_prior_orchestrator() {
    # The prior orchestrator runs:
    #   bash run_round11_alfworld_v39_v39c.sh
    # which spawns:
    #   python launcher.py --conf .../alfworld_qwen1.5b_duet_v39_postfix.yaml
    #   python launcher.py --conf .../alfworld_qwen1.5b_duet_v39c_postfix.yaml
    echo "[$(date '+%m-%d %H:%M')] Waiting for any v39_postfix or v39c_postfix launcher to exit..."
    while pgrep -f "launcher.py.*alfworld_qwen1.5b_duet_v39[c_]*_postfix" > /dev/null; do
        sleep 120
    done
    # And also wait for the orchestrator script itself
    while pgrep -f "run_round11_alfworld_v39_v39c.sh" > /dev/null; do
        sleep 60
    done
    echo "[$(date '+%m-%d %H:%M')] Prior orchestrator (v39+v39c AF retry) exited."
}
wait_for_prior_orchestrator

# Give env services and GPU 60s to settle
sleep 60

# ---------- Step 1: WebShop sweep (9 configs, ~27h) ----------
# Order: predicted win likelihood (slow EMA + strong BC first)
WS_SWEEP=(
    "swA_10_pk05_ema02"       # peak=0.5, slow EMA — 1.5B winner candidate
    "swA_12_pk05_ema02_v10"   # full combo: 0.5/0.10/slow
    "swA_11_pk05_v10"         # strong BC + high floor (fast EMA — risk)
    "swA_04_peak05"           # peak=0.5, default EMA
    "swA_03_peak04"           # peak=0.4, default EMA
    "swA_02_peak02"           # peak=0.2 (low BC, control)
    "swA_05_peak06"           # peak=0.6 (very strong BC)
    "swA_06_peak07"           # peak=0.7 (extreme)
    "swA_08_ema08"            # fast EMA control
)
WS_TOTAL=${#WS_SWEEP[@]}

echo ""
echo "============================================"
echo " WebShop sweep: ${WS_TOTAL} runs starting"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# Make sure ALFWorld env is stopped before WS sweep
bash start_env_alfworld.sh stop 2>/dev/null || true
sleep 5

i=0
for tag in "${WS_SWEEP[@]}"; do
    i=$((i+1))
    name="webshop_qwen1.5b_duet_${tag}"
    config="config/duet_paper_experiments_configs/webshop/sweep_1.5b/${name}.yaml"
    if [ ! -f "$config" ]; then
        echo "[$(date '+%m-%d %H:%M')] MISSING config: $config — skipping"
        continue
    fi
    run_one "webshop" "$config" "$name" "$i" "$WS_TOTAL"
done

# Stop WS env between phases
bash start_env_webshop.sh stop 2>/dev/null || true
sleep 10

# ---------- Step 2: ALFWorld sweep (9 configs, ~45h) ----------
AF_SWEEP=(
    "swA_10_pk05_ema02"
    "swA_12_pk05_ema02_v10"
    "swA_11_pk05_v10"
    "swA_04_peak05"
    "swA_03_peak04"
    "swA_02_peak02"
    "swA_05_peak06"
    "swA_06_peak07"
    "swA_08_ema08"
)
AF_TOTAL=${#AF_SWEEP[@]}

echo ""
echo "============================================"
echo " ALFWorld sweep: ${AF_TOTAL} runs starting"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

i=0
for tag in "${AF_SWEEP[@]}"; do
    i=$((i+1))
    name="alfworld_qwen1.5b_duet_${tag}"
    config="config/duet_paper_experiments_configs/alfworld/sweep_1.5b/${name}.yaml"
    if [ ! -f "$config" ]; then
        echo "[$(date '+%m-%d %H:%M')] MISSING config: $config — skipping"
        continue
    fi
    run_one "alfworld" "$config" "$name" "$i" "$AF_TOTAL"
done

bash start_env_alfworld.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " 1.5B v39b Sweep — Phase A complete"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
