#!/bin/bash
# ==============================================================================
# DUET Round 9 — Phase 2 Local Continuation
#
# Phase 1 winner: v39b (disc_acc α=0.5), WebShop 1.5B = 19.0% success
#
# Round 9 goals:
#   1. v39b on ALFWorld 1.5B (~5h) — cross-env validation
#      Expected: success@100 ≥ 42% (v39 baseline; v24 was 30.5%)
#      Success criterion: ≥ DUET-v1 (32.5%)
#
#   2. v39c (d_floor=0.4) on WebShop 1.5B (~3h) — param sweep
#      Expected: closer to v24's 22.0% than v39b's 19.0%
#      Success criterion: ≥ 20% success (pushing past v39b)
#
# Total ~8h on GPU 0-3.
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
mkdir -p logs

run_experiment() {
    local config=$1
    local name=$2
    local idx=$3
    local total=$4
    local ray_tmp="${RAY_TMPDIR}/${name}"
    mkdir -p "$ray_tmp"
    rm -rf "$ray_tmp"/session_* 2>/dev/null || true

    echo ""
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] RUN: $name"
    CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
        python launcher.py --conf "$config" \
        > "logs/${name}.log" 2>&1
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] DONE: $name"
}

echo "============================================"
echo " DUET Round 9: Phase 2 local continuation"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPU: $GPUS"
echo "============================================"

TOTAL=2

# Experiment 1: v39b on ALFWorld 1.5B (~5h)
if ! curl -s http://127.0.0.1:8081 >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] Starting ALFWorld environment..."
    bash start_env_alfworld.sh
fi
run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39b.yaml" "alfworld_qwen1.5b_duet_v39b" 1 $TOTAL

# Experiment 2: v39c on WebShop 1.5B (~3h)
echo ""
echo "[$(date '+%m-%d %H:%M')] Switching env: ALFWorld -> WebShop"
bash start_env_alfworld.sh stop 2>/dev/null || true
sleep 10

if ! curl -s http://127.0.0.1:8083 >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] Starting WebShop environment..."
    bash start_env_webshop.sh
fi
run_experiment "config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v39c.yaml" "webshop_qwen1.5b_duet_v39c" 2 $TOTAL

bash start_env_webshop.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " Round 9 complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
