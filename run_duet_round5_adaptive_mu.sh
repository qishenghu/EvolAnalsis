#!/bin/bash
# ==============================================================================
# DUET Round 5: Adaptive μ from advantage variance + v24 ALFWorld generalization
#
# v37: WebShop 1.5B with adaptive μ (theory: advantage-regularizer mechanism)
#   - Base: v24 config (DR3 + SC + decaying BC)
#   - Change: chord_mu_adaptive=true, V_A_target=0.035
#   - Target: ≥ 0.65 (match or beat v24's 0.678)
#   - Theory source: analysis_reports/duet_second_pass_theory.md
#
# v24-alfworld-1.5b: v24 recipe on ALFWorld 1.5B (P0 generalization test)
#   - Base: alfworld_qwen1.5b_duet + v24's BC params (μ=0.3→0.05 over 25 steps)
#   - Target: NOT less than DUET v1 ALFWorld (32.5%). If gain or even, v24 recipe
#     is universal. If worse, BC is WebShop-specific and we re-frame.
#
# Run on GPU 0-3, sequentially (env switching: webshop → alfworld).
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
echo " DUET Round 5: Adaptive μ + v24-ALFWorld"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPU: $GPUS"
echo "============================================"

TOTAL=2

# Experiment 1: v37 on WebShop 1.5B (~3h)
if ! curl -s http://127.0.0.1:8083 >/dev/null 2>&1; then
    echo "Starting WebShop environment..."
    bash start_env_webshop.sh
fi
run_experiment "config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v37.yaml" "webshop_qwen1.5b_duet_v37" 1 $TOTAL

# Experiment 2: v24 on ALFWorld 1.5B (~5h)
echo ""
echo "[$(date '+%m-%d %H:%M')] Switching env: WebShop -> ALFWorld"
bash start_env_webshop.sh stop 2>/dev/null || true
sleep 5

if ! curl -s http://127.0.0.1:8081 >/dev/null 2>&1; then
    echo "Starting ALFWorld environment..."
    bash start_env_alfworld.sh
fi
run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v24.yaml" "alfworld_qwen1.5b_duet_v24" 2 $TOTAL

bash start_env_alfworld.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " Round 5 complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
