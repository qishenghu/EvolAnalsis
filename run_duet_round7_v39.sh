#!/bin/bash
# ==============================================================================
# DUET Round 7: v39 — Adaptive μ based on discriminator accuracy
#
# Signal: μ_t tracks (1 - EMA(dr3/disc_acc))
#   - Offline reproduces v24's hand-tuned schedule with r=0.97, MAE=0.007
#   - Self-adjusts on ALFWorld (μ pins at μ_min by step 25)
#   - Based on empirical signal discovery in:
#     analysis_reports/adaptive_signal_discovery.md
#     analysis_reports/duet_third_pass_adaptive.md
#
# Rule: μ = clamp(μ_max · max(0, (1-d)/(1-d_floor)), μ_min, μ_max)
#   where d = EMA(alpha=0.2) of dr3/disc_acc, d_floor=0.5, μ_max=0.3, μ_min=0.05
#
# Expected results:
#   WebShop 1.5B: μ trajectory ≈ v24's (0.3→0.05 over ~25 steps) → Val@100 ≈ v24's 0.678
#   ALFWorld 1.5B: μ quickly → μ_min (templates easy) → preserve v1 behavior, avoid
#                  v24's -2pp Val@100 regression
#
# Success criteria:
#   - WebShop v39 Val@100 ≥ 0.65 (match v24)
#   - ALFWorld v39 Val@100 ≥ 32.5% (match DUET-v1, better than v24's 30.5%)
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
echo " DUET Round 7: v39 (adaptive μ via disc_acc)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPU: $GPUS"
echo "============================================"

TOTAL=2

# Experiment 1: v39 WebShop 1.5B (~3h)
if ! curl -s http://127.0.0.1:8083 >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] Starting WebShop environment..."
    bash start_env_webshop.sh
fi
run_experiment "config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v39.yaml" "webshop_qwen1.5b_duet_v39" 1 $TOTAL

# Experiment 2: v39 ALFWorld 1.5B (~5h)
echo ""
echo "[$(date '+%m-%d %H:%M')] Switching env: WebShop -> ALFWorld"
bash start_env_webshop.sh stop 2>/dev/null || true
sleep 10

if ! curl -s http://127.0.0.1:8081 >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] Starting ALFWorld environment..."
    bash start_env_alfworld.sh
fi
run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39.yaml" "alfworld_qwen1.5b_duet_v39" 2 $TOTAL

bash start_env_alfworld.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " Round 7 complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
