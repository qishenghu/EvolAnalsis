#!/bin/bash
# ==============================================================================
# DUET WebShop 1.5B Round 4: Narrative Rescue (5 variants, v25-v29)
#
# Goal: test whether DR3 alone — without explicit BC/SFT — can match v24 (0.678)
#       by adjusting PPO trust region and w_hat variance.
#
# Hypothesis: v24's BC term compensates for (a) PPO clip over-restricting
#             rare-token updates, and (b) w_hat variance during discriminator
#             warmup. If so, widening off_cliprange_high and/or tightening
#             w_hat EMA should recover v24 performance without BC.
#
# Base: v12 (DR3 + SC, no BC), which got 0.431. v24 got 0.678 (+24.7pp).
#
# Variants:
#   v25: v12 + off_cliprange_high 0.6→2.0            (moderate clip widen)
#   v26: v12 + off_cliprange_high 0.6→5.0            (aggressive clip widen)
#   v27: v12 + off_cliprange_high 2.0 + clip_ratio_high 0.28→1.0  (both widened)
#   v28: v12 + w_hat_ema_alpha 0.3→0.1               (variance reduction only)
#   v29: v12 + off_cliprange_high 2.0 + clip_max 2.0→5.0 + ema_alpha 0.1
#                                                    (combined rescue)
#
# Decision rule:
#   - If v25 or v26 ≥ 0.60: clip widening alone works. Framing becomes
#     "DR3 + teacher-widened clip + SC" — no BC reference needed.
#   - If v28 ≥ 0.60: variance reduction works. Framing highlights w_hat EMA.
#   - If v29 only (combined) ≥ 0.60: both mechanisms contribute; need to
#     present as coupled fix.
#   - If all < 0.55: narrative rescue failed. Fall back to Option F+H (BC is
#     needed as a variance-reduction anchor).
#
# Each ~4-5h on GPU 0-3, total ~20-25h for all 5.
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
CONFIG_DIR="config/duet_paper_experiments_configs/webshop"
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
echo " DUET WebShop Round 4: Narrative Rescue (5 variants)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPU: $GPUS"
echo "============================================"

if ! curl -s http://127.0.0.1:8083 >/dev/null 2>&1; then
    echo "Starting WebShop environment..."
    bash start_env_webshop.sh
fi

TOTAL=5

# Priority order: test moderate widening first (cheapest hypothesis),
# then aggressive, then combined. v28 standalone last — if v25/v26 win,
# we don't care whether v28 alone works.
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v25.yaml" "webshop_qwen1.5b_duet_v25" 1 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v26.yaml" "webshop_qwen1.5b_duet_v26" 2 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v27.yaml" "webshop_qwen1.5b_duet_v27" 3 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v29.yaml" "webshop_qwen1.5b_duet_v29" 4 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v28.yaml" "webshop_qwen1.5b_duet_v28" 5 $TOTAL

echo ""
echo "============================================"
echo " Round 4 complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
