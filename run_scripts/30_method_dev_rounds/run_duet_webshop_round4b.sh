#!/bin/bash
# ==============================================================================
# DUET WebShop 1.5B Round 4B: Narrative Rescue Extended (7 variants)
#
# v25 findings (catastrophic divergence):
#   - off_cliprange_high=2.0 killed policy at step 98 via grammar-token drift
#   - Root cause: BC's role is FORMAT STABILIZATION (not support lift)
#   - PPO teacher clip was never binding in v12 — hypothesis falsified
#
# v26 findings (same pattern, earlier):
#   - off_cliprange_high=5.0 crashed at step 67 (30 steps earlier than v25)
#
# New hypothesis: can we substitute BC's grammar-stabilizing role with other
# less invasive regularizers, preserving clean DR3 + SC dual-channel?
#
# Variants (in execution priority):
#   v29: v12 + off=2.0 + ema=0.1 + clip_max=5.0   (combined rescue, coin flip)
#   v30: v12 + kl_loss_coef 0.001→0.01           (strong KL to ref; cleanest)
#   v33: v12 + disc_temperature 1.5→3.0          (softer disc, lower w_hat var)
#   v28: v12 + w_hat_ema_alpha 0.3→0.1           (variance reduction control)
#   v36: v12 + const tiny BC μ=0.05 flat         (minimal-dose BC test)
#   v31: v12 + entropy_coeff 0→0.01              (explicit exploration)
#   v32: v12 + lr 1e-6→5e-7                      (engineering fix)
#
# Decision rules:
#   - Any variant ≥ 0.60: publishable narrative-rescue candidate
#   - v30 ≥ 0.60: WIN — clean DR3+SC+KL story, completely avoids BC
#   - v36 best among survivors: min-BC story ("DR3+SC+tiny-BC-anchor")
#   - All < 0.45: BC is genuinely irreducible for v24-level performance
#
# Kill protocol for each run:
#   duet/adv_teacher_effective_mean > 0.45 for 3 consecutive steps → KILL
#   dr3/disc_acc > 0.999 before step 80 → WATCH (high collapse risk)
#
# Each ~3h on GPU 0-3, total ~21h for all 7.
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
echo " DUET WebShop Round 4B: Narrative Rescue Extended (7 variants)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPU: $GPUS"
echo "============================================"

if ! curl -s http://127.0.0.1:8083 >/dev/null 2>&1; then
    echo "Starting WebShop environment..."
    bash start_env_webshop.sh
fi

TOTAL=7

# Priority order: v29 first (possibly best), v30 second (cleanest narrative if works).
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v29.yaml" "webshop_qwen1.5b_duet_v29" 1 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v30.yaml" "webshop_qwen1.5b_duet_v30" 2 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v33.yaml" "webshop_qwen1.5b_duet_v33" 3 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v28.yaml" "webshop_qwen1.5b_duet_v28" 4 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v36.yaml" "webshop_qwen1.5b_duet_v36" 5 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v31.yaml" "webshop_qwen1.5b_duet_v31" 6 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v32.yaml" "webshop_qwen1.5b_duet_v32" 7 $TOTAL

echo ""
echo "============================================"
echo " Round 4B complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
