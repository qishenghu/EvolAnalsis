#!/bin/bash
# ==============================================================================
# DUET WebShop 1.5B Round 2 Ablation (12 variants, v10-v21)
#
# All based on v8 (best from Round 1: 0.574). Target: beat CHORD (0.603).
#
# Priority order (based on expected probability of beating CHORD):
#   Tier 1 (highest confidence):
#     v12: disc_temp=1.5 + clip_max=2.0 (combined DR3 stabilization)
#     v19: disc_temp=1.5 + clip=3.0 + beta_decay + KL=0.002 (stable combo)
#     v10: disc_temp=1.5 (softer DR3 discriminator)
#   Tier 2 (strong hypotheses):
#     v15: beta_decay + disc_temp=1.5 (dual curriculum)
#     v13: KL=0.003 + disc_temp=1.5 (late stabilization)
#     v16: gap_gate + disc_temp=1.5 (per-task adaptive)
#   Tier 3 (worth testing):
#     v11: clip_max=2.0 (reduce IS variance alone)
#     v17: policy_shaping_beta=0.05 (stronger down-weighting)
#     v20: warmup=20 + disc_temp=1.5 (more calibration)
#   Tier 4 (exploratory):
#     v21: grpo_decouple=false + disc_temp=1.5
#     v14: SC beta=0.15
#     v18: temperature=0.7
#
# Each ~4-5h on GPU 0-3, total ~50h for all 12.
# Run Tier 1+2 first (6 variants, ~25h).
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
echo " DUET WebShop 1.5B Round 2 (12 variants)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPU: $GPUS"
echo "============================================"

# Ensure WebShop env is running
if ! curl -s http://127.0.0.1:8083 >/dev/null 2>&1; then
    echo "Starting WebShop environment..."
    bash start_env_webshop.sh
fi

TOTAL=12

# --- Tier 1: Highest confidence ---
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v12.yaml" "webshop_qwen1.5b_duet_v12" 1 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v19.yaml" "webshop_qwen1.5b_duet_v19" 2 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v10.yaml" "webshop_qwen1.5b_duet_v10" 3 $TOTAL

# --- Tier 2: Strong hypotheses ---
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v15.yaml" "webshop_qwen1.5b_duet_v15" 4 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v13.yaml" "webshop_qwen1.5b_duet_v13" 5 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v16.yaml" "webshop_qwen1.5b_duet_v16" 6 $TOTAL

# --- Tier 3: Worth testing ---
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v11.yaml" "webshop_qwen1.5b_duet_v11" 7 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v17.yaml" "webshop_qwen1.5b_duet_v17" 8 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v20.yaml" "webshop_qwen1.5b_duet_v20" 9 $TOTAL

# --- Tier 4: Exploratory ---
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v21.yaml" "webshop_qwen1.5b_duet_v21" 10 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v14.yaml" "webshop_qwen1.5b_duet_v14" 11 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v18.yaml" "webshop_qwen1.5b_duet_v18" 12 $TOTAL

bash start_env_webshop.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " Round 2 complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
