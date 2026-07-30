#!/bin/bash
# ==============================================================================
# DUET WebShop 1.5B Ablation Study (7 variants)
#
# Root cause: DUET learns to search+buy but NOT to select options (color/size).
# CHORD's SFT loss directly teaches option clicking at token level.
# DR3's trajectory-level gradient doesn't specifically teach option selection.
#
# Priority order:
#   v9: DUET-Lite (LUFFY shaping + trajectory SC, no DR3/step-deltas) — most promising
#   v7: No DR3 (LUFFY shaping + full SC with step deltas)
#   v4: No SC (pure DR3) — isolate DR3's contribution
#   v8: No step-level deltas (keep DR3 + trajectory SC)
#   v3: Stronger KL (0.001→0.01) — test policy drift
#   v5: No teacher baseline separation — test advantage distortion
#   v6: Combo (SC beta=0.05 + KL=0.005 + temp=0.5)
#
# Each ~4-5h on GPU 0-3, total ~30h
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
echo " DUET WebShop 1.5B Ablation (7 variants)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPU: $GPUS"
echo "============================================"

# Ensure WebShop env is running
if ! curl -s http://127.0.0.1:8083 >/dev/null 2>&1; then
    echo "Starting WebShop environment..."
    bash start_env_webshop.sh
fi

run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v9.yaml" "webshop_qwen1.5b_duet_v9" 1 7
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v7.yaml" "webshop_qwen1.5b_duet_v7" 2 7
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v4.yaml" "webshop_qwen1.5b_duet_v4" 3 7
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v8.yaml" "webshop_qwen1.5b_duet_v8" 4 7
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v3.yaml" "webshop_qwen1.5b_duet_v3" 5 7
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v5.yaml" "webshop_qwen1.5b_duet_v5" 6 7
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v6.yaml" "webshop_qwen1.5b_duet_v6" 7 7

bash start_env_webshop.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " Ablation complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
