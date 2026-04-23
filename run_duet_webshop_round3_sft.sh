#!/bin/bash
# ==============================================================================
# DUET WebShop 1.5B Round 3: DR3 + SC + mini-SFT Hybrid (3 variants)
#
# Key insight: CHORD's SFT provides unconditional token-level behavioral cloning.
# DUET's DR3 provides conditional trajectory-level gradient. Adding a small SFT
# term to DUET gives the best of both worlds.
#
# v22: DR3 + SC(traj) + mini-SFT(mu=0.05 constant)
# v23: DR3(temp=1.5,clip=2.0) + SC(traj) + mini-SFT(mu=0.1 constant)
# v24: DR3(temp=1.5,clip=2.0) + SC(traj) + decaying-SFT(mu=0.3→0.05)
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
echo " DUET + mini-SFT Hybrid (3 variants)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

if ! curl -s http://127.0.0.1:8083 >/dev/null 2>&1; then
    echo "Starting WebShop environment..."
    bash start_env_webshop.sh
fi

run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v22.yaml" "webshop_qwen1.5b_duet_v22" 1 3
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v23.yaml" "webshop_qwen1.5b_duet_v23" 2 3
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v24.yaml" "webshop_qwen1.5b_duet_v24" 3 3

echo ""
echo "============================================"
echo " Round 3 complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
