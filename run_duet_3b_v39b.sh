#!/bin/bash
# ==============================================================================
# DUET 3B — v39b (adaptive μ via disc_acc, fast EMA α=0.5)
#
# For deployment on REMOTE 8xA100 server.
#
# v39b is the Phase 1 empirical winner on WebShop 1.5B:
#   Val@100 success = 19.0% (v24 hand-tuned: 22.0%, CHORD: 11.5%)
#   Val@100 reward = 0.637 (v24: 0.678)
#   MAE from v24 schedule = 0.013
#
# Theoretical framing: DR3's discriminator D(s,a) is a sufficient statistic for
# both density-ratio correction (w = D/(1-D)) AND adaptive BC schedule
# (μ = μ_max·(1-acc(D))/(1-d_floor)). See:
#   analysis_reports/theory_empirics_reconciliation.md
#   analysis_reports/phase1_deep_dive.md
#
# Runs TWO experiments sequentially:
#   1. v39b on WebShop 3B (~8-10h)
#   2. v39b on ALFWorld 3B (~8-10h)
#
# Expected results:
#   WebShop 3B: success@100 ≥ v24-3B baseline
#   ALFWorld 3B: success@100 > v24-3B (based on v39's +11.5pp 1.5B gain)
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

# 3B runs: use all 8 GPUs (n_gpus_per_node=4 in config implies 4-GPU slot but
# adjust here if 3B needs larger TP). Defaults assume 8xA100-80G.
GPUS="${CUDA_GPUS:-0,1,2,3,4,5,6,7}"
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
echo " DUET 3B Cross-scale: v39b"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPU: $GPUS"
echo "============================================"

TOTAL=2

# Experiment 1: v39b WebShop 3B
if ! curl -s http://127.0.0.1:8083 >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] Starting WebShop environment..."
    bash start_env_webshop.sh
fi
run_experiment "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39b.yaml" "webshop_qwen3b_duet_v39b" 1 $TOTAL

# Experiment 2: v39b ALFWorld 3B
echo ""
echo "[$(date '+%m-%d %H:%M')] Switching env: WebShop -> ALFWorld"
bash start_env_webshop.sh stop 2>/dev/null || true
sleep 10

if ! curl -s http://127.0.0.1:8081 >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] Starting ALFWorld environment..."
    bash start_env_alfworld.sh
fi
run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39b.yaml" "alfworld_qwen3b_duet_v39b" 2 $TOTAL

bash start_env_alfworld.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " DUET 3B v39b complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
