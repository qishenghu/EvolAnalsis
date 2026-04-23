#!/bin/bash
# ==============================================================================
# DUET Round 6: v38 — Surprise-weighted DR3 (unified BC into DR3)
#
# v38 tests Spec 2 / Candidate B:
#   - Adds (1 - π_θ(a|s)) surprise weighting to DR3 teacher surrogate
#   - No explicit BC/CHORD term (use_chord=false)
#   - Only 1 new "hyperparameter" (spw_mask_on_positive_A=true, safety default)
#   - If ≥0.65 on WebShop 1.5B: paper narrative becomes "single unified operator"
#
# This script waits for Round 5 (v37 + v24-ALFWorld) to finish, switches env,
# then runs v38 on WebShop 1.5B.
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
CONFIG="config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v38.yaml"
NAME="webshop_qwen1.5b_duet_v38"
mkdir -p logs

echo "============================================"
echo " DUET Round 6: v38 (surprise-weighted DR3)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPU: $GPUS"
echo "============================================"

# Wait for Round 5 orchestrator (v24-ALFWorld) to complete
echo ""
echo "[$(date '+%m-%d %H:%M')] Waiting for Round 5 orchestrator to finish..."
while pgrep -f "run_duet_round5_adaptive_mu.sh" > /dev/null; do
    sleep 60
done
echo "[$(date '+%m-%d %H:%M')] Round 5 done. Switching env: ALFWorld -> WebShop"

# Stop ALFWorld env (if still running)
bash start_env_alfworld.sh stop 2>/dev/null || true
sleep 10

# Start WebShop env
if ! curl -s http://127.0.0.1:8083 >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] Starting WebShop environment..."
    bash start_env_webshop.sh
fi

# Wait for env ready
for i in {1..30}; do
    if curl -s http://127.0.0.1:8083 >/dev/null 2>&1; then break; fi
    sleep 2
done

# Launch v38
ray_tmp="${RAY_TMPDIR}/${NAME}"
mkdir -p "$ray_tmp"
rm -rf "$ray_tmp"/session_* 2>/dev/null || true

echo ""
echo "[$(date '+%m-%d %H:%M')] RUN: $NAME"
CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
    python launcher.py --conf "$CONFIG" \
    > "logs/${NAME}.log" 2>&1
echo "[$(date '+%m-%d %H:%M')] DONE: $NAME"

bash start_env_webshop.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " Round 6 complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
