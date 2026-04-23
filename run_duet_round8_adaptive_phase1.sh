#!/bin/bash
# ==============================================================================
# DUET Round 8 — Adaptive μ Phase 1: 4 frameworks, one representative each
#
# Sequential WebShop 1.5B runs (each ~3h, total ~12h):
#   1. v43a — KL-Lagrangian (η=0.3, ρ=0.9) ← PAPER PRIMARY NARRATIVE
#   2. v40b — NLL linear (0.02 + 0.156·NLL) ← empirical top signal
#   3. v39b — Disc fast-EMA (α=0.2→0.5) ← fix phase-lag
#   4. v41b — ESS saturating (pow=0.5) ← density-ratio quality
#
# Goal: identify which framework's basic implementation best matches v24
# (Val@100 success ≥ 22.0%). Then Phase 2 will sweep parameters of winner.
#
# Runs AFTER Round 7 (v39 ALFWorld) completes. Switches env.
#
# Theoretical framework summary:
#   v43a: BC as Lagrange multiplier on KL-to-teacher constraint (TRPO family)
#   v40b: BC retires as teacher NLL converges
#   v39b: BC retires as discriminator separates π_θ from π_teacher
#   v41b: BC retires as density-ratio ESS saturates
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
echo " DUET Round 8: Adaptive μ Phase 1 (4 frameworks)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPU: $GPUS"
echo "============================================"

# Wait for Round 7 (v39 ALFWorld) to finish
echo ""
echo "[$(date '+%m-%d %H:%M')] Waiting for Round 7 orchestrator to finish..."
while pgrep -f "run_duet_round7_v39.sh" > /dev/null; do
    sleep 60
done
echo "[$(date '+%m-%d %H:%M')] Round 7 done. Switching env: ALFWorld -> WebShop"

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

TOTAL=4

# Priority order: Lagrangian first (paper narrative), then empirical top, then hedges
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v43a.yaml" "webshop_qwen1.5b_duet_v43a" 1 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v40b.yaml" "webshop_qwen1.5b_duet_v40b" 2 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v39b.yaml" "webshop_qwen1.5b_duet_v39b" 3 $TOTAL
run_experiment "${CONFIG_DIR}/webshop_qwen1.5b_duet_v41b.yaml" "webshop_qwen1.5b_duet_v41b" 4 $TOTAL

bash start_env_webshop.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " Round 8 Phase 1 complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
