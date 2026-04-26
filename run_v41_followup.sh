#!/bin/bash
# ==============================================================================
# v41 follow-up — runs after round-2 orchestrator finishes
#
# Replaces the cancelled run_v40_postround_followup.sh. Per agent-team
# post-truth analysis (see analysis_reports/3b_v39b_post_truth_analysis.md
# and 3b_v39b_vs_luffy_gap_analysis.md), v40_psh and v40_strong_bc are
# both built on v39 (α=0.2) which is now known to be 13.5pp worse than
# v39b (α=0.5) on WebShop. v41 series is built on v39b winning baseline.
#
# Two variants (both predicted to challenge LUFFY 49.5%):
#   v41_psh : v39b + actor-level p/p_β shaping (LUFFY trick)  predict 49-53%
#   v41_tw  : v39b + chord_use_token_weighting=true            predict 47-50%
# ==============================================================================

set +e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

if ! command -v conda >/dev/null 2>&1; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
mkdir -p logs

ROUND2_PID=1230829
LOG="logs/v41_followup.log"

echo "[$(date '+%m-%d %H:%M')] v41 followup waiting for orchestrator PID $ROUND2_PID..."
while kill -0 $ROUND2_PID 2>/dev/null; do
    sleep 300
done
echo "[$(date '+%m-%d %H:%M')] orchestrator $ROUND2_PID gone; starting v41 followup."

# Make sure no stale env services
sleep 10
bash start_env_alfworld.sh stop 2>&1 | tail -2
sleep 5

run_one() {
    local config=$1
    local name=$2

    if ! curl -s --max-time 3 http://127.0.0.1:8083 >/dev/null 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] starting WebShop env..."
        bash start_env_webshop.sh
    fi

    local ray_tmp="${RAY_TMPDIR}/${name}"
    mkdir -p "$ray_tmp"
    rm -rf "$ray_tmp"/session_* 2>/dev/null || true

    echo "[$(date '+%m-%d %H:%M')] RUN: $name"
    CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
        python launcher.py --conf "$config" \
        > "logs/${name}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date '+%m-%d %H:%M')] DONE: $name"
    else
        echo "[$(date '+%m-%d %H:%M')] FAILED ($rc): $name"
    fi
}

# Priority order: v41_psh (highest predicted, the "challenge LUFFY" candidate)
# then v41_tw (token weighting alternate).
run_one \
    "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v41_psh.yaml" \
    "webshop_qwen3b_duet_v41_psh"

run_one \
    "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v41_tw.yaml" \
    "webshop_qwen3b_duet_v41_tw"

bash start_env_webshop.sh stop 2>&1 | tail -2

echo ""
echo "[$(date '+%m-%d %H:%M')] v41 followup complete!"
for exp in webshop_qwen3b_duet_v41_psh webshop_qwen3b_duet_v41_tw; do
    f="experiments/webshop/${exp}/validation_log/100.jsonl"
    if [ -f "$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ MISSING: $f"
    fi
done
