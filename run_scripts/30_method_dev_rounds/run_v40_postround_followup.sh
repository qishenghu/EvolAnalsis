#!/bin/bash
# ==============================================================================
# Post-round-2 follow-up:
#   1. Re-run webshop_qwen3b_duet_v40_psh with the FIXED config
#   2. Re-run webshop_qwen3b_duet_v40_strong_bc (previous run died at step 43
#      from a wandb BrokenPipe, while training metrics were healthy and rising)
#
# Robust completion check: waits for the round-2 orchestrator's exact PID
# (1230829) to die, instead of pgrep-pattern matching which mis-fired during
# the brief transition between v40_strong_bc death and v39b start.
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

ROUND2_PID=1230829   # webshop_qwen3b_duet_v40_ablation orchestrator
LOG="logs/v40_postround_followup.log"
TS=$(date +%Y%m%d_%H%M)

echo "[$(date '+%m-%d %H:%M')] post-round-2 followup waiting for orchestrator PID $ROUND2_PID..."

# Wait for orchestrator to die (PID-based — no pattern false-positive)
while kill -0 $ROUND2_PID 2>/dev/null; do
    sleep 300
    echo "[$(date '+%m-%d %H:%M')] orchestrator $ROUND2_PID still alive, waiting..."
done
echo "[$(date '+%m-%d %H:%M')] orchestrator $ROUND2_PID is gone; starting followup."

# Make sure no env services left over
sleep 10
bash start_env_alfworld.sh stop 2>&1 | tail -2
sleep 5
# We'll run on webshop env

# Helper for running an experiment
run_one() {
    local config=$1
    local name=$2
    local env=$3

    # Archive any previous run with this name (in case of partial)
    src="experiments/${env}/${name}"
    if [ -d "$src" ]; then
        mv "$src" "experiments/${env}/${name}_prevrun_${TS}"
        echo "[$(date '+%m-%d %H:%M')] archived prev run: $src"
    fi
    src_ckpt="checkpoints/agentevolver/${name}"
    if [ -d "$src_ckpt" ]; then
        mv "$src_ckpt" "checkpoints/agentevolver/${name}_prevrun_${TS}"
    fi

    # ensure webshop env up
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
        > "logs/${name}_followup.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date '+%m-%d %H:%M')] DONE: $name"
    else
        echo "[$(date '+%m-%d %H:%M')] FAILED ($rc): $name"
    fi
}

# Run v40_psh first (highest paper priority — LUFFY-style policy shaping)
run_one \
    "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v40_psh.yaml" \
    "webshop_qwen3b_duet_v40_psh" "webshop"

# Run v40_strong_bc (was healthy, killed by transient wandb crash)
run_one \
    "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v40_strong_bc.yaml" \
    "webshop_qwen3b_duet_v40_strong_bc" "webshop"

# cleanup
bash start_env_webshop.sh stop 2>&1 | tail -2

echo ""
echo "[$(date '+%m-%d %H:%M')] post-round-2 followup complete!"
for exp in webshop_qwen3b_duet_v40_psh webshop_qwen3b_duet_v40_strong_bc; do
    f="experiments/webshop/${exp}/validation_log/100.jsonl"
    if [ -f "$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ MISSING: $f"
    fi
done
