#!/bin/bash
# ==============================================================================
# Priority queue (after WebShop crisis 04-28):
#   1. wait for v_clean_ws (orphan launcher PID 1363006) to finish
#   2. run alfworld v_gap_af_a (~11h)
#   3. run alfworld v_gap_af_b (~11h)
#
# WebShop block is intentionally cut short — gap-driven on WS already shown
# to fail (v_gap_ws_a = 8.5%). v_gap_ws_b is unlikely to behave differently.
# WS root cause investigation runs in parallel.
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

VCLEAN_PID=1363006

echo "[$(date '+%m-%d %H:%M')] AF priority: waiting for v_clean_ws (PID $VCLEAN_PID) to finish..."
while kill -0 $VCLEAN_PID 2>/dev/null; do
    sleep 60
done
echo "[$(date '+%m-%d %H:%M')] v_clean_ws done; tearing down env."

sleep 5
bash start_env_webshop.sh stop 2>&1 | tail -1
bash start_env_alfworld.sh stop 2>&1 | tail -1
sleep 8

run_one() {
    local config=$1
    local name=$2
    local env=$3

    if [ ! -f "$config" ]; then
        echo "[$(date '+%m-%d %H:%M')] SKIP (config missing): $name"
        return 0
    fi

    echo "[$(date '+%m-%d %H:%M')] [pre-$name] stopping ALL env services..."
    bash start_env_alfworld.sh stop 2>&1 | tail -1
    bash start_env_webshop.sh stop 2>&1 | tail -1
    sleep 8

    echo "[$(date '+%m-%d %H:%M')] [pre-$name] starting fresh ${env} env..."
    if [ "$env" = "webshop" ]; then
        bash start_env_webshop.sh
    elif [ "$env" = "alfworld" ]; then
        bash start_env_alfworld.sh
    fi
    sleep 5

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

# ─── ALFWorld DUET* gap-driven runs ─────────────────────────────────────
run_one \
    "config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v_gap_af_a.yaml" \
    "alfworld_qwen3b_duet_v_gap_af_a" "alfworld"

run_one \
    "config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v_gap_af_b.yaml" \
    "alfworld_qwen3b_duet_v_gap_af_b" "alfworld"

bash start_env_alfworld.sh stop 2>&1 | tail -1

echo ""
echo "[$(date '+%m-%d %H:%M')] AF priority queue complete!"
for exp in alfworld_qwen3b_duet_v_gap_af_a alfworld_qwen3b_duet_v_gap_af_b; do
    f="experiments/alfworld/${exp}/validation_log/100.jsonl"
    if [ -f "$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ MISSING: $f"
    fi
done
