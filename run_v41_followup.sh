#!/bin/bash
# ==============================================================================
# v39b rerun + v41 followup + DUET* gap-driven μ parameter search.
#
# Order (priority: finish original v39 series before exploring new variants):
#   0) alfworld v39b RERUN  (was crashed by K8s OOM at step 89, original plan)
#   1) v41_tw               (WebShop, token-weighted BC, last item from v41 plan)
#   2) v_gap_ws_a           (WebShop, gap-driven μ peak=0.30, decay-cap γ=0.93)
#   3) v_no_bc_ws           (WebShop, BC disabled — does BC help 3B WS at all)
#   4) v_clean_ws           (WebShop, closed-form μ schedule — sanity check)
#   5) v_gap_ws_b           (WebShop, gap-driven μ peak=0.20)
#   6) v_gap_af_a           (ALFWorld, gap-driven μ peak=0.30)
#   7) v_gap_af_b           (ALFWorld, gap-driven μ peak=0.20, valley=0.02)
#
# Each run does a force stop+start of env services first to defend against the
# alfworld memory-leak that killed the original v39b at step 89.
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
bash start_env_webshop.sh stop 2>&1 | tail -2
sleep 5

run_one() {
    local config=$1
    local name=$2
    local env=$3   # "webshop" or "alfworld"

    if [ ! -f "$config" ]; then
        echo "[$(date '+%m-%d %H:%M')] SKIP (config missing): $name  ($config)"
        return 0
    fi

    # ⭐ Force restart env services BEFORE every run.
    # Reason: ALFWorld env service has a memory leak (~50GB RSS after 26h);
    # WebShop also leaks slowly. A fresh service per run avoids cumulative
    # crashes (see v39b death at step 89 of run_duet_3b_v40_ablation, 04-27).
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

# ─── PRIORITY 1: Re-run v39b on ALFWorld (crashed 04-27 02:08 with K8s OOM) ────
# Was the most-promising run: val@50=58.5%, succ_ema@step89=71.4% before env crash.
# Without val@100, we lose the headline ALFWorld DUET* number. Run this FIRST.
run_one \
    "config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39b.yaml" \
    "alfworld_qwen3b_duet_v39b" "alfworld"

bash start_env_alfworld.sh stop 2>&1 | tail -2
sleep 10

# ─── WebShop block (5 runs) ─────────────────────────────────────────────
# v41_tw remains (token-weighted BC) — original plan
run_one \
    "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v41_tw.yaml" \
    "webshop_qwen3b_duet_v41_tw" "webshop"

# DUET* gap-driven main + ablations
run_one \
    "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v_gap_ws_a.yaml" \
    "webshop_qwen3b_duet_v_gap_ws_a" "webshop"

run_one \
    "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v_no_bc_ws.yaml" \
    "webshop_qwen3b_duet_v_no_bc_ws" "webshop"

run_one \
    "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v_clean_ws.yaml" \
    "webshop_qwen3b_duet_v_clean_ws" "webshop"

run_one \
    "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v_gap_ws_b.yaml" \
    "webshop_qwen3b_duet_v_gap_ws_b" "webshop"

bash start_env_webshop.sh stop 2>&1 | tail -2
sleep 10

# ─── ALFWorld DUET* gap-driven runs (2 runs) ────────────────────────────
run_one \
    "config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v_gap_af_a.yaml" \
    "alfworld_qwen3b_duet_v_gap_af_a" "alfworld"

run_one \
    "config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v_gap_af_b.yaml" \
    "alfworld_qwen3b_duet_v_gap_af_b" "alfworld"

bash start_env_alfworld.sh stop 2>&1 | tail -2

echo ""
echo "[$(date '+%m-%d %H:%M')] v39b rerun + v41 followup + DUET* parameter search complete!"
for exp in alfworld_qwen3b_duet_v39b; do
    f="experiments/alfworld/${exp}/validation_log/100.jsonl"
    if [ -f "$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ MISSING: $f"
    fi
done
for exp in webshop_qwen3b_duet_v41_tw \
           webshop_qwen3b_duet_v_gap_ws_a \
           webshop_qwen3b_duet_v_no_bc_ws \
           webshop_qwen3b_duet_v_clean_ws \
           webshop_qwen3b_duet_v_gap_ws_b; do
    f="experiments/webshop/${exp}/validation_log/100.jsonl"
    if [ -f "$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ MISSING: $f"
    fi
done
for exp in alfworld_qwen3b_duet_v_gap_af_a alfworld_qwen3b_duet_v_gap_af_b; do
    f="experiments/alfworld/${exp}/validation_log/100.jsonl"
    if [ -f "$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ MISSING: $f"
    fi
done
