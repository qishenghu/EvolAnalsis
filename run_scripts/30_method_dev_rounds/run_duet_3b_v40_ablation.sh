#!/bin/bash
# ==============================================================================
# DUET 3B v39/v40 ablation — round 2 (after agent-team analysis 2026-04-25)
#
# Priority queue (5 experiments, ~28.5h on 4× L20X):
#   1. webshop_qwen3b_duet_v40_psh        ~3.5h  P0 highest-hope: enable
#                                                LUFFY-style p/p_β shaping
#                                                (predicted 42-48% success)
#   2. webshop_qwen3b_duet_v40_strong_bc  ~3.5h  P0 alternate: μ_peak 0.6 +
#                                                token weighting (predicted 38-44%)
#   3. webshop_qwen3b_duet_v39b           ~3.5h  α=0.5 ablation (does α matter
#                                                under bug-fixed code?)
#   4. alfworld_qwen3b_duet_v39           ~9h    paper headline ALFWorld
#                                                (DUET strong env, was killed
#                                                mid-run for v40 reordering)
#   5. alfworld_qwen3b_duet_v39b          ~9h    α=0.5 ablation on ALFWorld
#
# Dropped from previous queue: v39c (defensive variant; v40_psh and v40_strong_bc
# now cover that role with mechanistic justification rather than parameter sweep).
#
# Designed to start fresh from a clean state (no env services running, no GPU
# residue, no pid files). Idempotent: skips runs whose validation_log/100.jsonl
# already exists.
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

current_env=""

ensure_env_running() {
    local env=$1
    if [ "$current_env" = "$env" ]; then
        return 0
    fi
    if [ "$current_env" = "webshop" ]; then
        bash start_env_webshop.sh stop 2>&1 | tail -2
        sleep 5
    elif [ "$current_env" = "alfworld" ]; then
        bash start_env_alfworld.sh stop 2>&1 | tail -2
        sleep 5
    fi
    case "$env" in
        webshop)
            if ! curl -s --max-time 3 http://127.0.0.1:8083 >/dev/null 2>&1; then
                echo "[$(date '+%m-%d %H:%M')] Starting WebShop env..."
                bash start_env_webshop.sh
            fi
            ;;
        alfworld)
            if ! curl -s --max-time 3 http://127.0.0.1:8081 >/dev/null 2>&1; then
                echo "[$(date '+%m-%d %H:%M')] Starting ALFWorld env..."
                bash start_env_alfworld.sh
            fi
            ;;
    esac
    current_env="$env"
}

run_experiment() {
    local config=$1
    local name=$2
    local env=$3
    local idx=$4
    local total=$5

    local val_done="experiments/${env}/${name}/validation_log/100.jsonl"
    if [ -f "$val_done" ]; then
        echo "[$(date '+%m-%d %H:%M')] [$idx/$total] SKIP (val@100 exists): $name"
        return 0
    fi

    ensure_env_running "$env"

    local ray_tmp="${RAY_TMPDIR}/${name}"
    mkdir -p "$ray_tmp"
    rm -rf "$ray_tmp"/session_* 2>/dev/null || true

    echo ""
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] RUN: $name"
    CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
        python launcher.py --conf "$config" \
        > "logs/${name}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date '+%m-%d %H:%M')] [$idx/$total] DONE: $name"
    else
        echo "[$(date '+%m-%d %H:%M')] [$idx/$total] FAILED ($rc): $name — see logs/${name}.log"
    fi
    return $rc
}

echo "============================================"
echo " DUET 3B v40/v39 ablation round-2"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPU: $GPUS"
echo " Queue: v40_psh ws → v40_strong_bc ws → v39b ws → v39 af → v39b af"
echo "============================================"

TOTAL=5

# WebShop block (3 experiments, ~10.5h, env stays up)
run_experiment \
    "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v40_psh.yaml" \
    "webshop_qwen3b_duet_v40_psh" "webshop" 1 $TOTAL

run_experiment \
    "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v40_strong_bc.yaml" \
    "webshop_qwen3b_duet_v40_strong_bc" "webshop" 2 $TOTAL

run_experiment \
    "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39b.yaml" \
    "webshop_qwen3b_duet_v39b" "webshop" 3 $TOTAL

# ALFWorld block (2 experiments, ~18h, env stays up)
run_experiment \
    "config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39.yaml" \
    "alfworld_qwen3b_duet_v39" "alfworld" 4 $TOTAL

run_experiment \
    "config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39b.yaml" \
    "alfworld_qwen3b_duet_v39b" "alfworld" 5 $TOTAL

# cleanup
if [ "$current_env" = "webshop" ]; then
    bash start_env_webshop.sh stop 2>&1 | tail -2
elif [ "$current_env" = "alfworld" ]; then
    bash start_env_alfworld.sh stop 2>&1 | tail -2
fi

echo ""
echo "============================================"
echo " round-2 complete!"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""
echo "Validation results:"
for exp in webshop_qwen3b_duet_v40_psh webshop_qwen3b_duet_v40_strong_bc webshop_qwen3b_duet_v39b alfworld_qwen3b_duet_v39 alfworld_qwen3b_duet_v39b; do
    env=$(echo "$exp" | cut -d_ -f1)
    f="experiments/${env}/${exp}/validation_log/100.jsonl"
    if [ -f "$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ MISSING: $f"
    fi
done
