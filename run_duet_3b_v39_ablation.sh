#!/bin/bash
# ==============================================================================
# DUET 3B — v39 / v39b / v39c ablation (post bug-fix rerun)
#
# 6 experiments total: {v39, v39b, v39c} × {WebShop, ALFWorld} on 4×L20X-144G.
# All run on bug-fixed code (commit eabd1019 onward): B1 SC decay fixed,
# B2 cross-rank μ broadcast fixed, B3 teacher_off_pg_loss diag added,
# U1 gap_linear fade-by-success.
#
# Hyperparameter ablation (only knob differences):
#   v39  : α=0.2  valley=0.05  kl=default     (mirrors 1.5B v39 success)
#   v39b : α=0.5  valley=0.05  kl=default     (Phase-1 winner setting)
#   v39c : α=0.2  valley=0.10  kl=default×3   (defensive on top of bug fixes)
#
# Priority order (so partial completion preserves the most paper-critical data):
#   1. v39 WebShop   ~3.5h    (paper headline result)
#   2. v39 ALFWorld  ~9h
#   3. v39b WebShop  ~3.5h    (ablation: α matters?)
#   4. v39b ALFWorld ~9h
#   5. v39c WebShop  ~3.5h    (defensive variant if v39 unstable)
#   6. v39c ALFWorld ~9h
#
# Total ~37h. Idempotent: restart-safe (skips runs whose val@100 jsonl exists).
# ==============================================================================

set +e   # don't abort the whole ablation if one experiment fails
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

if ! command -v conda >/dev/null 2>&1; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
mkdir -p logs

# ---- helpers --------------------------------------------------------------

current_env=""

ensure_env_running() {
    local env=$1
    if [ "$current_env" = "$env" ]; then
        return 0
    fi
    # tear down whatever is up
    if [ "$current_env" = "webshop" ]; then
        bash start_env_webshop.sh stop 2>&1 | tail -2
        sleep 5
    elif [ "$current_env" = "alfworld" ]; then
        bash start_env_alfworld.sh stop 2>&1 | tail -2
        sleep 5
    fi
    # bring up the new env
    case "$env" in
        webshop)
            local port_check_url="http://127.0.0.1:8083"
            if ! curl -s --max-time 3 "$port_check_url" >/dev/null 2>&1; then
                echo "[$(date '+%m-%d %H:%M')] Starting WebShop env..."
                bash start_env_webshop.sh
            fi
            ;;
        alfworld)
            local port_check_url="http://127.0.0.1:8081"
            if ! curl -s --max-time 3 "$port_check_url" >/dev/null 2>&1; then
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

    # Idempotency: skip if val@100 already exists
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

# ---- main --------------------------------------------------------------

echo "============================================"
echo " DUET 3B v39 ablation: v39 / v39b / v39c × WebShop / ALFWorld"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPU: $GPUS"
echo "============================================"

TOTAL=6

# --- v39 (α=0.2) — paper headline result ---
run_experiment \
    "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39.yaml" \
    "webshop_qwen3b_duet_v39" "webshop" 1 $TOTAL

run_experiment \
    "config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39.yaml" \
    "alfworld_qwen3b_duet_v39" "alfworld" 2 $TOTAL

# --- v39b (α=0.5) — ablation companion ---
run_experiment \
    "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39b.yaml" \
    "webshop_qwen3b_duet_v39b" "webshop" 3 $TOTAL

run_experiment \
    "config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39b.yaml" \
    "alfworld_qwen3b_duet_v39b" "alfworld" 4 $TOTAL

# --- v39c (defensive: valley=0.10, kl×3) ---
run_experiment \
    "config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39c.yaml" \
    "webshop_qwen3b_duet_v39c" "webshop" 5 $TOTAL

run_experiment \
    "config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_duet_v39c.yaml" \
    "alfworld_qwen3b_duet_v39c" "alfworld" 6 $TOTAL

# --- cleanup ---
if [ "$current_env" = "webshop" ]; then
    bash start_env_webshop.sh stop 2>&1 | tail -2
elif [ "$current_env" = "alfworld" ]; then
    bash start_env_alfworld.sh stop 2>&1 | tail -2
fi

echo ""
echo "============================================"
echo " DUET 3B v39 ablation complete!"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""
echo "Validation log files:"
for v in v39 v39b v39c; do
    for env in webshop alfworld; do
        f="experiments/${env}/${env}_qwen3b_duet_${v}/validation_log/100.jsonl"
        if [ -f "$f" ]; then
            echo "  ✓ $f"
        else
            echo "  ✗ MISSING: $f"
        fi
    done
done
