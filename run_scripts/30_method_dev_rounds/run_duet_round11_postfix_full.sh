#!/bin/bash
# ==============================================================================
# DUET Round 11 — Full v39 family rerun after B1/B2/U1/B3 fixes
#
# All v39 family experiments before commit eabd1019 were polluted by:
#   B2 (HIGH): cross-rank disc_acc desync — only rank0 had real disc_acc
#   B1 (HIGH, ALFWorld only): SC β decay killed SC channel from step 1
#   U1 (HIGH, ALFWorld only): adaptive_weight reverse-amplified teacher
#   B3 (MED): teacher_off_pg_loss diagnostic missing
#
# Round 11 reruns the entire v39 hyperparameter sweep on the fixed code.
#
# 6 experiments (sequential, ~24h total):
#   WebShop 1.5B (~9h, env switch once)
#     1. v39b_postfix — α=0.5, d_floor=0.5  ← Phase 1 winner
#     2. v39_postfix  — α=0.2, d_floor=0.5  ← original
#     3. v39c_postfix — α=0.5, d_floor=0.4  ← param sweep
#   ALFWorld 1.5B (~15h)
#     4. v39b_postfix — α=0.5, d_floor=0.5  ← Phase 1 winner
#     5. v39_postfix  — α=0.2, d_floor=0.5  ← original
#     6. v39c_postfix — α=0.5, d_floor=0.4  ← param sweep
#
# Resilience features (from Round 10 lessons):
#   - NO `set -e` — one failure doesn't abort subsequent experiments
#   - Pre-flight: kill ray stragglers + wait for GPU clean before each run
#   - Post-flight: cleanup ray + sleep before next experiment
# ==============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
mkdir -p logs

wait_for_gpu_clean() {
    for i in {1..30}; do
        local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>200' | wc -l)
        if [ "$used" = "0" ]; then
            echo "  GPU clean"
            return 0
        fi
        echo "  Waiting for GPU to clear ($used GPUs still busy)... ${i}/30"
        sleep 10
    done
    echo "  WARN: GPU not fully clean after 300s, proceeding anyway"
}

kill_ray_stragglers() {
    local n=$(ps -ef | grep -E "ray::|raylet" | grep -v grep | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "  Killing $n stale Ray processes..."
        ps -ef | grep -E "ray::|raylet" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
        sleep 5
        ps -ef | grep -E "ray::|raylet" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
        sleep 3
    fi
}

run_experiment() {
    local config=$1
    local name=$2
    local idx=$3
    local total=$4
    # FIX: use short path to avoid AF_UNIX 107-byte limit on socket file.
    # Old: RAY_TMPDIR/${name}/session_X/sockets/plasma_store can exceed 107 bytes.
    # New: just use base RAY_TMPDIR; Ray auto-creates session_TIMESTAMP subdir.
    local ray_tmp="${RAY_TMPDIR}"

    echo ""
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] PREP: $name"
    kill_ray_stragglers
    wait_for_gpu_clean

    mkdir -p "$ray_tmp"

    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] RUN: $name"
    (
        CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
            python launcher.py --conf "$config" \
            > "logs/${name}.log" 2>&1
    )
    local rc=$?
    if [ "$rc" = "0" ]; then
        echo "[$(date '+%m-%d %H:%M')] [$idx/$total] DONE: $name"
    else
        echo "[$(date '+%m-%d %H:%M')] [$idx/$total] FAILED (rc=$rc): $name — continuing"
    fi

    kill_ray_stragglers
    sleep 5
}

echo "============================================"
echo " DUET Round 11: v39 family postfix rerun (6 experiments)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPU: $GPUS"
echo " All bug fixes from commit eabd1019: B1/B2/U1/B3"
echo "============================================"

TOTAL=6

# ============================================
# Phase 1 of Round 11: WebShop 1.5B (~9h)
# ============================================
if ! curl -s http://127.0.0.1:8083 >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] Starting WebShop environment..."
    bash start_env_webshop.sh
fi

# v39b_postfix first (Phase 1 winner — validate fix actually works)
run_experiment "config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v39b_postfix.yaml" \
    "webshop_qwen1.5b_duet_v39b_postfix" 1 $TOTAL

# v39_postfix (α=0.2 — original)
run_experiment "config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v39_postfix.yaml" \
    "webshop_qwen1.5b_duet_v39_postfix" 2 $TOTAL

# v39c_postfix (d_floor=0.4 — earlier knee)
run_experiment "config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v39c_postfix.yaml" \
    "webshop_qwen1.5b_duet_v39c_postfix" 3 $TOTAL

# ============================================
# Switch env: WebShop -> ALFWorld
# ============================================
echo ""
echo "[$(date '+%m-%d %H:%M')] Switching env: WebShop -> ALFWorld"
bash start_env_webshop.sh stop 2>/dev/null || true
sleep 10

if ! curl -s http://127.0.0.1:8081 >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] Starting ALFWorld environment..."
    bash start_env_alfworld.sh
fi

# ============================================
# Phase 2 of Round 11: ALFWorld 1.5B (~15h)
# ============================================

# v39b_postfix first
run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39b_postfix.yaml" \
    "alfworld_qwen1.5b_duet_v39b_postfix" 4 $TOTAL

# v39_postfix
run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39_postfix.yaml" \
    "alfworld_qwen1.5b_duet_v39_postfix" 5 $TOTAL

# v39c_postfix
run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39c_postfix.yaml" \
    "alfworld_qwen1.5b_duet_v39c_postfix" 6 $TOTAL

bash start_env_alfworld.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " Round 11 complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
