#!/bin/bash
# ==============================================================================
# DUET Paper: WebShop Qwen2.5-3B Experiments (OnPolicy / LUFFY / CHORD / DUET)
#
# 4 GPUs per experiment, 2 experiments in parallel on 8xA100
# Config: bs=8, 800 tasks, rollout.n=8, 100 steps/epoch
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

CONFIG_DIR="config/duet_paper_experiments_configs/webshop"

cleanup_training() {
    echo ">>> Cleaning up stale training processes..."
    rm -rf ${RAY_TMPDIR}/webshop_qwen3b_*/session_* 2>/dev/null || true
}

start_env() {
    echo ">>> Starting WebShop environment (daemonized)..."
    bash start_env_webshop.sh
    echo ">>> Environment ready."
}

stop_env() {
    bash start_env_webshop.sh stop
}

run_experiment() {
    local gpus=$1
    local config=$2
    local name=$3
    local ray_tmp="${RAY_TMPDIR}/${name}"
    mkdir -p "$ray_tmp"

    echo "[RUN] $name on GPUs $gpus"
    CUDA_VISIBLE_DEVICES=$gpus RAY_TMPDIR="$ray_tmp" \
        python launcher.py --conf "$config" \
        > "logs/${name}.log" 2>&1
    echo "[DONE] $name"
}

echo "============================================"
echo " DUET Paper: WebShop Qwen2.5-3B All Experiments"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

mkdir -p logs

# ---------- Round 1: OnPolicy + LUFFY in parallel ----------
echo ""
echo "[Round 1/2] OnPolicy (GPU 0-3) + LUFFY (GPU 4-7)"
echo "============================================"
cleanup_training
start_env

run_experiment "0,1,2,3" "${CONFIG_DIR}/webshop_qwen3b_onpolicy.yaml" "webshop_qwen3b_onpolicy" &
PID1=$!
run_experiment "4,5,6,7" "${CONFIG_DIR}/webshop_qwen3b_luffy.yaml" "webshop_qwen3b_luffy" &
PID2=$!

wait $PID1; echo "[Round 1] OnPolicy finished (exit: $?)"
wait $PID2; echo "[Round 1] LUFFY finished (exit: $?)"

# ---------- Round 2: CHORD + DUET in parallel ----------
echo ""
echo "[Round 2/2] CHORD (GPU 0-3) + DUET (GPU 4-7)"
echo "============================================"
stop_env
cleanup_training
start_env

run_experiment "0,1,2,3" "${CONFIG_DIR}/webshop_qwen3b_chord.yaml" "webshop_qwen3b_chord" &
PID3=$!
run_experiment "4,5,6,7" "${CONFIG_DIR}/webshop_qwen3b_duet.yaml" "webshop_qwen3b_duet" &
PID4=$!

wait $PID3; echo "[Round 2] CHORD finished (exit: $?)"
wait $PID4; echo "[Round 2] DUET finished (exit: $?)"

# ---------- Cleanup ----------
stop_env
cleanup_training
echo ""
echo "============================================"
echo " All WebShop Qwen2.5-3B experiments done."
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
