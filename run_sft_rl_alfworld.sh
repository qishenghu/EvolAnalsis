#!/bin/bash
# ==============================================================================
# DUET Paper: ALFWorld SFT+RL Baseline
#
# Two-phase training:
#   Phase 1 (SFT): CHORD mu=1.0 on teacher data, 400 tasks, 50 steps
#   Phase 2 (RL):  On-policy GRPO from SFT checkpoint, 400 tasks, 50 steps
#
# Total: 50+50=100 gradient steps, matching other methods
# Runs both 1.5B and 3B sequentially (each uses 4 GPUs)
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

CONFIG_DIR="config/duet_paper_experiments_configs/alfworld"

cleanup_training() {
    local name=$1
    rm -rf ${RAY_TMPDIR}/${name}/session_* 2>/dev/null || true
}

start_env() {
    echo ">>> Starting ALFWorld environment..."
    bash start_env_alfworld.sh
    nohup bash watchdog_agentgym.sh > /dev/null 2>&1 &
    disown
    echo ">>> Environment + watchdog ready."
}

stop_env() {
    if [ -f /tmp/watchdog_alfworld.pid ]; then
        kill $(cat /tmp/watchdog_alfworld.pid) 2>/dev/null || true
        rm -f /tmp/watchdog_alfworld.pid
    fi
    bash start_env_alfworld.sh stop
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

run_sft_rl() {
    local size=$1  # qwen1.5b or qwen3b
    local gpus=$2  # e.g., "0,1,2,3"

    echo ""
    echo "============================================"
    echo " SFT+RL for alfworld_${size}"
    echo " $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================"

    # --- Phase 1: SFT (50 steps) ---
    echo "[Phase 1/2] SFT on teacher data (50 steps)..."
    cleanup_training "alfworld_${size}_sft"
    run_experiment "$gpus" \
        "${CONFIG_DIR}/alfworld_${size}_sft.yaml" \
        "alfworld_${size}_sft"

    # Verify SFT checkpoint exists
    local sft_ckpt="checkpoints/agentevolver/alfworld_${size}_sft/global_step_50/actor"
    if [ ! -d "$sft_ckpt" ]; then
        echo "ERROR: SFT checkpoint not found at $sft_ckpt"
        echo "Checking available checkpoints..."
        ls -d checkpoints/agentevolver/alfworld_${size}_sft/global_step_*/actor 2>/dev/null || echo "No checkpoints found"
        return 1
    fi
    echo "[Phase 1] SFT checkpoint saved at: $sft_ckpt"

    # --- Phase 2: RL from SFT checkpoint (50 steps) ---
    echo "[Phase 2/2] On-policy GRPO from SFT checkpoint (50 steps)..."
    cleanup_training "alfworld_${size}_sft_rl"
    run_experiment "$gpus" \
        "${CONFIG_DIR}/alfworld_${size}_sft_rl.yaml" \
        "alfworld_${size}_sft_rl"

    echo "[DONE] SFT+RL for alfworld_${size} complete"
}

echo "============================================"
echo " DUET Paper: ALFWorld SFT+RL Baselines"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

mkdir -p logs
start_env

# Run 1.5B and 3B in parallel (4 GPUs each)
run_sft_rl "qwen1.5b" "0,1,2,3" &
PID1=$!
run_sft_rl "qwen3b" "4,5,6,7" &
PID2=$!

wait $PID1; echo "1.5B SFT+RL finished (exit: $?)"
wait $PID2; echo "3B SFT+RL finished (exit: $?)"

stop_env
echo ""
echo "============================================"
echo " All ALFWorld SFT+RL baselines done."
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
