#!/bin/bash
# ==============================================================================
# DUET Paper: ALFWorld 7B Experiments (LUFFY / On-policy / CHORD / DUET)
#
# 4 GPUs per experiment, 2 experiments in parallel on 8xA100
# Config: bs=8, 800 tasks, rollout.n=8, 100 steps/epoch
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

CONFIG_DIR="config/duet_paper_experiments_configs/alfworld"

cleanup_training() {
    echo ">>> Cleaning up stale training processes..."
    rm -rf ${RAY_TMPDIR}/alfworld_7b_*/session_* 2>/dev/null || true
}

start_env() {
    echo ">>> Starting ALFWorld environment (daemonized)..."
    bash start_env_alfworld.sh
    echo ">>> Starting watchdog..."
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

echo "============================================"
echo " DUET Paper: ALFWorld 7B All Experiments"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

mkdir -p logs

# ---------- Round 1: LUFFY + On-policy in parallel ----------
echo ""
echo "[Round 1/2] LUFFY (GPU 0-3) + On-policy (GPU 4-7)"
echo "============================================"
cleanup_training
start_env

#run_experiment "0,1,2,3" "${CONFIG_DIR}/alfworld_7b_luffy.yaml" "alfworld_7b_luffy" &
#PID1=$!
#run_experiment "4,5,6,7" "${CONFIG_DIR}/alfworld_7b_onpolicy.yaml" "alfworld_7b_onpolicy" &
#PID2=$!

#wait $PID1; echo "[Round 1] LUFFY finished (exit: $?)"
#wait $PID2; echo "[Round 1] On-policy finished (exit: $?)"

# ---------- Round 2: CHORD + DUET in parallel ----------
echo ""
echo "[Round 2/2] CHORD (GPU 0-3) + DUET (GPU 4-7)"
echo "============================================"
#stop_env
#cleanup_training
#start_env

run_experiment "0,1,2,3" "${CONFIG_DIR}/alfworld_7b_chord.yaml" "alfworld_7b_chord" &
PID3=$!
run_experiment "4,5,6,7" "${CONFIG_DIR}/alfworld_7b_duet.yaml" "alfworld_7b_duet" &
PID4=$!

wait $PID3; echo "[Round 2] CHORD finished (exit: $?)"
wait $PID4; echo "[Round 2] DUET finished (exit: $?)"

# ---------- Cleanup ----------
stop_env
cleanup_training
echo ""
echo "============================================"
echo " All ALFWorld 7B experiments done (4 total)."
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
