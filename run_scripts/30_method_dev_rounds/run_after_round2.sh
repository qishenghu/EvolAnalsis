#!/bin/bash
# Runs after Round 2 completes:
#   1. Round 3: DUET+SFT hybrid (v22, v23, v24) — ~7.5h
#   2. ALFWorld 3B CHORD rerun (with fixed config) — ~8h

set -e
cd /data/home/qisheng/EvolAnalsis
source env_config.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
mkdir -p logs

run_experiment() {
    local config=$1
    local name=$2
    local idx=$3
    local total=$4
    local ray_tmp="${RAY_TMPDIR}/${name}"
    mkdir -p "$ray_tmp"
    rm -rf "$ray_tmp"/session_* 2>/dev/null || true

    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] RUN: $name"
    CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
        python launcher.py --conf "$config" \
        > "logs/${name}.log" 2>&1
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] DONE: $name"
}

echo "============================================"
echo " Post-Round2: v22-v24 + 3B CHORD"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# ---- Round 3: DUET+SFT hybrid ----
echo ">>> Round 3: DUET+SFT Hybrid"
if ! curl -s http://127.0.0.1:8083 >/dev/null 2>&1; then
    bash start_env_webshop.sh
fi

run_experiment "config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v22.yaml" "webshop_qwen1.5b_duet_v22" 1 4
run_experiment "config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v23.yaml" "webshop_qwen1.5b_duet_v23" 2 4
run_experiment "config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v24.yaml" "webshop_qwen1.5b_duet_v24" 3 4

bash start_env_webshop.sh stop 2>/dev/null || true

# ---- ALFWorld 3B CHORD rerun ----
echo ""
echo ">>> ALFWorld 3B CHORD (optimizer_offload=true, gpu_mem=0.5)"
bash start_env_alfworld.sh
nohup bash watchdog_agentgym.sh > /dev/null 2>&1 & disown

run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_chord.yaml" "alfworld_qwen3b_chord" 4 4

if [ -f /tmp/watchdog_alfworld.pid ]; then
    kill $(cat /tmp/watchdog_alfworld.pid) 2>/dev/null || true
    rm -f /tmp/watchdog_alfworld.pid
fi
bash start_env_alfworld.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " All done! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
