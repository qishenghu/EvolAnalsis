#!/bin/bash
# ==============================================================================
# 补跑实验：
#   1. WebShop 1.5B DUET v2 (SC beta=0.1, 解决 train-val gap)
#   2. ALFWorld 3B CHORD (主表缺失)
# GPU 0-3, 顺序执行
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="0,1,2,3"
mkdir -p logs

run_experiment() {
    local config=$1
    local name=$2
    local ray_tmp="${RAY_TMPDIR}/${name}"
    mkdir -p "$ray_tmp"
    rm -rf "$ray_tmp"/session_* 2>/dev/null || true

    echo ""
    echo "[$(date '+%H:%M:%S')] RUN: $name on GPUs $GPUS"
    CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
        python launcher.py --conf "$config" \
        > "logs/${name}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE: $name"
}

echo "============================================"
echo " 补跑: DUET v2 + 3B CHORD"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# ============================================
# 1. WebShop 1.5B DUET v2 (SC beta=0.1)
# ============================================
echo ">>> [1/2] WebShop 1.5B DUET v2"
bash start_env_webshop.sh
run_experiment \
    "config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v2.yaml" \
    "webshop_qwen1.5b_duet_v2"
bash start_env_webshop.sh stop 2>/dev/null || true

# ============================================
# 2. ALFWorld 3B CHORD
# ============================================
echo ">>> [2/2] ALFWorld 3B CHORD"
bash start_env_alfworld.sh
nohup bash watchdog_agentgym.sh > /dev/null 2>&1 & disown
run_experiment \
    "config/duet_paper_experiments_configs/alfworld/alfworld_qwen3b_chord.yaml" \
    "alfworld_qwen3b_chord"
if [ -f /tmp/watchdog_alfworld.pid ]; then
    kill $(cat /tmp/watchdog_alfworld.pid) 2>/dev/null || true
    rm -f /tmp/watchdog_alfworld.pid
fi
bash start_env_alfworld.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " 补跑完成!"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
