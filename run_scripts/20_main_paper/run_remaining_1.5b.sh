#!/bin/bash
# ==============================================================================
# Remaining 1.5B experiments on GPU 4-7 (sequential, 1 at a time)
#
# 6 experiments:
#   1. ALFWorld  - SFT+GRPO
#   2. WebShop   - OnPolicy
#   3. WebShop   - LUFFY
#   4. WebShop   - CHORD
#   5. WebShop   - DUET
#   6. WebShop   - SFT+GRPO
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"  # Default to GPU 0-3, override with CUDA_GPUS env var
ALF_CONFIG="config/duet_paper_experiments_configs/alfworld"
WS_CONFIG="config/duet_paper_experiments_configs/webshop"

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
echo " Remaining 1.5B Experiments (GPU $GPUS)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " 6 experiments, sequential"
echo "============================================"

# ============================================
# 1. ALFWorld - SFT+GRPO
# ============================================
echo ""
echo ">>> [1/6] ALFWorld SFT+GRPO"
bash start_env_alfworld.sh

# Phase 1: SFT
run_experiment "${ALF_CONFIG}/alfworld_qwen1.5b_sft.yaml" "alfworld_qwen1.5b_sft"

# Verify and convert SFT checkpoint to HF format
SFT_CKPT="checkpoints/agentevolver/alfworld_qwen1.5b_sft/global_step_50/actor"
SFT_CKPT_HF="${SFT_CKPT}_hf"
if [ ! -d "$SFT_CKPT" ]; then
    echo "ERROR: SFT checkpoint not found at $SFT_CKPT"
    ls -d checkpoints/agentevolver/alfworld_qwen1.5b_sft/global_step_*/actor 2>/dev/null
    exit 1
fi
if [ ! -f "$SFT_CKPT_HF/model.safetensors.index.json" ] && [ ! -f "$SFT_CKPT_HF/model.safetensors" ]; then
    echo "Converting FSDP checkpoint to HF format..."
    python scripts/merge_fsdp_checkpoint.py \
        --ckpt_dir "$SFT_CKPT" \
        --base_model /data/shared_models/Qwen2.5-1.5B-Instruct \
        --output_dir "$SFT_CKPT_HF"
fi

# Phase 2: RL from SFT checkpoint
run_experiment "${ALF_CONFIG}/alfworld_qwen1.5b_sft_rl.yaml" "alfworld_qwen1.5b_sft_rl"

bash start_env_alfworld.sh stop 2>/dev/null || true

# ============================================
# 2-5. WebShop - OnPolicy / LUFFY / CHORD / DUET
# ============================================
echo ""
echo ">>> Starting WebShop environment"
bash start_env_webshop.sh

echo ">>> [2/6] WebShop OnPolicy"
run_experiment "${WS_CONFIG}/webshop_qwen1.5b_onpolicy.yaml" "webshop_qwen1.5b_onpolicy"

echo ">>> [3/6] WebShop LUFFY"
run_experiment "${WS_CONFIG}/webshop_qwen1.5b_luffy.yaml" "webshop_qwen1.5b_luffy"

echo ">>> [4/6] WebShop CHORD"
run_experiment "${WS_CONFIG}/webshop_qwen1.5b_chord.yaml" "webshop_qwen1.5b_chord"

echo ">>> [5/6] WebShop DUET"
run_experiment "${WS_CONFIG}/webshop_qwen1.5b_duet.yaml" "webshop_qwen1.5b_duet"

# ============================================
# 6. WebShop - SFT+GRPO
# ============================================
echo ">>> [6/6] WebShop SFT+GRPO"

# Phase 1: SFT
run_experiment "${WS_CONFIG}/webshop_qwen1.5b_sft.yaml" "webshop_qwen1.5b_sft"

SFT_CKPT="checkpoints/agentevolver/webshop_qwen1.5b_sft/global_step_50/actor"
SFT_CKPT_HF="${SFT_CKPT}_hf"
if [ ! -d "$SFT_CKPT" ]; then
    echo "ERROR: SFT checkpoint not found at $SFT_CKPT"
    ls -d checkpoints/agentevolver/webshop_qwen1.5b_sft/global_step_*/actor 2>/dev/null
    exit 1
fi
if [ ! -f "$SFT_CKPT_HF/model.safetensors.index.json" ] && [ ! -f "$SFT_CKPT_HF/model.safetensors" ]; then
    echo "Converting FSDP checkpoint to HF format..."
    python scripts/merge_fsdp_checkpoint.py \
        --ckpt_dir "$SFT_CKPT" \
        --base_model /data/shared_models/Qwen2.5-1.5B-Instruct \
        --output_dir "$SFT_CKPT_HF"
fi

# Phase 2: RL from SFT checkpoint
run_experiment "${WS_CONFIG}/webshop_qwen1.5b_sft_rl.yaml" "webshop_qwen1.5b_sft_rl"

bash start_env_webshop.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " All 6 experiments complete!"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
