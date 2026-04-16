#!/bin/bash
# ==============================================================================
# New Server Setup for Qwen3-30B Teacher Trajectory Collection
#
# This script sets up everything needed to collect teacher trajectories
# on a fresh server. Run once, then use run_qwen3_teacher_parallel.sh.
#
# Prerequisites:
#   - NVIDIA GPU(s) with ≥80GB VRAM (H100/H200/A100)
#   - conda installed (or miniconda)
#   - Internet access for model download
#
# Usage:
#   bash setup_new_server.sh
#
# After setup:
#   nohup bash run_qwen3_teacher_parallel.sh webshop > logs/teacher_webshop.log 2>&1 &
#   nohup bash run_qwen3_teacher_parallel.sh sciworld > logs/teacher_sciworld.log 2>&1 &
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ===== Auto-detect conda =====
if [ -z "$CONDA_PATH" ]; then
    if [ -n "$CONDA_EXE" ]; then
        CONDA_PATH="$(dirname $(dirname $CONDA_EXE))"
    elif [ -d "/data/code/exp/anaconda3" ]; then
        CONDA_PATH="/data/code/exp/anaconda3"
    elif [ -d "$HOME/anaconda3" ]; then
        CONDA_PATH="$HOME/anaconda3"
    elif [ -d "$HOME/miniconda3" ]; then
        CONDA_PATH="$HOME/miniconda3"
    else
        echo "ERROR: Cannot find conda. Set CONDA_PATH env var."
        exit 1
    fi
fi
echo "Using conda at: $CONDA_PATH"

# Source conda
eval "$($CONDA_PATH/bin/conda shell.bash hook)"

# ===== Check GPU =====
echo ""
echo "=== GPU Check ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || {
    echo "ERROR: nvidia-smi failed. GPU driver not installed?"
    exit 1
}
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Found $GPU_COUNT GPU(s)"

# ===== Step 1: Create duet conda env =====
echo ""
echo "=== Step 1: Setting up 'duet' conda environment ==="
if conda env list | grep -q "^duet "; then
    echo "  'duet' env already exists, skipping creation"
else
    echo "  Creating 'duet' env (python 3.11)..."
    conda create -n duet python=3.11 -y
fi

echo "  Installing packages in 'duet'..."
conda activate duet
pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || \
    pip install -q torch --index-url https://download.pytorch.org/whl/cu124
pip install -q "vllm>=0.8.5"
pip install -q transformers huggingface_hub loguru omegaconf tqdm jieba ray

# Verify vLLM
python -c "import vllm; print(f'  vLLM {vllm.__version__} OK')"
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# ===== Step 2: WebShop environment =====
echo ""
echo "=== Step 2: Setting up WebShop environment ==="
if conda env list | grep -q "^agentenv-webshop "; then
    echo "  'agentenv-webshop' env already exists"
else
    echo "  Creating 'agentenv-webshop' env..."
    conda create -n agentenv-webshop python=3.9 -y
    conda activate agentenv-webshop
    if [ -d "$SCRIPT_DIR/AgentGym/agentenv-webshop" ]; then
        cd "$SCRIPT_DIR/AgentGym/agentenv-webshop"
        pip install -q -e .
        cd "$SCRIPT_DIR"
    else
        echo "  WARNING: AgentGym/agentenv-webshop not found. Install manually."
    fi
    # Java for WebShop
    conda install -c conda-forge openjdk=11 -y 2>/dev/null || \
        echo "  WARNING: Failed to install Java. Install manually."
fi

# ===== Step 3: SciWorld environment =====
echo ""
echo "=== Step 3: Setting up SciWorld environment ==="
if conda env list | grep -q "^agentenv-sciworld "; then
    echo "  'agentenv-sciworld' env already exists"
else
    echo "  Creating 'agentenv-sciworld' env..."
    conda create -n agentenv-sciworld python=3.8 -y
    conda activate agentenv-sciworld
    if [ -d "$SCRIPT_DIR/AgentGym/agentenv-sciworld" ]; then
        cd "$SCRIPT_DIR/AgentGym/agentenv-sciworld"
        pip install -q -e .
        cd "$SCRIPT_DIR"
    else
        echo "  WARNING: AgentGym/agentenv-sciworld not found. Install manually."
    fi
    # Java for SciWorld
    conda install -c conda-forge openjdk=11 -y 2>/dev/null || \
        echo "  WARNING: Failed to install Java. Install manually."
fi

# ===== Step 4: Download teacher model =====
echo ""
echo "=== Step 4: Downloading Qwen3-30B-A3B-Thinking-2507 (~61GB) ==="
conda activate duet
MODEL_DIR="$SCRIPT_DIR/models/Qwen/Qwen3-30B-A3B-Thinking-2507"
if [ -f "$MODEL_DIR/config.json" ]; then
    SHARDS=$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l)
    echo "  Model already exists ($SHARDS shards)"
else
    echo "  Downloading... (this will take a while)"
    mkdir -p "$MODEL_DIR"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-30B-A3B-Thinking-2507',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4,
)
print('  Model download complete!')
"
fi

# ===== Step 5: Verify setup =====
echo ""
echo "=== Step 5: Verification ==="
conda activate duet

# Check model
if [ -f "$MODEL_DIR/config.json" ]; then
    echo "  [OK] Teacher model: $MODEL_DIR"
else
    echo "  [FAIL] Teacher model not found"
fi

# Check task files
for env in webshop sciworld; do
    f="data/${env}/task_ids_800_seed2026.txt"
    if [ -f "$f" ]; then
        echo "  [OK] Task file: $f ($(wc -l < $f) tasks)"
    else
        echo "  [FAIL] Task file not found: $f"
    fi
done

# Check env binaries
for envname in agentenv-webshop agentenv-sciworld; do
    binname=$(echo $envname | sed 's/agentenv-//')
    binpath="$CONDA_PATH/envs/$envname/bin/$binname"
    if [ -x "$binpath" ]; then
        echo "  [OK] $envname binary: $binpath"
    else
        echo "  [WARN] $envname binary not found at $binpath"
    fi
done

# Check Java
if command -v java &>/dev/null || [ -x "$CONDA_PATH/envs/agentenv-sciworld/lib/jvm/bin/java" ]; then
    echo "  [OK] Java available"
else
    echo "  [WARN] Java not found (needed for SciWorld)"
fi

echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   # Set CONDA_PATH if not auto-detected:"
echo "   export CONDA_PATH=$CONDA_PATH"
echo ""
echo "   # Collect WebShop teacher trajectories:"
echo "   nohup bash run_qwen3_teacher_parallel.sh webshop > logs/teacher_webshop.log 2>&1 &"
echo ""
echo "   # Collect SciWorld teacher trajectories:"
echo "   nohup bash run_qwen3_teacher_parallel.sh sciworld > logs/teacher_sciworld.log 2>&1 &"
echo ""
echo "   # Monitor progress:"
echo "   tail -f logs/collect_webshop_partA.log"
echo "============================================================"
