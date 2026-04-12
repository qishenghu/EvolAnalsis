#!/usr/bin/env bash
# ==============================================================================
# One-shot setup for DUET experiments on a new server.
#
# What it does:
#   1. Creates "duet" conda env (Python 3.11) + installs all training deps
#   2. Creates "agentenv-webshop" conda env (Python 3.8) + builds search index
#   3. Installs ALFWorld + AgentGym wrappers into duet env
#   4. Downloads ALFWorld game data
#   5. Creates .env file
#
# Prerequisites:
#   - conda installed and in PATH
#   - CUDA drivers installed
#   - sudo access for creating /data/ray (or create it manually beforehand)
#
# Usage:
#   bash setup_envs.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DUET_ENV="duet"
WEBSHOP_ENV="agentenv-webshop"
CONDA_BASE="$(conda info --base)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING:${NC} $1"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $1"; }

# ---- Check prerequisites ----
if ! command -v conda &> /dev/null; then
    err "conda not found. Install Miniconda first."
    exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "============================================"
echo " DUET Environment Setup"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

# ==============================================================================
# Step 1: Create duet conda env
# ==============================================================================
if conda info --envs | awk '{print $1}' | grep -qx "$DUET_ENV"; then
    log "Conda env '$DUET_ENV' already exists, skipping creation."
else
    log "Creating conda env '$DUET_ENV' (Python 3.11)..."
    conda create -y -n "$DUET_ENV" python=3.11

    conda activate "$DUET_ENV"

    log "Installing CUDA toolkit..."
    conda install -y -c nvidia cuda-toolkit

    log "Installing Python dependencies from requirements.txt..."
    pip install -r requirements.txt

    log "Installing flash-attn..."
    export TMPDIR="${TMPDIR:-$HOME/tmp}"
    mkdir -p "$TMPDIR"
    pip install --verbose flash-attn==2.7.4.post1 ring-flash-attn --no-build-isolation

    conda deactivate
fi

# ==============================================================================
# Step 2: Install ALFWorld into duet env
# ==============================================================================
log "Installing ALFWorld into '$DUET_ENV' env..."
conda activate "$DUET_ENV"

# alfworld + headless opencv
pip install alfworld==0.3.3 2>/dev/null | tail -3
pip uninstall opencv-python -y 2>/dev/null || true
pip install opencv-python-headless 2>/dev/null | tail -1

# agentenv_alfworld wrapper
cd "$SCRIPT_DIR/AgentGym/agentenv-alfworld"
pip install -e . 2>/dev/null | tail -3

# Ensure numpy compatible with numba (vLLM dependency)
pip install "numpy<2.3" 2>/dev/null | tail -1

conda deactivate
cd "$SCRIPT_DIR"

# ==============================================================================
# Step 3: Download ALFWorld game data
# ==============================================================================
ALFWORLD_DATA="$HOME/alfworld_data"
if [ -d "$ALFWORLD_DATA/json_2.1.1" ]; then
    log "ALFWorld data already exists at $ALFWORLD_DATA, skipping download."
else
    log "Downloading ALFWorld game data to $ALFWORLD_DATA..."
    ALFWORLD_DATA="$ALFWORLD_DATA" conda run -n "$DUET_ENV" alfworld-download
fi

# Symlink to default cache location
if [ ! -e "$HOME/.cache/alfworld" ]; then
    mkdir -p "$HOME/.cache"
    ln -s "$ALFWORLD_DATA" "$HOME/.cache/alfworld"
    log "Symlinked $ALFWORLD_DATA -> ~/.cache/alfworld"
fi

# ==============================================================================
# Step 4: Create agentenv-webshop conda env
# ==============================================================================
if conda info --envs | awk '{print $1}' | grep -qx "$WEBSHOP_ENV"; then
    log "Conda env '$WEBSHOP_ENV' already exists, skipping creation."
else
    log "Creating conda env '$WEBSHOP_ENV' (Python 3.8 + faiss + openjdk)..."
    conda create -y -n "$WEBSHOP_ENV" python=3.8 faiss-cpu=1.7 openjdk=11 -c conda-forge -c defaults

    conda activate "$WEBSHOP_ENV"

    log "Installing WebShop dependencies..."
    cd "$SCRIPT_DIR/AgentGym/agentenv-webshop/webshop"
    pip install -r requirements.txt 2>/dev/null | tail -3
    pip install -U "Werkzeug>=2,<3" "mkl>=2021,<2022" "typing_extensions<4.6.0" "gym==0.23.1" 2>/dev/null | tail -3
    pip install "en-core-web-lg@https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.3.0/en_core_web_lg-3.3.0-py3-none-any.whl" 2>/dev/null | tail -3

    log "Building WebShop search engine indexes..."
    cd search_engine
    mkdir -p resources resources_100 resources_1k resources_100k
    python convert_product_file_format.py 2>&1 | tail -3
    mkdir -p indexes indexes_100 indexes_1k indexes_100k
    bash run_indexing.sh 2>&1 | tail -3

    cd "$SCRIPT_DIR/AgentGym/agentenv-webshop"
    pip install -e . 2>/dev/null | tail -3
    pip install "numpy==1.22.4" 2>/dev/null | tail -1
    pip install "typing_extensions<4.6.0" 2>/dev/null | tail -1

    conda deactivate
    cd "$SCRIPT_DIR"
fi

# ==============================================================================
# Step 5: Create temp/ray directories
# ==============================================================================
log "Creating temp directories..."
mkdir -p "$HOME/tmp" 2>/dev/null || true

if [ -d "/data/ray" ] && [ -w "/data/ray" ]; then
    log "/data/ray already exists and is writable."
elif [ -d "/data" ]; then
    warn "/data/ray does not exist or is not writable."
    echo "  Please run:  sudo mkdir -p /data/ray && sudo chmod 777 /data/ray"
    echo "  Or set RAY_TMPDIR in env_config.sh to a writable location."
else
    warn "/data partition not found. Using ~/ray_tmp instead."
    mkdir -p "$HOME/ray_tmp"
    echo "  Set RAY_TMPDIR=$HOME/ray_tmp in env_config.sh"
fi

# ==============================================================================
# Step 6: Create .env file
# ==============================================================================
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    log "Creating .env file..."
    cat > "$SCRIPT_DIR/.env" << EOF
HF_ENDPOINT=https://hf-mirror.com

# ALFWorld
ALFWORLD_DATA=$ALFWORLD_DATA
ALFWORLD_SERVER_URL=http://127.0.0.1:36001

# WebShop
WEBSHOP_SERVER_URL=http://127.0.0.1:36003
EOF
    log ".env created."
else
    log ".env already exists, skipping."
fi

# ==============================================================================
# Step 7: Verify installation
# ==============================================================================
echo ""
log "Verifying installation..."

echo -n "  duet env: "
conda run -n "$DUET_ENV" python -c "
import torch, vllm, ray, alfworld, cv2
print(f'torch={torch.__version__} vllm={vllm.__version__} ray={ray.__version__} alfworld={alfworld.__version__} cv2={cv2.__version__}')
" 2>/dev/null || echo "FAILED"

echo -n "  webshop env: "
conda run -n "$WEBSHOP_ENV" python -c "
import gym, spacy, agentenv_webshop
print(f'gym={gym.__version__} spacy={spacy.__version__} agentenv_webshop OK')
" 2>/dev/null || echo "FAILED"

echo -n "  ALFWorld data: "
[ -d "$ALFWORLD_DATA/json_2.1.1/train" ] && echo "OK ($ALFWORLD_DATA)" || echo "MISSING"

echo -n "  WebShop indexes: "
[ -d "$SCRIPT_DIR/AgentGym/agentenv-webshop/webshop/search_engine/indexes_1k" ] && echo "OK" || echo "MISSING"

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " Quick start:"
echo "   conda activate $DUET_ENV"
echo "   bash start_env_alfworld.sh       # Start ALFWorld env"
echo "   bash start_env_webshop.sh        # Start WebShop env"
echo "   bash run_7b_baselines_alfworld.sh  # Run experiments"
echo ""
echo " If paths differ on this server, edit env_config.sh"
echo "============================================"
