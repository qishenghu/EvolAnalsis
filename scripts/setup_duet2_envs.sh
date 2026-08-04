#!/bin/bash
# Build the two new conda envs for Qwen3.5 training (ICLR2027_PLAN §3.2, route (b)).
#   duet2 : training stack — verl 0.4.0.dev0 (vendored copy) + transformers 5.5.1 + torch 2.9, NO vllm
#   vllm2 : serving stack — vllm 0.20.2 (student rollout servers + Qwen3.5-122B teacher)
# Safe to re-run; never touches the 'duet' env.
set -x
CONDA=/data/home/qisheng/miniconda3
DUET_SP=$CONDA/envs/duet/lib/python3.11/site-packages

# ---------- duet2 ----------
if [ ! -d "$CONDA/envs/duet2" ]; then
  $CONDA/bin/conda create -n duet2 python=3.12 -y
fi
P2=$CONDA/envs/duet2/bin/pip
PY2=$CONDA/envs/duet2/bin/python

$P2 install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 || \
  $P2 install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
$P2 install "transformers==5.5.1" accelerate datasets peft safetensors \
  "ray[default]==2.47.1" hydra-core omegaconf codetiming dill pybind11 \
  tensordict pandas numpy'<2.0.0' wandb fastapi uvicorn aiohttp openai \
  python-dotenv loguru scikit-learn tensorboard torchdata "pyarrow>=15" pylatexenc

# GDN fast kernels (HF training path falls back to slow PyTorch ops without these)
$P2 install flash-linear-attention==0.5.2 || $P2 install flash-linear-attention==0.4.2
$P2 install causal-conv1d --no-build-isolation || echo "WARN: causal-conv1d build failed (slow fallback ok for smoke)"

# flash-attn: try official wheel path first, then source build (sm80 only)
$P2 install flash-attn==2.8.3.post1 --no-build-isolation || \
  MAX_JOBS=32 FLASH_ATTENTION_FORCE_BUILD=TRUE $P2 install flash-attn==2.8.3.post1 --no-build-isolation || \
  echo "WARN: flash-attn build failed — Gate-S can run with sdpa + use_remove_padding=False"

# vendor verl 0.4.0.dev0 byte-identical from duet env (no network, deterministic)
D2_SP=$($PY2 -c 'import site;print(site.getsitepackages()[0])')
rsync -a --delete $DUET_SP/verl $D2_SP/
rsync -a $DUET_SP/verl-0.4.0.dev0.dist-info $D2_SP/ 2>/dev/null || true
$PY2 -c "import verl; print('verl vendored:', verl.__version__)"

# ---------- vllm2 ----------
if [ ! -d "$CONDA/envs/vllm2" ]; then
  $CONDA/bin/conda create -n vllm2 python=3.12 -y
fi
PV=$CONDA/envs/vllm2/bin/pip
$PV install vllm==0.20.2
$CONDA/envs/vllm2/bin/python -c "import vllm; print('vllm:', vllm.__version__)"

echo "=== ENV BUILD COMPLETE ==="
