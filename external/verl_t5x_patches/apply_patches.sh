#!/bin/bash
# ==============================================================================
# Apply transformers-5.x / no-vllm compatibility patches to a vendored verl
# 0.4.0.dev0 install, then run import proofs.
#
# Usage:
#   bash apply_patches.sh [conda_env_name]   # default: duet2
#   bash apply_patches.sh /path/to/site-packages/verl
#
# Import proofs run only when a conda env name is given (needs that env's python).
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../env_config.sh"

TARGET="${1:-duet2}"

if [ -d "${TARGET}" ]; then
    VERL_ROOT="${TARGET}"
    PYTHON="python3"
    RUN_PROOFS=0
else
    ENV_NAME="${TARGET}"
    PYTHON="${CONDA_PATH}/envs/${ENV_NAME}/bin/python"
    if [ ! -x "${PYTHON}" ]; then
        echo "ERROR: ${PYTHON} not found (env '${ENV_NAME}' missing?)" >&2
        exit 1
    fi
    VERL_ROOT="$("${PYTHON}" -c 'import importlib.util; s = importlib.util.find_spec("verl"); print(s.submodule_search_locations[0])')"
    RUN_PROOFS=1
fi

echo "Patching verl at: ${VERL_ROOT}"
"${PYTHON}" "${SCRIPT_DIR}/patch_verl.py" "${VERL_ROOT}"

if [ "${RUN_PROOFS}" = "1" ]; then
    echo ""
    echo "=== Import proofs (${ENV_NAME}) ==="
    # Model dir for the config-load proof: cluster-portable via env_config.sh.
    export DUET_QWEN35_4B_DIR="${DUET_MODELS_DIR:-/data/shared_models}/Qwen3.5-4B"
    "${PYTHON}" - <<'EOF'
import transformers

print(f"transformers {transformers.__version__}")

import verl

print(f"verl {verl.__version__} imported OK")

from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # noqa: F401

print("RayPPOTrainer import OK")

from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker  # noqa: F401

print("fsdp_workers import OK")

from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler  # noqa: F401

print("ChatCompletionScheduler import OK (no vllm required)")

import os

cfg = transformers.AutoConfig.from_pretrained(os.environ["DUET_QWEN35_4B_DIR"])
print(f"Qwen3.5-4B config OK: model_type={cfg.model_type}, architectures={cfg.architectures}")
EOF
    echo "=== All import proofs passed ==="
fi
