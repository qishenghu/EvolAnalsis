#!/bin/bash
# ============================================================================
# Download mid-quality teacher models for the rebuttal teacher-quality ablation.
# Runs on network only (no GPU) — safe to run while training occupies GPUs.
#
# Usage:
#   bash run_download_teacher_models.sh          # Qwen2.5-14B-Instruct (~28GB)
#   bash run_download_teacher_models.sh 32b      # also Qwen2.5-32B-Instruct (~65GB)
# ============================================================================
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env_config.sh"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

DEST=/data/shared_models

dl() {
    local repo=$1 dir=$2
    # Completeness check: index must exist AND all shards it lists must be present.
    # (A bare config.json means an aborted download — do not trust it.)
    if [ -f "$DEST/$dir/model.safetensors.index.json" ]; then
        local missing
        missing=$(python - "$DEST/$dir" <<'PY'
import json, os, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
shards = set(idx["weight_map"].values())
print(sum(1 for s in shards if not os.path.exists(os.path.join(d, s))))
PY
)
        if [ "$missing" = "0" ]; then
            echo "$dir already complete — skipping"
            return
        fi
        echo "$dir incomplete ($missing shards missing) — resuming download"
    fi
    echo "Downloading $repo -> $DEST/$dir ..."
    huggingface-cli download "$repo" --local-dir "$DEST/$dir" \
        --exclude "*.pth" "original/*" 2>&1 | tail -2 \
    || { echo "huggingface-cli failed; trying modelscope..."; \
         python -c "from modelscope import snapshot_download; snapshot_download('$repo', local_dir='$DEST/$dir')"; }
    ls "$DEST/$dir/config.json" && echo "$dir OK"
}

dl Qwen/Qwen2.5-14B-Instruct Qwen2.5-14B-Instruct
if [ "${1:-}" = "32b" ]; then
    dl Qwen/Qwen2.5-32B-Instruct Qwen2.5-32B-Instruct
fi
echo "DOWNLOADS DONE"
