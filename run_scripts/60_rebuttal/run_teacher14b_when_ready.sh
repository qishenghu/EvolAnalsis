#!/bin/bash
# Waits for the Qwen2.5-14B download to be COMPLETE (all shards present, not just
# config.json), then brings up the AUX ALFWorld stack and starts teacher sampling
# on the spare GPU. Safe to run alongside the main rebuttal queue: the aux stack
# uses ports 18011/18091, outside this host's ephemeral range.
#
# Usage: nohup bash run_teacher14b_when_ready.sh > logs/teacher14b_autostart.log 2>&1 &
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=/data/shared_models/Qwen2.5-14B-Instruct
GPU="${GPU:-6}"
MAXWAIT=${MAXWAIT:-240}   # 240 x 60s = 4h

complete() {
    [ -f "$MODEL/model.safetensors.index.json" ] || return 1
    /data/home/qisheng/miniconda3/envs/duet/bin/python - "$MODEL" <<'PY'
import json, os, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
missing = [s for s in set(idx["weight_map"].values()) if not os.path.exists(os.path.join(d, s))]
sys.exit(1 if missing else 0)
PY
}

echo "[$(date '+%m-%d %H:%M')] waiting for $MODEL to finish downloading..."
for i in $(seq 1 $MAXWAIT); do
    if complete; then echo "[$(date '+%m-%d %H:%M')] model complete after ${i} min"; break; fi
    if [ "$i" = "$MAXWAIT" ]; then echo "TIMEOUT waiting for model download"; exit 1; fi
    sleep 60
done

# Free GPU check — never contend with the main queue's GPUs (0,1,2,4)
used=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 20000 ]; then
    echo "GPU $GPU busy (${used} MiB) — not starting sampling. Rerun when free."
    exit 1
fi

echo "[$(date '+%m-%d %H:%M')] starting AUX ALFWorld stack (18011/18091)"
bash start_env_alfworld_aux.sh || { echo "aux env failed"; exit 1; }

echo "[$(date '+%m-%d %H:%M')] starting 14B teacher sampling on GPU $GPU"
GPUS=$GPU TP=1 MODEL=$MODEL MODEL_NAME=qwen14b ENV_URL=http://localhost:18091 \
    bash run_a100_teacher_sampling.sh
rc=$?
echo "[$(date '+%m-%d %H:%M')] sampling finished rc=$rc"
bash start_env_alfworld_aux.sh stop 2>/dev/null || true
