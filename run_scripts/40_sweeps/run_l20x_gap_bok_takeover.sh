#!/bin/bash
# ==============================================================================
# L20X gap-mode best-of-k takeover (2026-05-03 02:35):
#
# Replaces the queued ws_swC_v_pk03_aggr (low-info, 5th velocity-mode variant)
# with ws_swC_v_gap_bok_pk02 (gap mode + best-of-k + token weighting).
#
# Trigger: when ws_swC_v_pk05/validation_log/100.jsonl appears (pk05 done),
# we kill the old orchestrator before it can start pk03_aggr, then launch
# gap_bok_pk02 fresh.
#
# Per A100 agent's GAP_MODE_BEST_OF_K_PROPOSAL_2026-05-03 doc.
# ==============================================================================

set +e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source env_config.sh

if ! command -v conda >/dev/null 2>&1; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
LOG="logs/l20x_gap_bok_takeover.log"
mkdir -p logs

SENTINEL="experiments/webshop/ws_swC_v_pk05/validation_log/100.jsonl"

echo "[$(date '+%m-%d %H:%M')] watchdog: waiting for pk05 to finish..." | tee -a "$LOG"
while [ ! -f "$SENTINEL" ]; do
    sleep 120
done
echo "[$(date '+%m-%d %H:%M')] pk05 DONE — taking over before pk03_aggr starts." | tee -a "$LOG"

# Phase 1: kill the old orchestrator + any lingering launcher (pk05 cleanup phase)
echo "[$(date '+%m-%d %H:%M')] killing old orchestrator + lingering launcher..." | tee -a "$LOG"
pgrep -f "run_l20x_velocity_v2.sh" | xargs -r kill -9 2>/dev/null
sleep 3
pgrep -f "launcher.py" | xargs -r kill -9 2>/dev/null
pgrep -f "ray::"       | xargs -r kill -9 2>/dev/null
sleep 5

# Phase 2: env restart cleanup
bash start_env_alfworld.sh stop 2>&1 | tail -1
bash start_env_webshop.sh  stop 2>&1 | tail -1
sleep 8

# Phase 3: launch gap_bok_pk02
NAME="ws_swC_v_gap_bok_pk02"
CFG="config/duet_paper_experiments_configs/webshop/sweep_phase_c/${NAME}.yaml"

if [ -f "experiments/webshop/${NAME}/validation_log/100.jsonl" ]; then
    echo "[$(date '+%m-%d %H:%M')] SKIP (already done): $NAME" | tee -a "$LOG"
    exit 0
fi

echo "[$(date '+%m-%d %H:%M')] [pre-$NAME] starting fresh webshop env..." | tee -a "$LOG"
bash start_env_webshop.sh
sleep 5

ray_tmp="${RAY_TMPDIR}/${NAME}"
mkdir -p "$ray_tmp"; rm -rf "$ray_tmp"/session_* 2>/dev/null || true

echo "[$(date '+%m-%d %H:%M')] RUN: $NAME" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
    python launcher.py --conf "$CFG" \
    > "logs/${NAME}.log" 2>&1
rc=$?

# Report result
if [ -f "experiments/webshop/${NAME}/validation_log/100.jsonl" ]; then
    sr=$(python -c "
import json
n=0; sr=0
with open('experiments/webshop/${NAME}/validation_log/100.jsonl') as fh:
  for line in fh:
    d=json.loads(line); n+=1
    if d.get('score',d.get('reward',0))>=1.0: sr+=1
print(f'{sr/n*100:.1f}')
" 2>/dev/null)
    echo "[$(date '+%m-%d %H:%M')] DONE: $NAME val@100 SR=${sr}% (rc=$rc)" | tee -a "$LOG"
else
    echo "[$(date '+%m-%d %H:%M')] FAILED: $NAME (rc=$rc)" | tee -a "$LOG"
fi

bash start_env_webshop.sh stop 2>&1 | tail -1
echo "[$(date '+%m-%d %H:%M')] gap_bok takeover COMPLETE." | tee -a "$LOG"
