#!/bin/bash
# ==============================================================================
# LUFFY rerun on L20X — sanity check whether the 49.5% baseline reproduces.
#
# Pure LUFFY: use_chord=false, use_dr3=false, teacher_policy_shaping=p_div_p_beta,
# no State Channel. Same env, same model, same seed (2026), same code.
#
# If reproduction lands ~49.5%: our 36.5% gap_bok is real (DUET* below LUFFY).
# If reproduction lands ~30-40%: 49.5% was infra-inflated; we've been chasing
# a phantom target and our 36.5% is actually competitive.
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
LOG="logs/luffy_rerun.log"
mkdir -p logs

NAME="ws_luffy_rerun_2026_05_03"
CFG="config/duet_paper_experiments_configs/webshop/${NAME}.yaml"

echo "[$(date '+%m-%d %H:%M')] LUFFY rerun starting" | tee -a "$LOG"

if [ -f "experiments/webshop/${NAME}/validation_log/100.jsonl" ]; then
    echo "[$(date '+%m-%d %H:%M')] SKIP (already done): $NAME" | tee -a "$LOG"
    exit 0
fi

echo "[$(date '+%m-%d %H:%M')] [pre-$NAME] env restart..." | tee -a "$LOG"
bash start_env_alfworld.sh stop 2>&1 | tail -1
bash start_env_webshop.sh  stop 2>&1 | tail -1
sleep 8
bash start_env_webshop.sh
sleep 5

ray_tmp="${RAY_TMPDIR}/${NAME}"
mkdir -p "$ray_tmp"; rm -rf "$ray_tmp"/session_* 2>/dev/null || true

echo "[$(date '+%m-%d %H:%M')] RUN: $NAME" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
    python launcher.py --conf "$CFG" \
    > "logs/${NAME}.log" 2>&1
rc=$?

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
echo "[$(date '+%m-%d %H:%M')] LUFFY rerun COMPLETE." | tee -a "$LOG"
