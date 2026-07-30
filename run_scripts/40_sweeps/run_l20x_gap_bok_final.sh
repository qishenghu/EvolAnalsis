#!/bin/bash
# ==============================================================================
# L20X gap-mode best-of-k FINAL queue (2026-05-03 02:45):
#
# After velocity-mode (v1/v2 latch) capped at 25-29% on 3B WS, pivot fully to
# gap mode + best-of-k variant per A100's GAP_MODE_BEST_OF_K_PROPOSAL doc.
#
# 3 peak variants of best-of-k gap (all valley=0.05, token_weighting=true,
# gap_decay_gamma=0.0). Most-likely-winner first.
#
# Total wall-clock: 3 × 3.5h ≈ 10.5h. Started 02:45 → ends ~13:15 today.
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
LOG="logs/l20x_gap_bok_final.log"
mkdir -p logs

echo "[$(date '+%m-%d %H:%M')] L20X gap-bok FINAL queue starting" | tee -a "$LOG"

run_one() {
    local config=$1
    local name=$2

    if [ ! -f "$config" ]; then
        echo "[$(date '+%m-%d %H:%M')] SKIP (missing): $name" | tee -a "$LOG"
        return 1
    fi
    if [ -f "experiments/webshop/${name}/validation_log/100.jsonl" ]; then
        echo "[$(date '+%m-%d %H:%M')] SKIP (done): $name" | tee -a "$LOG"
        return 0
    fi

    echo "[$(date '+%m-%d %H:%M')] [pre-$name] env restart..." | tee -a "$LOG"
    bash start_env_alfworld.sh stop 2>&1 | tail -1
    bash start_env_webshop.sh  stop 2>&1 | tail -1
    sleep 8
    bash start_env_webshop.sh
    sleep 5

    local ray_tmp="${RAY_TMPDIR}/${name}"
    mkdir -p "$ray_tmp"; rm -rf "$ray_tmp"/session_* 2>/dev/null || true

    echo "[$(date '+%m-%d %H:%M')] RUN: $name" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
        python launcher.py --conf "$config" \
        > "logs/${name}.log" 2>&1
    local rc=$?

    if [ -f "experiments/webshop/${name}/validation_log/100.jsonl" ]; then
        local sr=$(python -c "
import json
n=0; sr=0
with open('experiments/webshop/${name}/validation_log/100.jsonl') as fh:
  for line in fh:
    d=json.loads(line); n+=1
    if d.get('score',d.get('reward',0))>=1.0: sr+=1
print(f'{sr/n*100:.1f}')
" 2>/dev/null)
        echo "[$(date '+%m-%d %H:%M')] DONE: $name val@100 SR=${sr}% (rc=$rc)" | tee -a "$LOG"
    else
        echo "[$(date '+%m-%d %H:%M')] FAILED: $name (rc=$rc)" | tee -a "$LOG"
    fi
}

# Order rationale (most-likely-winner first):
#   1. pk020 — A100's primary recommendation, swE_02-style μ steady ~0.10
#   2. pk015 — lower peak, hedge toward "constant low μ" empirical sweet spot
#   3. pk025 — slightly higher peak, test if more BC scaffolding helps capability
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_gap_bok_pk02.yaml"   "ws_swC_v_gap_bok_pk02"
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_gap_bok_pk015.yaml" "ws_swC_v_gap_bok_pk015"
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_gap_bok_pk025.yaml" "ws_swC_v_gap_bok_pk025"

bash start_env_webshop.sh stop 2>&1 | tail -1

echo "" | tee -a "$LOG"
echo "[$(date '+%m-%d %H:%M')] L20X gap-bok FINAL queue COMPLETE." | tee -a "$LOG"

echo "=== L20X gap-bok leaderboard ===" | tee -a "$LOG"
python <<'PYEOF' | tee -a "$LOG"
import json, os
runs = ["ws_swC_v_gap_bok_pk02", "ws_swC_v_gap_bok_pk015", "ws_swC_v_gap_bok_pk025"]
for n in runs:
    f = f"experiments/webshop/{n}/validation_log/100.jsonl"
    if os.path.exists(f):
        cnt=0; sr=0
        with open(f) as fh:
            for line in fh:
                d=json.loads(line); cnt += 1
                if d.get("score", d.get("reward", 0)) >= 1.0: sr += 1
        if cnt > 0: print(f"  {sr/cnt*100:5.1f}%  {n}")
print(f"  Targets: LUFFY 49.5% | DUET v1 53.0%")
PYEOF
