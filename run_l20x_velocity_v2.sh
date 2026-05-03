#!/bin/bash
# ==============================================================================
# L20X velocity-v2 queue (2026-05-02 14:50):
#
# After 4×A100 demonstrated v1 latch fires prematurely on noise (their pk03_v00
# v1 = 36.5% vs broken whip-saw 39.5%), we ALL switch to v2 triple-gated latch.
#
# This orchestrator reruns ALL 5 L20X velocity configs under v2:
#   - pk03_v00_K15  (low peak, K=15 slow detect — best v2 candidate first)
#   - pk05_v00      (mid peak, valley=0 — was 27% on v1, expected better on v2)
#   - pk07_v00      (high peak, valley=0 — first time finishing high peak v2)
#   - pk05          (mid peak, valley=0.05 — direct v1/v2 BC residual comparison)
#   - pk03_aggr     (low peak, K=5 vt=0.005 — most affected by v1 premature latch)
#
# All 5 use the v2 het_actor.py (commit 07a61b4c) — fresh launcher.py spawn
# reads latest code from disk. Configs unchanged (use new defaults).
#
# Total wall-clock: 5 × 3.5h ≈ 17.5h. Started 14:50 → ends ~08:20 next day.
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
LOG="logs/l20x_velocity_v2.log"
mkdir -p logs

echo "[$(date '+%m-%d %H:%M')] L20X velocity-v2 queue starting" | tee -a "$LOG"

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

# Order rationale:
#   1. pk03_v00_K15 — low peak + slow detect, best v2 candidate per A100 analysis
#   2. pk05_v00 (rerun) — direct comparison to v1's 27% result
#   3. pk07_v00 (rerun) — high peak, first clean v2 high-peak data point
#   4. pk05 — has valley=0.05, isolates BC-residual effect on top of v2 latch
#   5. pk03_aggr — low peak + aggressive K, most affected by v1 premature latch
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk03_v00_K15.yaml" "ws_swC_v_pk03_v00_K15"
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk05_v00.yaml"     "ws_swC_v_pk05_v00"
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk07_v00.yaml"     "ws_swC_v_pk07_v00"
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk05.yaml"         "ws_swC_v_pk05"
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk03_aggr.yaml"    "ws_swC_v_pk03_aggr"

bash start_env_webshop.sh stop 2>&1 | tail -1

echo "" | tee -a "$LOG"
echo "[$(date '+%m-%d %H:%M')] L20X velocity-v2 queue COMPLETE." | tee -a "$LOG"

echo "=== Final WS velocity-v2 leaderboard (L20X) ===" | tee -a "$LOG"
python <<'PYEOF' | tee -a "$LOG"
import json, os
runs = [
    "ws_swC_v_pk03_v00_K15", "ws_swC_v_pk05_v00", "ws_swC_v_pk07_v00",
    "ws_swC_v_pk05", "ws_swC_v_pk03_aggr",
]
results = {}
for n in runs:
    f = f"experiments/webshop/{n}/validation_log/100.jsonl"
    if os.path.exists(f):
        cnt=0; sr=0
        with open(f) as fh:
            for line in fh:
                d=json.loads(line); cnt += 1
                if d.get("score", d.get("reward", 0)) >= 1.0: sr += 1
        if cnt > 0: results[n] = sr/cnt*100
for n,s in sorted(results.items(), key=lambda kv: -kv[1]):
    print(f"  {s:5.1f}%  {n}")
print(f"  Targets: LUFFY 49.5% | DUET v1 53.0%")
PYEOF
