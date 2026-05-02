#!/bin/bash
# ==============================================================================
# L20X Phase 2 takeover (2026-05-02 update):
#
# Original plan was 3-seed Phase C of winner. Pivoted: instead, run 3 ADDITIONAL
# velocity-mode variants to maximize SOTA chances within the same time budget.
#
# This watchdog script:
#   1. Waits until ws_swC_v_pk03_aggr (last run of phase 1 orchestrator) is DONE
#   2. Kills the old orchestrator (it'd otherwise enter 3-seed dispatch)
#   3. Runs 3 new variants sequentially (~10.5h):
#        ws_swC_v_pk05_v00     (peak=0.5, valley=0.0, K=10, vt=0.01)
#        ws_swC_v_pk07_v00     (peak=0.7, valley=0.0, K=10, vt=0.01)
#        ws_swC_v_pk03_v00_K15 (peak=0.3, valley=0.0, K=15, vt=0.015)
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
LOG="logs/l20x_extra_variants.log"
mkdir -p logs

echo "[$(date '+%m-%d %H:%M')] watchdog: waiting for ws_swC_v_pk03_aggr to finish..." | tee -a "$LOG"

# Phase 1: wait for pk03_aggr completion marker
SENTINEL="experiments/webshop/ws_swC_v_pk03_aggr/validation_log/100.jsonl"
while [ ! -f "$SENTINEL" ]; do
    sleep 120
done
echo "[$(date '+%m-%d %H:%M')] pk03_aggr DONE — proceeding to takeover." | tee -a "$LOG"

# Phase 2: kill the old orchestrator + any still-running launcher (pk03_aggr should already be cleaned up)
echo "[$(date '+%m-%d %H:%M')] killing old orchestrator (run_l20x_velocity_queue.sh) + lingering launcher..." | tee -a "$LOG"
pgrep -f "run_l20x_velocity_queue.sh" | xargs -r kill -9 2>/dev/null
sleep 3
pgrep -f "launcher.py" | xargs -r kill -9 2>/dev/null
pgrep -f "ray::"       | xargs -r kill -9 2>/dev/null
sleep 5

# Phase 3: env restart cleanup before our first new run
bash start_env_alfworld.sh stop 2>&1 | tail -1
bash start_env_webshop.sh  stop 2>&1 | tail -1
sleep 8

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

# Run the 3 new variants (priority: most likely winner first)
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk05_v00.yaml"     "ws_swC_v_pk05_v00"
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk07_v00.yaml"     "ws_swC_v_pk07_v00"
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk03_v00_K15.yaml" "ws_swC_v_pk03_v00_K15"

bash start_env_webshop.sh stop 2>&1 | tail -1

echo "[$(date '+%m-%d %H:%M')] L20X extra variants COMPLETE." | tee -a "$LOG"

# Summary across all 5 velocity runs
echo "" | tee -a "$LOG"
echo "=== Final velocity-mode WS leaderboard ===" | tee -a "$LOG"
python <<'PYEOF' | tee -a "$LOG"
import json, os
runs = [
    "ws_swC_v_pk05", "ws_swC_v_pk03_aggr", "ws_swC_v_pk03",
    "ws_swC_v_pk05_v00", "ws_swC_v_pk07_v00", "ws_swC_v_pk03_v00_K15",
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
