#!/bin/bash
# ==============================================================================
# SOTA Hunt 2026-05-03 — 4 settings, gap-mode + token-weighting
#
# Order (most-critical first):
#   [1/4] 3B WS  (~3.5h) — peak=0.2 valley=0.05 γ=0 + DR3 accel  → target ≥49.5%
#   [2/4] 1.5B WS (~3.5h) — peak=0.3 valley=0.10 γ=0             → target ≥36% (SOTA preserve)
#   [3/4] 3B AF  (~10h)  — peak=0.2 valley=0.02 γ=0.95           → target ≥75%
#   [4/4] 1.5B AF (~10h) — peak=0.3 valley=0.05 γ=0.97           → target ≥45%
#
# Total: ~27h sequential. Per-run env restart (memory leak prevention).
# Reports val@100 to analysis_reports/handoff/results_log.md after each run.
# ==============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
mkdir -p logs analysis_reports/handoff

RESULTS_LOG="analysis_reports/handoff/results_log.md"

wait_for_gpu_clean() {
    local _i
    local our_gpus="${CUDA_GPUS:-0,1,2,3}"
    for _i in {1..30}; do
        local used
        used=$(nvidia-smi --id="$our_gpus" --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>30000' | wc -l)
        if [ "$used" = "0" ]; then echo "  GPU clean (cohabitant tolerance 30GB)"; return 0; fi
        echo "  Waiting for GPU $our_gpus ($used busy)... ${_i}/30"; sleep 10
    done
    echo "  WARN: GPU not fully clean after 300s, proceeding anyway"
}

kill_ray_stragglers() {
    local n
    n=$(ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "  Killing $n training Ray actors..."
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
        sleep 5
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
        sleep 3
    fi
}

restart_env() {
    local env=$1
    echo "  Restarting $env env service..."
    bash start_env_alfworld.sh stop 2>/dev/null || true
    bash start_env_webshop.sh stop 2>/dev/null || true
    sleep 8
    bash start_env_${env}.sh
    sleep 12
}

parse_and_log() {
    local name=$1
    local env=$2
    local val_log="experiments/${env}/${name}/validation_log/100.jsonl"
    python - <<PYEOF
import json, os, datetime
val_log = "${val_log}"
name = "${name}"
env = "${env}"
results_log = "${RESULTS_LOG}"
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
if not os.path.exists(val_log):
    line = f"| {ts} | 4xA100 | {name} | — | — | val@100 missing ({val_log}) |"
else:
    n = sr = 0
    rw = 0.0
    with open(val_log) as fh:
        for ln in fh:
            try:
                x = json.loads(ln)
            except Exception:
                continue
            n += 1
            s = x.get("score", x.get("reward", 0.0))
            rw += s
            if s >= 1.0:
                sr += 1
    if n == 0:
        line = f"| {ts} | 4xA100 | {name} | — | — | val@100 empty |"
    else:
        line = f"| {ts} | 4xA100 | {name} | {sr/n*100:.1f}% (n={n}) | {rw/n:.4f} | env={env} GAP+TOKWEIGHT |"
print(line)
with open(results_log, "a") as fh:
    fh.write(line + "\n")
PYEOF
}

run_one() {
    local config=$1
    local name=$2
    local env=$3
    local idx=$4
    local total=$5
    local ray_tmp="${RAY_TMPDIR}/${name}"

    echo ""
    echo "============================================"
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] PREP: $name (env=$env)"
    echo "============================================"
    kill_ray_stragglers
    wait_for_gpu_clean
    restart_env "$env"
    mkdir -p "$ray_tmp"
    rm -rf "$ray_tmp"/session_* 2>/dev/null || true

    if [ ! -f "$config" ]; then
        echo "[$(date '+%m-%d %H:%M')] MISSING config: $config — skipping"
        return
    fi

    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] RUN: $name"
    (
        CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
            python launcher.py --conf "$config" \
            > "logs/${name}.log" 2>&1
    )
    local rc=$?
    if [ "$rc" = "0" ]; then
        echo "[$(date '+%m-%d %H:%M')] [$idx/$total] DONE: $name"
    else
        echo "[$(date '+%m-%d %H:%M')] [$idx/$total] FAILED (rc=$rc): $name — continuing"
    fi
    parse_and_log "$name" "$env"
    kill_ray_stragglers
    sleep 5
}

echo "============================================"
echo " 4xA100 SOTA Hunt 2026-05-03 (gap mode + token weighting)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""
echo "Queue order (4 settings, ~27h sequential):"
echo "  [1/4] 3B WS  (3.5h) → target ≥49.5% (break LUFFY)"
echo "  [2/4] 1.5B WS (3.5h) → target ≥36% (preserve SOTA)"
echo "  [3/4] 3B AF  (10h)  → target ≥75% (preserve SOTA)"
echo "  [4/4] 1.5B AF (10h) → target ≥45% (preserve SOTA)"
echo ""

QUEUE=(
    "config/duet_paper_experiments_configs/sota_hunt_2026_05_03/ws_3b_gap_pk02_v05_tw_dr3fast.yaml|ws_3b_gap_pk02_v05_tw_dr3fast|webshop"
    "config/duet_paper_experiments_configs/sota_hunt_2026_05_03/ws_1_5b_gap_pk03_v10_tw.yaml|ws_1_5b_gap_pk03_v10_tw|webshop"
    "config/duet_paper_experiments_configs/sota_hunt_2026_05_03/af_3b_gap_pk02_v02_g095_tw.yaml|af_3b_gap_pk02_v02_g095_tw|alfworld"
    "config/duet_paper_experiments_configs/sota_hunt_2026_05_03/af_1_5b_gap_pk03_v05_g097_tw.yaml|af_1_5b_gap_pk03_v05_g097_tw|alfworld"
)
TOTAL=${#QUEUE[@]}

idx=0
for entry in "${QUEUE[@]}"; do
    idx=$((idx+1))
    IFS='|' read -r cfg name env <<< "$entry"
    run_one "$cfg" "$name" "$env" "$idx" "$TOTAL"
done

bash start_env_webshop.sh stop 2>/dev/null || true
bash start_env_alfworld.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " SOTA Hunt complete"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""
echo "Results: $RESULTS_LOG"
