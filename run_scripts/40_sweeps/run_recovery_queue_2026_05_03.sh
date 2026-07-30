#!/bin/bash
# ==============================================================================
# Recovery Queue 2026-05-03 — fix failures + add LUFFY verification
#
# What went wrong this round:
#   - [3/6] best-of-k FAILED (ray socket path too long, 110 > 107 byte limit)
#   - [2/6] 1.5B WS gap NO_TW: 4.5% (disaster — gap mode unstable on 1.5B WS)
#   - [4/6] insurance + [5/6] AF killed by my recovery process
#
# This recovery queue:
#   [1/5] ws_3b_luffy_v       (~3.5h) — verify LUFFY 49.5% on our infra
#   [2/5] ws_3b_bok_v10       (~3.5h) — best-of-k retry with SHORT name
#   [3/5] ws_1_5b_swC02_da    (~3.5h) — 1.5B WS using swC_02 disc_acc recipe (preserve SOTA)
#   [4/5] af_3b_gap_pk02_v02_g095_tw  (~10h) — 3B AF gap+TW
#   [5/5] af_1_5b_gap_pk03_v05_g097_tw (~10h) — 1.5B AF gap+TW
#
# Total: ~30.5h. Done ~05:15 May 5.
#
# Safeguard: ray_tmp uses short hash to avoid 107-byte socket path limit.
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
    local _i; local our_gpus="${CUDA_GPUS:-0,1,2,3}"
    for _i in {1..30}; do
        local used
        used=$(nvidia-smi --id="$our_gpus" --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>30000' | wc -l)
        if [ "$used" = "0" ]; then echo "  GPU clean"; return 0; fi
        echo "  Waiting ($used busy)... ${_i}/30"; sleep 10
    done
    echo "  WARN: proceeding"
}

kill_ray_stragglers() {
    local n
    n=$(ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | wc -l)
    if [ "$n" -gt 0 ]; then
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
        sleep 5
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
        sleep 3
    fi
}

restart_env() {
    local env=$1
    bash start_env_alfworld.sh stop 2>/dev/null || true
    bash start_env_webshop.sh stop 2>/dev/null || true
    sleep 8
    bash start_env_${env}.sh
    sleep 12
}

parse_and_log() {
    local name=$1; local env=$2
    local val_log="experiments/${env}/${name}/validation_log/100.jsonl"
    python - <<PYEOF
import json, os, datetime
val_log = "${val_log}"; name = "${name}"; env = "${env}"
results_log = "${RESULTS_LOG}"
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
if not os.path.exists(val_log):
    line = f"| {ts} | 4xA100 | {name} | — | — | val@100 missing |"
else:
    n = sr = 0; rw = 0.0
    with open(val_log) as fh:
        for ln in fh:
            try: x = json.loads(ln)
            except Exception: continue
            n += 1; s = x.get("score", x.get("reward", 0.0)); rw += s
            if s >= 1.0: sr += 1
    line = f"| {ts} | 4xA100 | {name} | {sr/n*100:.1f}% (n={n}) | {rw/n:.4f} | env={env} RECOVERY |" if n else f"| {ts} | 4xA100 | {name} | — | — | val@100 empty |"
print(line)
with open(results_log, "a") as fh: fh.write(line + "\n")
PYEOF
}

run_one() {
    local config=$1; local name=$2; local env=$3; local idx=$4; local total=$5
    # Use short hash for ray_tmp dir to avoid 107-byte socket path limit
    local short_hash=$(echo "$name" | md5sum | head -c 8)
    local ray_tmp="${RAY_TMPDIR}/r${short_hash}"
    echo ""
    echo "============================================"
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] PREP: $name (env=$env)"
    echo "  ray_tmp: $ray_tmp (hash=${short_hash})"
    echo "============================================"
    kill_ray_stragglers; wait_for_gpu_clean; restart_env "$env"
    mkdir -p "$ray_tmp"; rm -rf "$ray_tmp"/session_* 2>/dev/null || true
    if [ ! -f "$config" ]; then
        echo "MISSING config: $config — skipping"; return
    fi
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] RUN: $name"
    (CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" python launcher.py --conf "$config" > "logs/${name}.log" 2>&1)
    local rc=$?
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] $([ $rc = 0 ] && echo DONE || echo FAILED rc=$rc): $name"
    parse_and_log "$name" "$env"
    kill_ray_stragglers; sleep 5
}

echo "============================================"
echo " 4xA100 RECOVERY $(date '+%Y-%m-%d %H:%M:%S')"
echo " Queue: LUFFY-verify → bok retry → 1.5B-WS recovery → 3B AF → 1.5B AF"
echo "============================================"

QUEUE=(
    "config/duet_paper_experiments_configs/sota_hunt_2026_05_03/ws_3b_luffy_v.yaml|ws_3b_luffy_v|webshop"
    "config/duet_paper_experiments_configs/sota_hunt_2026_05_03/ws_3b_bok_v10.yaml|ws_3b_bok_v10|webshop"
    "config/duet_paper_experiments_configs/sota_hunt_2026_05_03/ws_1_5b_swC02_da.yaml|ws_1_5b_swC02_da|webshop"
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
echo " RECOVERY complete $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo "Results: $RESULTS_LOG"
