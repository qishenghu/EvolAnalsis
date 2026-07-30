#!/bin/bash
# ==============================================================================
# SOTA Hunt 2026-05-03 FOLLOW-UP — refined based on agent diagnostic of Run A 38%
#
# Agent verdict: token_weighting=true cut effective BC dose by 15× on 3B WS.
# DR3 acceleration was benign and can stay. Single-knob fix: drop TW.
#
# Order:
#   [1/5] 3B WS A_revised  (~3.5h) — drop TW + valley=0.10 + DR3 fast → target 45-47%
#   [2/5] 3B WS B_revised  (~3.5h) — A_revised + best_of_k gap     → target 44-48%
#   [3/5] 3B WS C_revised  (~3.5h) — TW + boosted peak 2.0 (insurance) → target 41-44%
#   [4/5] 3B AF (gap+TW)   (~10h)  — preserve SOTA 77.5%
#   [5/5] 1.5B AF (gap+TW) (~10h)  — preserve SOTA 47.5%
#
# Total: ~30.5h sequential.
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
        echo "  Waiting for GPU ($used busy)... ${_i}/30"; sleep 10
    done
    echo "  WARN: proceeding anyway"
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
    line = f"| {ts} | 4xA100 | {name} | {sr/n*100:.1f}% (n={n}) | {rw/n:.4f} | env={env} FOLLOWUP |" if n else f"| {ts} | 4xA100 | {name} | — | — | val@100 empty |"
print(line)
with open(results_log, "a") as fh: fh.write(line + "\n")
PYEOF
}

run_one() {
    local config=$1; local name=$2; local env=$3; local idx=$4; local total=$5
    local ray_tmp="${RAY_TMPDIR}/${name}"
    echo ""
    echo "============================================"
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] PREP: $name (env=$env)"
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
echo " 4xA100 SOTA Hunt FOLLOW-UP $(date '+%Y-%m-%d %H:%M:%S')"
echo " Refined: drop TW (agent #1 recommendation)"
echo "============================================"

QUEUE=(
    "config/duet_paper_experiments_configs/sota_hunt_2026_05_03/ws_3b_gap_pk02_v10_NOtw_dr3fast.yaml|ws_3b_gap_pk02_v10_NOtw_dr3fast|webshop"
    "config/duet_paper_experiments_configs/sota_hunt_2026_05_03/ws_1_5b_gap_pk03_v10_NOtw.yaml|ws_1_5b_gap_pk03_v10_NOtw|webshop"
    "config/duet_paper_experiments_configs/sota_hunt_2026_05_03/ws_3b_gap_bok_pk02_v10_NOtw_dr3fast.yaml|ws_3b_gap_bok_pk02_v10_NOtw_dr3fast|webshop"
    "config/duet_paper_experiments_configs/sota_hunt_2026_05_03/ws_3b_gap_pk20_v50_TW_dr3fast.yaml|ws_3b_gap_pk20_v50_TW_dr3fast|webshop"
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
echo " SOTA Hunt FOLLOW-UP complete $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo "Results: $RESULTS_LOG"
