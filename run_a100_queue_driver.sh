#!/bin/bash
# ============================================================================
# File-driven experiment queue.
#
# Reads the NEXT experiment from logs/queue_list.txt at the start of every
# iteration and removes it once launched, so the running order can be changed
# at any time by editing that file — no restart, and nothing in flight is ever
# killed. (Restarting a queue to reprioritise costs whatever the current run has
# done: it happened once here and threw away 54 steps of a 150-step run.)
#
# Launch: nohup bash run_a100_queue_driver.sh > logs/a100_driver.log 2>&1 &
# Reorder: edit logs/queue_list.txt (one experiment name per line)
# Stop after the current run: touch logs/A100_DRIVER_STOP
# ============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
mkdir -p logs
PIDFILE=logs/.a100_driver.pid
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: driver already running (PID $(cat "$PIDFILE"))."; exit 1
fi
echo $$ > "$PIDFILE"; trap 'rm -f "$PIDFILE"' EXIT

source "$SCRIPT_DIR/env_config.sh"
eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENV_DUET}"
GPUS="0,1,2,4"
CFG_ROOT="config/duet_paper_experiments_configs/rebuttal_neurips"
RESULTS_LOG="NeurIPS_2026_Latex/data/a100_rebuttal_results.md"
QUEUE_FILE=logs/queue_list.txt
STOP_FILE=logs/A100_DRIVER_STOP

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

parse_and_log() {
    local name=$1
    python - <<PYEOF
import json, os, datetime, glob
name = "${name}"
env = "webshop" if name.startswith("webshop_") else "alfworld"
cands = sorted(glob.glob(f"experiments/{env}/{name}/validation_log/*.jsonl"),
               key=lambda p: int(os.path.basename(p).split('.')[0]), reverse=True)
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
if not cands:
    line = f"| {ts} | 4xA100(0124) | {name} | - | - | - | - | val MISSING |"
else:
    p = cands[0]; n=s=l=0; rw=0.0
    for ln in open(p):
        try: x=json.loads(ln)
        except Exception: continue
        n+=1; sc=float(x.get("score", x.get("reward",0.0))); rw+=sc
        s+= sc>=1.0; l+= sc>=0.9
    step = os.path.basename(p).split('.')[0]
    line = (f"| {ts} | 4xA100(0124) | {name} (val@{step}) | {s/n*100:.1f}% | {l/n*100:.1f}% | "
            f"{rw/n:.4f} | {n} | OK |") if n else f"| {ts} | 4xA100(0124) | {name} | - | - | - | 0 | EMPTY |"
print(line)
open("${RESULTS_LOG}","a").write(line+"\n")
PYEOF
}

echo "[$(date '+%m-%d %H:%M')] driver started; queue file: $QUEUE_FILE"
while true; do
    [ -f "$STOP_FILE" ] && { echo "STOP file set — exiting."; break; }
    name=$(grep -vE '^\s*(#|$)' "$QUEUE_FILE" 2>/dev/null | head -1)
    [ -z "$name" ] && { echo "[$(date '+%m-%d %H:%M')] queue empty — done."; break; }
    # remove this entry before launching, so a crash cannot loop on it
    grep -vxF "$name" "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"

    case "$name" in webshop_*) env=webshop ;; *) env=alfworld ;; esac
    cfg="${CFG_ROOT}/${env}/${name}.yaml"
    [ -f "$cfg" ] || { echo "MISSING config $cfg — skipping"; continue; }
    ray_tmp="${RAY_TMPDIR}/r$(echo "$name" | md5sum | head -c 8)"
    echo "[$(date '+%m-%d %H:%M')] PREP: $name (env=$env)"
    kill_ray_stragglers
    bash start_env_alfworld.sh stop 2>/dev/null || true
    bash start_env_webshop.sh stop 2>/dev/null || true
    sleep 8
    if [ "$env" = "webshop" ]; then bash start_env_webshop.sh; else bash start_env_alfworld.sh; fi
    sleep 12
    mkdir -p "$ray_tmp"; rm -rf "$ray_tmp"/session_* 2>/dev/null || true
    echo "[$(date '+%m-%d %H:%M')] RUN: $name"
    (CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" python launcher.py --conf "$cfg" > "logs/${name}.log" 2>&1)
    echo "[$(date '+%m-%d %H:%M')] rc=$? : $name"
    parse_and_log "$name"
    kill_ray_stragglers
done
bash start_env_alfworld.sh stop 2>/dev/null || true
bash start_env_webshop.sh stop 2>/dev/null || true
echo "[$(date '+%m-%d %H:%M')] DRIVER COMPLETE"
