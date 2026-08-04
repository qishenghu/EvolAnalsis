#!/bin/bash
# Watcher: wait for current [2/4] 1.5B WS to finish DONE/FAILED,
# then kill the existing orchestrator and launch the followup queue.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
cd "$SCRIPT_DIR"

CURRENT_LOG="logs/sota_hunt_orchestrator.log"
TARGET_PATTERN="\[2/4\] (DONE|FAILED)"

echo "[$(date '+%H:%M:%S')] watcher started; waiting for [2/4] DONE/FAILED in $CURRENT_LOG"

# Poll until the target pattern appears
while ! grep -E "$TARGET_PATTERN" "$CURRENT_LOG" 2>/dev/null > /dev/null; do
    sleep 30
done

echo "[$(date '+%H:%M:%S')] [2/4] finished — initiating swap"
sleep 5  # give parse_and_log a moment to flush

echo "[$(date '+%H:%M:%S')] killing current orchestrator + any stale launcher.py / ray"
pkill -9 -f "run_sota_hunt_2026_05_03.sh" 2>/dev/null || true
sleep 3
pkill -9 -f "launcher.py.*sota_hunt_2026" 2>/dev/null || true
pkill -9 -f "main_ppo" 2>/dev/null || true
pkill -9 -f "ray::WorkerDict\|ray::TaskRunner\|ray::AsyncvLLMServer\|ray::WorkerGroupRegisterCenter" 2>/dev/null || true
sleep 5

echo "[$(date '+%H:%M:%S')] launching followup orchestrator"
nohup bash run_sota_hunt_followup.sh > logs/sota_hunt_followup_orchestrator.log 2>&1 &
NEW_PID=$!
echo "[$(date '+%H:%M:%S')] swap complete; new orchestrator PID=$NEW_PID"
echo "[$(date '+%H:%M:%S')] queue: A_revised → B_revised (best-of-k) → C_revised → 3B AF → 1.5B AF"
