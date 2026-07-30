#!/bin/bash
# Wait for swA_02 retry val@100 to land, then kill old orchestrator and
# restart with updated priority (Tier S only).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
LOG="logs/swap_orch.log"

echo "[$(date '+%m-%d %H:%M')] waiting for ws_swA_02 retry val@100..." >> "$LOG"
while [ ! -f experiments/webshop/ws_swA_02_peak02/validation_log/100.jsonl ]; do
    sleep 60
done
echo "[$(date '+%m-%d %H:%M')] ws_swA_02 done. Killing old orchestrator + launcher..." >> "$LOG"

# Kill old orchestrator + its child launcher.py + any lingering ray
kill 2239332 2>/dev/null
sleep 5
pgrep -f "run_ws_sweep_phase_b.sh" | xargs -r kill -9 2>/dev/null
pgrep -f "launcher.py" | xargs -r kill -9 2>/dev/null
pgrep -f "ray::" | xargs -r kill -9 2>/dev/null
sleep 10

# Stop env services for clean restart
bash start_env_webshop.sh stop 2>&1 | tail -1 >> "$LOG"
bash start_env_alfworld.sh stop 2>&1 | tail -1 >> "$LOG"
sleep 8

# Relaunch with new priority
echo "[$(date '+%m-%d %H:%M')] relaunching Phase B (Tier S only, 11 configs)..." >> "$LOG"
nohup bash run_ws_sweep_phase_b.sh > logs/ws_sweep_phase_b_v2.log 2>&1 &
NEW_PID=$!
echo "[$(date '+%m-%d %H:%M')] new Phase B PID: $NEW_PID" >> "$LOG"
