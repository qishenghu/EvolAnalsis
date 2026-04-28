#!/bin/bash
# ==============================================================================
# Wait for AF orchestrator to finish, then launch WS Sweep Phase A.
# Runs detached so user doesn't need to babysit.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

AF_ORCH_PID=2010041   # run_diagnose_and_af.sh
LOG="logs/after_af_then_ws_sweep.log"

echo "[$(date '+%m-%d %H:%M')] waiting for AF orchestrator PID $AF_ORCH_PID..." >> "$LOG"

while kill -0 $AF_ORCH_PID 2>/dev/null; do
    sleep 300
done

echo "[$(date '+%m-%d %H:%M')] AF orchestrator done. Launching WS Sweep Phase A..." >> "$LOG"

# Sleep a bit to let env services fully shut down
sleep 30

# Launch Phase A
nohup bash run_ws_sweep_phase_a.sh > logs/ws_sweep_phase_a.log 2>&1 &
echo "[$(date '+%m-%d %H:%M')] WS Phase A launched (PID $!)" >> "$LOG"
