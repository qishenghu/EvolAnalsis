#!/bin/bash
# ==============================================================================
# Auto-launch: wait for Phase A orchestrator (PID 3473937) to finish, then launch Phase B.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PHASE_A_PID=3473937
LOG="logs/after_phase_a_then_b.log"

echo "[$(date '+%m-%d %H:%M')] waiting for Phase A orchestrator PID $PHASE_A_PID..." >> "$LOG"

while kill -0 $PHASE_A_PID 2>/dev/null; do
    sleep 300
done

echo "[$(date '+%m-%d %H:%M')] Phase A done. Launching Phase B aggressive sweep..." >> "$LOG"

# Quick analysis of Phase A
python scripts/analyze_ws_sweep.py --phase A >> "$LOG" 2>&1
echo "" >> "$LOG"

sleep 30
nohup bash run_ws_sweep_phase_b.sh > logs/ws_sweep_phase_b.log 2>&1 &
echo "[$(date '+%m-%d %H:%M')] Phase B launched (PID $!)" >> "$LOG"
