#!/bin/bash
# Watchdog for AgentGym ALFWorld server
# Monitors RSS memory and auto-restarts when it exceeds threshold.
#
# Usage: nohup bash watchdog_agentgym.sh &
#   To stop: kill $(cat /tmp/watchdog_alfworld.pid)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

AGENTGYM_PORT=36001
RSS_THRESHOLD_MB=80000   # Safety net: ~2.5MB/cycle leak in textworld C internals
CHECK_INTERVAL=60

WATCHDOG_LOG="$DUET_PROJECT_ROOT/logs/watchdog_alfworld.log"
mkdir -p "$(dirname $WATCHDOG_LOG)"

echo $$ > /tmp/watchdog_alfworld.pid

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [watchdog] $1" | tee -a "$WATCHDOG_LOG"
}

restart_agentgym() {
    log "Restarting AgentGym server..."
    local pids=$(lsof -ti:$AGENTGYM_PORT 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "$pids" | xargs kill 2>/dev/null
        sleep 2
    fi

    nohup $ALFWORLD_BIN --host 127.0.0.1 --port $AGENTGYM_PORT \
        >> "$DUET_PROJECT_ROOT/logs/alfworld_agentgym.log" 2>&1 &
    disown

    for i in $(seq 1 30); do
        if curl -s http://127.0.0.1:$AGENTGYM_PORT/ 2>/dev/null | grep -q "AlfWorld"; then
            log "AgentGym restarted (PID: $(lsof -ti:$AGENTGYM_PORT | tail -1))"
            return 0
        fi
        sleep 1
    done
    log "ERROR: AgentGym failed to restart!"
    return 1
}

log "Watchdog started. RSS threshold: ${RSS_THRESHOLD_MB}MB, check interval: ${CHECK_INTERVAL}s"

while true; do
    PID=$(lsof -ti:$AGENTGYM_PORT 2>/dev/null | tail -1)

    if [ -z "$PID" ]; then
        log "AgentGym not running! Restarting..."
        restart_agentgym
        sleep $CHECK_INTERVAL
        continue
    fi

    RSS_KB=$(ps -p $PID -o rss --no-headers 2>/dev/null | tr -d ' ')
    if [ -z "$RSS_KB" ]; then
        sleep $CHECK_INTERVAL
        continue
    fi

    RSS_MB=$((RSS_KB / 1024))
    if [ $RSS_MB -gt $RSS_THRESHOLD_MB ]; then
        log "RSS=${RSS_MB}MB exceeds threshold ${RSS_THRESHOLD_MB}MB. Restarting..."
        restart_agentgym
    fi

    sleep $CHECK_INTERVAL
done
