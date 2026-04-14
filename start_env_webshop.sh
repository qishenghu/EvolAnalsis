#!/bin/bash
# Start WebShop environment stack (AgentGym server + env_service wrapper)
# Usage: bash start_env_webshop.sh
#   To stop: bash start_env_webshop.sh stop

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

AGENTGYM_PORT=36003
ENVSERVICE_PORT=8083
PYTHON_DUET="${CONDA_PATH}/envs/${CONDA_ENV_DUET}/bin/python"
WEBSHOP_BIN="${CONDA_PATH}/envs/${CONDA_ENV_WEBSHOP}/bin/webshop"
export RAY_TMPDIR="${RAY_TMPDIR}/envsvc"
mkdir -p "$RAY_TMPDIR"

LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

kill_port() {
    local port=$1
    local pids=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Killing processes on port $port: $pids"
        echo "$pids" | xargs kill 2>/dev/null || true
        sleep 1
    fi
}

if [ "${1:-}" = "stop" ]; then
    echo "Stopping WebShop services..."
    kill_port $AGENTGYM_PORT
    kill_port $ENVSERVICE_PORT
    echo "Done."
    exit 0
fi

echo "=== Starting WebShop Environment Stack ==="

kill_port $AGENTGYM_PORT
kill_port $ENVSERVICE_PORT

echo "[1/2] Starting AgentGym WebShop server on port $AGENTGYM_PORT..."
nohup $WEBSHOP_BIN --host 127.0.0.1 --port $AGENTGYM_PORT \
    > "$LOGDIR/webshop_agentgym.log" 2>&1 &
disown

for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:$AGENTGYM_PORT/ 2>/dev/null | grep -q "ok"; then
        echo "  AgentGym server ready (PID: $(lsof -ti:$AGENTGYM_PORT | tail -1))"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "  ERROR: AgentGym server timed out. Check $LOGDIR/webshop_agentgym.log"
        exit 1
    fi
    sleep 1
done

echo "[2/2] Starting env_service wrapper on port $ENVSERVICE_PORT..."
rm -rf "${RAY_TMPDIR}/session_"* 2>/dev/null || true

PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH" \
WEBSHOP_SERVER_URL=http://127.0.0.1:$AGENTGYM_PORT \
nohup $PYTHON_DUET -m env_service.env_service --env webshop --portal 127.0.0.1 --port $ENVSERVICE_PORT \
    > "$LOGDIR/webshop_envservice.log" 2>&1 &
disown

for i in $(seq 1 30); do
    if lsof -ti:$ENVSERVICE_PORT >/dev/null 2>&1; then
        echo "  env_service ready (PID: $(lsof -ti:$ENVSERVICE_PORT | head -1))"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "  ERROR: env_service failed to start. Check $LOGDIR/webshop_envservice.log"
        kill_port $AGENTGYM_PORT
        exit 1
    fi
    sleep 1
done

echo ""
echo "=== WebShop Environment Stack Running ==="
echo "  AgentGym server: http://127.0.0.1:$AGENTGYM_PORT"
echo "  env_service:     http://127.0.0.1:$ENVSERVICE_PORT"
echo "  Logs:            $LOGDIR/webshop_*.log"
echo "  To stop: bash start_env_webshop.sh stop"
