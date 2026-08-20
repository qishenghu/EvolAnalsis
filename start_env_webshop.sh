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

# Preserve the outgoing environment logs instead of truncating them. These files
# used to be overwritten by every queue launch, so a finished run's environment
# could never be audited afterwards — which is exactly the evidence we wanted when
# a result looked wrong. Rotate by timestamp; the queue log gives the run/time map.
ENV_ARCHIVE="$LOGDIR/env_archive"
mkdir -p "$ENV_ARCHIVE"
for _old in "$LOGDIR/webshop_agentgym.log" "$LOGDIR/webshop_envservice.log"; do
    if [ -s "$_old" ]; then
        mv "$_old" "$ENV_ARCHIVE/$(basename "$_old" .log)_$(date +%Y%m%d_%H%M%S).log" 2>/dev/null || true
    fi
done

kill_port() {
    local port=$1
    # -sTCP:LISTEN 只杀监听者(lsof 会把连着该端口的客户端一并列出,
    # 2026-08-08 曾误杀采集驱动 rc=143)。
    local pids=$(lsof -ti:$port -sTCP:LISTEN 2>/dev/null)
    local pid args owner me
    me=$(id -un)
    for pid in $pids; do
        # NTU GaaS: compute nodes are shared, so ports are node-global and
        # another user's job may legitimately hold this one. Only ever touch
        # our own processes.
        owner=$(ps -p "$pid" -o user= 2>/dev/null | tr -d ' ')
        if [ -n "$owner" ] && [ "$owner" != "$me" ]; then
            echo "  REFUSING to kill PID $pid on port $port (owned by '$owner', not '$me')"
            continue
        fi
        args=$(ps -p "$pid" -o args= 2>/dev/null)
        # The ephemeral port range on this host is 32768-60999, which OVERLAPS the
        # service ports below. vLLM/Ray actors bind RANDOM ports in that range, so a
        # blind kill-by-port can take down a running training job. Never kill a
        # process that looks like training infrastructure.
        case "$args" in
            *ray::*|*vllm*|*EngineCore*|*main_ppo*|*launcher.py*|*collect_openrouter*|*collect_student*)
                echo "  REFUSING to kill PID $pid on port $port (training process): ${args:0:80}"
                continue
                ;;
        esac
        echo "Killing process on port $port: $pid"
        kill "$pid" 2>/dev/null || true
    done
    sleep 1
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
# The vendored `webshop` tree is not pip-installed; `web_agent_site` resolves
# via PYTHONPATH. Data/index paths inside it are file-relative, so cwd is free.
export PYTHONPATH="$SCRIPT_DIR/AgentGym/agentenv-webshop/webshop${PYTHONPATH:+:$PYTHONPATH}"
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
