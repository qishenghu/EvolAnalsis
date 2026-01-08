#!/bin/bash

# Launch script for BabyAI environment service
# This script starts the env_service with BabyAI environment support

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$ENV_SERVICE_DIR")"

# Set environment variables
export RAY_ENV_NAME=babyai

# BabyAI server URL (AgentGym's BabyAI server)
# Default port: 36002 (different from alfworld which uses 36001)
export BABYAI_SERVER_URL="${BABYAI_SERVER_URL:-http://127.0.0.1:36002}"

echo "=== BabyAI Environment Service ==="
echo "BABYAI_SERVER_URL: $BABYAI_SERVER_URL"
echo "PROJECT_ROOT: $PROJECT_ROOT"

# Navigate to project root
cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Print current working directory and PYTHONPATH for debugging
echo "Current working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

echo ""
echo "NOTE: Make sure the AgentGym BabyAI server is running:"
echo "  cd AgentGym/agentenv-babyai"
echo "  babyai --host 0.0.0.0 --port 36002"
echo ""

# Run Python command
# Port 8082 for babyai (different from alfworld's 8081)
exec python -m env_service.env_service --env babyai --portal 127.0.0.1 --port 8082

