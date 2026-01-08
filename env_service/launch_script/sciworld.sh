#!/bin/bash

# Launch script for ScienceWorld environment service
# This script starts the env_service with ScienceWorld environment support

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$ENV_SERVICE_DIR")"

# Set environment variables
export RAY_ENV_NAME=sciworld

# ScienceWorld server URL (AgentGym's ScienceWorld server)
# Default port: 36004
export SCIWORLD_SERVER_URL="${SCIWORLD_SERVER_URL:-http://127.0.0.1:36004}"

echo "=== ScienceWorld Environment Service ==="
echo "SCIWORLD_SERVER_URL: $SCIWORLD_SERVER_URL"
echo "PROJECT_ROOT: $PROJECT_ROOT"

# Navigate to project root
cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Print current working directory and PYTHONPATH for debugging
echo "Current working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

echo ""
echo "NOTE: Make sure the AgentGym ScienceWorld server is running:"
echo "  cd AgentGym/agentenv-sciworld"
echo "  # First time setup:"
echo "  #   conda create --name agentenv-sciworld python=3.8"
echo "  #   conda activate agentenv-sciworld"
echo "  #   pip install -e ."
echo "  # Requires Java 1.8+ on your system"
echo "  sciworld --host 0.0.0.0 --port 36004"
echo ""

# Run Python command
# Port 8084 for sciworld
exec python -m env_service.env_service --env sciworld --portal 127.0.0.1 --port 8084

