#!/bin/bash

# Launch script for WebShop environment service
# This script starts the env_service with WebShop environment support

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$ENV_SERVICE_DIR")"

# Set environment variables
export RAY_ENV_NAME=webshop

# WebShop server URL (AgentGym's WebShop server)
# Default port: 36003
export WEBSHOP_SERVER_URL="${WEBSHOP_SERVER_URL:-http://127.0.0.1:36003}"

echo "=== WebShop Environment Service ==="
echo "WEBSHOP_SERVER_URL: $WEBSHOP_SERVER_URL"
echo "PROJECT_ROOT: $PROJECT_ROOT"

# Navigate to project root
cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Print current working directory and PYTHONPATH for debugging
echo "Current working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

echo ""
echo "NOTE: Make sure the AgentGym WebShop server is running:"
echo "  cd AgentGym/agentenv-webshop"
echo "  # First time setup:"
echo "  #   conda env create -n agentenv-webshop -f environment.yml"
echo "  #   conda activate agentenv-webshop"
echo "  #   bash ./setup.sh"
echo "  webshop --host 0.0.0.0 --port 36003"
echo ""

# Run Python command
# Port 8083 for webshop
exec python -m env_service.env_service --env webshop --portal 127.0.0.1 --port 8083

