#!/bin/bash

# Launch script for AlfWorld environment service
# This script starts the env_service with AlfWorld environment support

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
BEYONDAGENT_DIR="$(dirname "$(dirname "$ENV_SERVICE_DIR")")"
AGENTGYM_ROOT="${AGENTGYM_ROOT:-$(cd "$BEYONDAGENT_DIR/EvolAnalsis/AgentGym/agentenv-alfworld" && pwd)}"

export AGENTGYM_ROOT
export RAY_ENV_NAME=alfworld

# Set ALFWORLD_DATA if not already set
if [ -z "$ALFWORLD_DATA" ]; then
    export ALFWORLD_DATA=/data/code/exp/alfworld
fi

echo "AGENTGYM_ROOT: $AGENTGYM_ROOT"
echo "ALFWORLD_DATA: $ALFWORLD_DATA"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Navigate to project root (env_service)
PROJECT_ROOT="$SCRIPT_DIR/../../"
cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$AGENTGYM_ROOT:$PYTHONPATH"

# Print current working directory and PYTHONPATH for debugging
echo "Current working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# Run Python command
exec python -m env_service.env_service --env alfworld --portal 127.0.0.1 --port 8081

