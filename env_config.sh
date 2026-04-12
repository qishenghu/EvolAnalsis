#!/bin/bash
# ==============================================================================
# Centralized path config for DUET experiments.
# All scripts source this file. To run on a new server, ONLY edit this file.
# ==============================================================================

# --- Conda ---
export CONDA_PATH="${CONDA_PATH:-$(dirname $(dirname $CONDA_EXE))}"
export CONDA_ENV_DUET="${CONDA_ENV_DUET:-duet}"
export CONDA_ENV_WEBSHOP="${CONDA_ENV_WEBSHOP:-agentenv-webshop}"

# --- Project root (auto-detected from this file's location) ---
export DUET_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Temp directories (use a partition with plenty of space) ---
export TMPDIR="${TMPDIR:-/data/home/$(whoami)/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export RAY_TMPDIR="${RAY_TMPDIR:-/data/ray}"

# --- ALFWorld ---
export ALFWORLD_DATA="${ALFWORLD_DATA:-/data/home/$(whoami)/alfworld_data}"
export ALFWORLD_SERVER_URL="${ALFWORLD_SERVER_URL:-http://127.0.0.1:36001}"
export ALFWORLD_BIN="${CONDA_PATH}/envs/${CONDA_ENV_DUET}/bin/alfworld"

# --- WebShop ---
export WEBSHOP_SERVER_URL="${WEBSHOP_SERVER_URL:-http://127.0.0.1:36003}"
export JAVA_HOME="${CONDA_PATH}/envs/${CONDA_ENV_WEBSHOP}/lib/jvm"
export PATH="${JAVA_HOME}/bin:${PATH}"

# --- Create dirs ---
mkdir -p "$TMPDIR" "$RAY_TMPDIR" 2>/dev/null || true
