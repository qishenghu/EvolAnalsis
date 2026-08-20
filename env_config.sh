#!/bin/bash
# ==============================================================================
# Centralized path config for DUET experiments.
# All scripts source this file. To run on a new server, ONLY edit this file.
#
# HOST: NTU HPC "GaaS" cluster (hpc-gaas-*). Adapted 2026-08-04 from the
# rebuttal-era EvolAnalsis_old/env_config.sh.
#   - Login node hpc-gaas-hn2 has an L40S; H200s come from PBS jobs on the
#     `gpu_as` queue (project gs_ccds_wangwy), so training/serving runs inside
#     PBS batch jobs.
#   - There is NO conda on PATH. The system install /usr/local/anaconda2025 is
#     read-only; all envs live under ${DUET_SCRATCH}/conda/envs (duet is a
#     conda-pack unpack; vllm2 / agentenv-webshop are prefix-created envs).
#   - /data/... (the A100 server's layout) does not exist here.
# ==============================================================================

# --- Scratch root: the writable project volume on this cluster ---
export DUET_SCRATCH="${DUET_SCRATCH:-/projects_vol/gp_wangwy/qisheng/duet_h200}"

# --- Conda ---
export CONDA_PATH="${CONDA_PATH:-${DUET_SCRATCH}/conda}"
export CONDA_BIN="${CONDA_BIN:-/usr/local/anaconda2025/bin/conda}"
export CONDA_ENV_DUET="${CONDA_ENV_DUET:-duet}"
export CONDA_ENV_WEBSHOP="${CONDA_ENV_WEBSHOP:-agentenv-webshop}"
export CONDA_ENV_ALFWORLD="${CONDA_ENV_ALFWORLD:-duet}"
export CONDA_ENV_VLLM2="${CONDA_ENV_VLLM2:-vllm2}"

# --- Project root (auto-detected from this file's location) ---
export DUET_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Shared models on this cluster ---
export DUET_MODELS_DIR="${DUET_MODELS_DIR:-/projects_vol/gp_wangwy/models}"

# --- Temp directories ---
# PBS sets TMPDIR to node-local scratch inside a job; honour it when present.
export TMPDIR="${TMPDIR:-${DUET_SCRATCH}/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
# AF_UNIX socket paths are capped at 107 bytes; RAY_TMPDIR must stay short.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/dray_$(id -un)}"

# --- ALFWorld ---
export ALFWORLD_DATA="${ALFWORLD_DATA:-/home/qisheng001/DUET_H200/alfworld_data}"
export ALFWORLD_SERVER_URL="${ALFWORLD_SERVER_URL:-http://127.0.0.1:36001}"
export ALFWORLD_BIN="${CONDA_PATH}/envs/${CONDA_ENV_ALFWORLD}/bin/alfworld"

# --- WebShop ---
export WEBSHOP_SERVER_URL="${WEBSHOP_SERVER_URL:-http://127.0.0.1:36003}"
export JAVA_HOME="${CONDA_PATH}/envs/${CONDA_ENV_WEBSHOP}/lib/jvm"
export PATH="${JAVA_HOME}/bin:${PATH}"

# --- Wandb ---
# Use a mode-0600 ~/.netrc (or WANDB_CREDENTIALS_FILE), never this tracked
# file or a plaintext WANDB_API_KEY/.env entry. Training rejects env API keys
# so credentials cannot leak into Ray task arguments or GPU worker processes.

# --- Activation helper ---
# The 'duet' env is a conda-pack unpack (no base env), so `conda activate duet`
# cannot resolve it. Sourcing the env's own bin/activate is the portable way.
duet_activate() {
    local prefix="${CONDA_PATH}/envs/${CONDA_ENV_DUET}"
    if [ ! -x "${prefix}/bin/python" ]; then
        echo "ERROR: duet env not found at ${prefix} (did the conda-pack unpack run?)" >&2
        return 1
    fi

    # conda's bin/activate references vars that may be unset (_CE_M, PS1, ...).
    # Under `set -u` that is FATAL for the calling shell; drop -u while sourcing.
    local _had_u=0
    case "$-" in *u*) _had_u=1; set +u ;; esac

    # shellcheck disable=SC1091
    source "${prefix}/bin/activate" 2>/dev/null || {
        export PATH="${prefix}/bin:${PATH}"
        export CONDA_PREFIX="${prefix}"
    }

    [ "$_had_u" = "1" ] && set -u

    case ":${PATH}:" in
        *":${prefix}/bin:"*) ;;
        *) export PATH="${prefix}/bin:${PATH}" ;;
    esac
    return 0
}

# --- Create dirs ---
mkdir -p "$TMPDIR" "$RAY_TMPDIR" 2>/dev/null || true
