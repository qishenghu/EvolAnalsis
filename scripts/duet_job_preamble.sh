#!/bin/bash
# =============================================================================
# DUET job preamble — the ONE place every PBS job script gets its start-up
# discipline from.  SOURCE it; it never execs anything and never exits your
# shell (functions return non-zero instead, so the caller decides).
#
# WHY THIS FILE EXISTS
# --------------------
# Three separate multi-hour GPU burns in 2026-08 all had the same shape: a
# start-up hazard was found and patched in ONE job script, then the next new
# script was written from a different ancestor and re-hit the same hazard.
# The landmine register lives in docs/infra/LANDMINES.md; this file is the
# executable half of it.  If you write a new run_*.pbs, source this and call
# the functions — do not re-derive the logic.
#
#   L1  GPU index    -> duet_preamble_gpu          (PBS hands out GPU UUIDs;
#                                                   vLLM int()s them and dies)
#   L2  AlfWorld leak-> duet_preamble_tmp +
#                       duet_preamble_alfworld_cleaner
#   L3  resume_mode  -> duet_validate_train_config (rule 1)
#   L4  Ray pkg dirs -> duet_validate_train_config (rule 2)
#   L6  stale pidfile-> duet_preamble_pidfile_heal
#   L8  save_freq    -> duet_validate_train_config (rule 3)
#   credential hygiene-> duet_preamble_wandb
#
# USAGE
#   source "$REPO/env_config.sh"            # must come FIRST
#   source "$REPO/scripts/duet_job_preamble.sh"
#   duet_preamble_gpu 4            || exit 1
#   duet_preamble_wandb require-online || exit 1
#   duet_preamble_tmp
#   ALF_CLEANER_PID=$(duet_preamble_alfworld_cleaner)
#   duet_preamble_pidfile_heal "$REPO/logs/rollout_servers_8201.pids" || exit 1
#   duet_validate_train_config "$CONF" --lane-mns 16 --lane-gmu 0.35 || exit 1
#
# CONTRACT
#   * every function is idempotent — calling it twice is a no-op the second
#     time (or re-does harmless work), never an error;
#   * every function is callable on its own — no hidden ordering except that
#     env_config.sh must be sourced first, and duet_preamble_tmp must run
#     before duet_preamble_alfworld_cleaner (it defines ALFWORLD_TMPDIR);
#   * every failure prints a line starting with "PREAMBLE ERROR:" naming the
#     landmine it is protecting against, then returns non-zero.
# =============================================================================

# Resolve the repo root from this file's own location so the library works no
# matter what the caller's cwd is.
DUET_PREAMBLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUET_REPO_ROOT="${DUET_REPO_ROOT:-$(cd "$DUET_PREAMBLE_DIR/.." && pwd)}"
export DUET_REPO_ROOT

_duet_err()  { echo "PREAMBLE ERROR: $*" >&2; }
_duet_warn() { echo "PREAMBLE WARN:  $*" >&2; }
_duet_ok()   { echo "PREAMBLE OK:    $*"; }

# Require env_config.sh to have run (it exports DUET_PROJECT_ROOT / RAY_TMPDIR,
# which h200_node_preflight.sh reads).
_duet_require_env_config() {
    if [ -z "${DUET_PROJECT_ROOT:-}" ]; then
        _duet_err "env_config.sh has not been sourced (DUET_PROJECT_ROOT unset)."
        _duet_err "  fix: source \"\$REPO/env_config.sh\" BEFORE scripts/duet_job_preamble.sh"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# L1 — GPU visibility.  PBS batch jobs get CUDA_VISIBLE_DEVICES=GPU-<uuid>,...
# and vLLM does int() on each entry -> ValueError -> the server never comes up
# and the whole job burns its walltime doing nothing.  pf_run_all rewrites CVD
# to the cgroup-local numeric indices and refuses to guess when the cgroup is
# not actually isolating us.
#
#   duet_preamble_gpu [expected_ngpus]
#
# Jobs that own NO service port (pure forward passes, or stacks on other
# ports) should set DUET_PREAMBLE_PORTS="" first so the 36001/8081 occupancy
# check — which is about the AlfWorld stack, not about GPUs — cannot abort
# them for a neighbour's unrelated listener.
# ---------------------------------------------------------------------------
duet_preamble_gpu() {
    _duet_require_env_config || return 1

    local want="${1:-${PF_EXPECT_NGPUS:-2}}"
    case "$want" in
        ''|*[!0-9]*) _duet_err "duet_preamble_gpu: expected_ngpus must be an integer, got '$want'"; return 1 ;;
    esac
    PF_EXPECT_NGPUS="$want"

    # Sourcing is idempotent: the file only defines functions and resets the
    # accumulator.  Re-source every call so a second invocation starts clean.
    # shellcheck disable=SC1091
    source "${DUET_REPO_ROOT}/scripts/h200_node_preflight.sh" || {
        _duet_err "cannot source scripts/h200_node_preflight.sh"
        return 1
    }
    # h200_node_preflight.sh initialises _pf_fail at source time and pf_run_all
    # only ever ORs into it; belt and braces for repeat calls.
    _pf_fail=0

    # PF_PORTS is honoured by pf_check_ports (default "36001 8081").
    if [ "${DUET_PREAMBLE_PORTS+set}" = "set" ]; then
        PF_PORTS="$DUET_PREAMBLE_PORTS"
        export PF_PORTS
    fi

    if ! pf_run_all; then
        _duet_err "node preflight failed (L1 GPU indices / ray tmp / ports / pidfile)."
        _duet_err "  expected ${want} GPU(s); see the PREFLIGHT ERROR lines above."
        return 1
    fi
    _duet_ok "GPU lane ready: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} (expected ${want})"
    return 0
}

# ---------------------------------------------------------------------------
# L2 — node-local scratch.  Two independent reasons:
#   * AlfWorld's planner copies libdownward.so into a fresh temp dir on every
#     env reset.  Keeping that traffic off the PBS job dir and off NFS makes
#     the sweeper below cheap and keeps the inode churn node-local.
#   * Triton / vLLM compile caches on shared storage serialise every worker.
#
# NOTE this only reclaims DISK.  The process-level mmap exhaustion that
# actually kills the AlfWorld server is NOT a disk problem — see
# docs/infra/LANDMINES.md L2.  Moving TMPDIR was measured not to help.
#
#   duet_preamble_tmp [--with-tmpdir]
#
# --with-tmpdir additionally repoints the job's own TMPDIR/TEMP/TMP at node
# local disk.  It is opt-in because most job scripts deliberately hand a
# separate TMPDIR only to the AlfWorld child process.
# ---------------------------------------------------------------------------
duet_preamble_tmp() {
    local with_tmpdir=0 arg
    for arg in "$@"; do
        case "$arg" in
            --with-tmpdir) with_tmpdir=1 ;;
            *) _duet_err "duet_preamble_tmp: unknown option '$arg'"; return 1 ;;
        esac
    done

    local me; me="$(id -un)"

    export ALFWORLD_TMPDIR="${ALFWORLD_TMPDIR:-/tmp/duet_alfworld_tmp_${me}}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${me}}"
    export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache_${me}}"
    mkdir -p "$ALFWORLD_TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" || {
        _duet_err "cannot create node-local cache dirs under /tmp (disk full? permissions?)"
        return 1
    }

    local tmpdir_note=""
    if [ "$with_tmpdir" = "1" ]; then
        export TMPDIR="/tmp/duet_tmp_${me}"
        export TEMP="$TMPDIR"
        export TMP="$TMPDIR"
        mkdir -p "$TMPDIR" || { _duet_err "cannot create TMPDIR=$TMPDIR"; return 1; }
        tmpdir_note=" TMPDIR=$TMPDIR"
    fi

    _duet_ok "node-local scratch: ALFWORLD_TMPDIR=$ALFWORLD_TMPDIR TRITON_CACHE_DIR=$TRITON_CACHE_DIR VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT${tmpdir_note}"
    return 0
}

# ---------------------------------------------------------------------------
# L2 — background sweeper for the libdownward temp-dir litter.  Deletes only
# directories untouched for 30+ minutes, so a live episode is never robbed.
#
#   ALF_CLEANER_PID=$(duet_preamble_alfworld_cleaner)
#   trap 'kill "$ALF_CLEANER_PID" 2>/dev/null' EXIT
#
# Prints the pid on stdout (and nothing else) so it can be captured; all
# diagnostics go to stderr.  Idempotent: if this shell already started one and
# it is still alive, the existing pid is returned instead of starting a second.
# ---------------------------------------------------------------------------
duet_preamble_alfworld_cleaner() {
    if [ -z "${ALFWORLD_TMPDIR:-}" ]; then
        _duet_err "duet_preamble_alfworld_cleaner: ALFWORLD_TMPDIR unset — call duet_preamble_tmp first"
        return 1
    fi
    mkdir -p "$ALFWORLD_TMPDIR" 2>/dev/null

    if [ -n "${DUET_ALF_CLEANER_PID:-}" ] && kill -0 "$DUET_ALF_CLEANER_PID" 2>/dev/null; then
        _duet_warn "alfworld tmp cleaner already running (pid=$DUET_ALF_CLEANER_PID) — reusing"
        echo "$DUET_ALF_CLEANER_PID"
        return 0
    fi

    # stdout/stderr MUST be closed off: this function is called through $(...),
    # and a background child that inherits the command-substitution pipe keeps
    # its write end open forever — the caller would hang instead of getting the pid.
    ( while true; do
        find "$ALFWORLD_TMPDIR" -maxdepth 1 -name 'tmp*' -type d -mmin +30 \
            -exec rm -rf {} + 2>/dev/null
        sleep 600
      done ) >/dev/null 2>&1 &
    DUET_ALF_CLEANER_PID=$!
    _duet_ok "alfworld tmp cleaner pid=$DUET_ALF_CLEANER_PID dir=$ALFWORLD_TMPDIR" >&2
    echo "$DUET_ALF_CLEANER_PID"
    return 0
}

# ---------------------------------------------------------------------------
# L6 — stale rollout pidfile.  The pidfile lives on shared storage but the PIDs
# in it are node-local, so a qdel'd predecessor (or a requeue onto a different
# node) leaves a file whose PIDs now belong to strangers.  Refusing to start on
# that file cost a whole queue slot once.
#
#   duet_preamble_pidfile_heal <pidfile>
#
# Removes the file only when NO listed pid is a live vLLM process on THIS node.
# Returns 1 (do not proceed) when a genuinely live lane is found — that has to
# be stopped by hand, never auto-killed.
# ---------------------------------------------------------------------------
duet_preamble_pidfile_heal() {
    local pidfile="${1:-}"
    if [ -z "$pidfile" ]; then
        _duet_err "duet_preamble_pidfile_heal: needs a pidfile path"
        return 1
    fi
    [ -f "$pidfile" ] || { _duet_ok "no stale pidfile at $pidfile"; return 0; }

    local live=0 _port _pid
    while IFS=':' read -r _port _pid; do
        [ -n "${_pid:-}" ] || continue
        if [ -e "/proc/${_pid}" ] && tr '\0' ' ' < "/proc/${_pid}/cmdline" 2>/dev/null | grep -q vllm; then
            live=1
        fi
    done < "$pidfile"

    if [ "$live" = "0" ]; then
        rm -f "$pidfile"
        _duet_ok "removed stale rollout pidfile from a previous job: $pidfile"
        return 0
    fi
    _duet_err "rollout pidfile $pidfile lists live vLLM PIDs on $(hostname)."
    _duet_err "  a real rollout lane is up; stop it by hand (start_rollout_servers.sh stop) — never auto-kill."
    return 1
}

# ---------------------------------------------------------------------------
# Credential hygiene.  Training refuses env-var API keys so a key can never
# leak into Ray task arguments or a GPU worker's /proc/<pid>/environ; auth goes
# through a mode-0600 ~/.netrc.  Never put a key in a tracked file.
#
#   duet_preamble_wandb require-online   # trainers: abort unless W&B is online
#   duet_preamble_wandb disabled         # pure inference/collection jobs
#   duet_preamble_wandb                  # scrub keys, leave WANDB_MODE as-is
#
# require-online expects duet_preamble_gpu to have run already: pf_set_wandb_mode
# probes api.wandb.ai and sets WANDB_MODE=online/offline from the real result.
# ---------------------------------------------------------------------------
duet_preamble_wandb() {
    local mode="${1:-scrub}"

    # Always scrub, in every mode: plaintext keys must not survive into any
    # child process regardless of what this job does with W&B.
    unset WANDB_API_KEY WANDB_IDENTITY_TOKEN_FILE WANDB_CREDENTIALS_FILE WANDB_DISABLED

    case "$mode" in
        scrub)
            _duet_ok "W&B credentials scrubbed (WANDB_MODE=${WANDB_MODE:-<unset>})"
            ;;
        require-online)
            if [ "${WANDB_MODE:-}" != "online" ]; then
                _duet_err "WANDB_MODE='${WANDB_MODE:-}' — the launcher mandates online W&B."
                _duet_err "  this node cannot reach api.wandb.ai, or duet_preamble_gpu did not run first."
                return 1
            fi
            _duet_ok "W&B online, credentials scrubbed (netrc auth)"
            ;;
        disabled)
            export WANDB_MODE=disabled
            _duet_ok "W&B disabled for this job, credentials scrubbed"
            ;;
        *)
            _duet_err "duet_preamble_wandb: unknown mode '$mode' (want: scrub|require-online|disabled)"
            return 1
            ;;
    esac
    return 0
}

# ---------------------------------------------------------------------------
# L3/L4/L8 — training-config health check.  Thin wrapper around
# scripts/validate_job_config.py so job scripts do not each carry their own
# inline python for data-path and lane checks.
#
#   duet_validate_train_config <yaml> [--lane-mns N] [--lane-gmu F]
#                                     [--allow-disable] [--allow-sparse-ckpt]
#
# Uses whatever `python` is active; falls back to python3.  The validator only
# needs pyyaml.
# ---------------------------------------------------------------------------
duet_validate_train_config() {
    local conf="${1:-}"
    if [ -z "$conf" ]; then
        _duet_err "duet_validate_train_config: needs a yaml path"
        return 1
    fi
    shift

    local py
    py="$(command -v python || command -v python3)"
    if [ -z "$py" ]; then
        _duet_err "duet_validate_train_config: no python on PATH"
        return 1
    fi

    if ! "$py" "${DUET_REPO_ROOT}/scripts/validate_job_config.py" "$conf" "$@"; then
        _duet_err "config validation failed for $conf — see the CHECK lines above."
        _duet_err "  this is the L3/L4/L8 gate; fix the yaml, do not bypass it."
        return 1
    fi
    return 0
}

# Marker so `--audit-scripts` (and humans) can tell the library was loaded.
DUET_JOB_PREAMBLE_LOADED=1
