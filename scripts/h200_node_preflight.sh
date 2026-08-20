#!/bin/bash
# =============================================================================
# Per-node preflight for DUET jobs on the NTU GaaS cluster (hpc-gaas-*).
# SOURCE this from a PBS job script; it does not exec anything.
#
# Responsibilities:
#   1. Turn PBS's GPU allocation into the numeric CUDA_VISIBLE_DEVICES vLLM needs.
#   2. Assert we got exactly the GPUs we asked for (never touch a neighbour's).
#   3. Point RAY_TMPDIR at node-local disk when the node has one.
#   4. Check the AgentGym / env_service ports are actually free on this shared node.
#   5. Decide wandb online vs offline from real reachability.
#
# Every check that fails in a way that would corrupt a run returns non-zero, so
# the caller can abort before burning GPU hours.
# =============================================================================

_pf_fail=0
_pf_err() { echo "PREFLIGHT ERROR: $*" >&2; _pf_fail=1; }
_pf_warn() { echo "PREFLIGHT WARN:  $*" >&2; }
_pf_ok()   { echo "PREFLIGHT OK:    $*"; }

# Number of GPUs this job is supposed to have. Callers override via env.
PF_EXPECT_NGPUS="${PF_EXPECT_NGPUS:-2}"

# ---------------------------------------------------------------------------
# 1 + 2. GPU visibility
#
# In BATCH mode this cluster hands out GPUs via cgroup isolation AND sets
# CUDA_VISIBLE_DEVICES to GPU UUIDs (GPU-xxxxxxxx-...). vLLM does int(device_id)
# on that string and dies with ValueError. Because the cgroup already restricts
# the job, nvidia-smi inside the job lists ONLY our GPUs, renumbered 0..N-1 —
# so re-expressing CVD as those indices is both correct and safe.
# (Interactive jobs already get numeric indices, which is why `qsub -I` worked.)
# ---------------------------------------------------------------------------
pf_fix_gpus() {
    local visible n
    visible=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd,)
    n=$(printf '%s' "$visible" | tr ',' '\n' | grep -c . || true)

    if [ -z "$visible" ]; then
        _pf_err "nvidia-smi reported no GPUs"
        return 1
    fi

    # Guard against the case where cgroup isolation is NOT in effect: if the node
    # shows more GPUs than we requested, blindly taking 0..N-1 would run on GPUs
    # allocated to another user's job on this shared node. Refuse instead.
    if [ "$n" -ne "$PF_EXPECT_NGPUS" ]; then
        _pf_err "expected $PF_EXPECT_NGPUS GPUs, nvidia-smi shows $n (indices: $visible)."
        _pf_err "Refusing to guess — cgroup isolation may not be active. Neighbouring jobs share this node."
        return 1
    fi

    export CUDA_VISIBLE_DEVICES="$visible"
    _pf_ok "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ($n GPU(s))"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/                 /'

    # The queue script hardcodes GPUS="0,1"; make sure that matches reality.
    if [ "$PF_EXPECT_NGPUS" = "2" ] && [ "$visible" != "0,1" ]; then
        _pf_err "GPU indices are '$visible', but run_h200_rebuttal_queue.sh assumes '0,1'."
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# 3. Ray temp dir. NFS works (the group's other project runs that way) but node-
#    local disk is much better for Ray's object spill and session files.
# ---------------------------------------------------------------------------
#    Ray's plasma socket lives under RAY_TMPDIR and AF_UNIX caps the full path at
#    107 bytes; Ray itself appends ~61 and callers ~10, so the chosen root must
#    stay short (<=35 chars) or Ray dies at startup.
pf_pick_ray_tmp() {
    local cand best_free=0 best=""
    for cand in "${PBS_JOBFS:-}" /tmp /scratch /raid /local "${TMPDIR:-}"; do
        [ -n "$cand" ] || continue
        [ -d "$cand" ] || continue
        # Only node-local filesystems; skip anything NFS-mounted.
        local fstype
        fstype=$(stat -f -c %T "$cand" 2>/dev/null)
        case "$fstype" in
            nfs*|wekafs|fuse*) continue ;;
        esac
        [ -w "$cand" ] || continue
        local freegb
        freegb=$(df -BG --output=avail "$cand" 2>/dev/null | tail -1 | tr -dc '0-9')
        [ -n "$freegb" ] || continue
        if [ "$freegb" -gt "$best_free" ]; then best_free=$freegb; best=$cand; fi
    done

    if [ -n "$best" ] && [ "$best_free" -ge 100 ]; then
        # Keep it short: "dray_<user>" not "ray_<jobid>" — see the AF_UNIX note above.
        export RAY_TMPDIR="${best}/dray_$(id -un)"
        mkdir -p "$RAY_TMPDIR"
        _pf_ok "RAY_TMPDIR=$RAY_TMPDIR (node-local ${best}, ${best_free}GB free)"
    else
        _pf_warn "no suitable node-local scratch (best='${best:-none}' ${best_free}GB); keeping RAY_TMPDIR=$RAY_TMPDIR"
        mkdir -p "$RAY_TMPDIR"
    fi

    # Hard budget check: Ray appends ~61 bytes, the queue script another ~10.
    local budget=$(( ${#RAY_TMPDIR} + 71 ))
    if [ "$budget" -gt 107 ]; then
        _pf_err "RAY_TMPDIR too long: '$RAY_TMPDIR' (${#RAY_TMPDIR} chars) -> ~${budget}B plasma socket path > 107B AF_UNIX limit"
        return 1
    fi
    _pf_ok "RAY_TMPDIR path budget ok (~${budget}B of 107B)"
    return 0
}

# ---------------------------------------------------------------------------
# 4. Ports. gpu_as nodes are `sharing = default_shared`, so 36001/8081 are
#    node-global and a neighbouring job can legitimately hold them. Also note
#    this host's ephemeral range is 32768-60999, which CONTAINS 36001, so a
#    random vLLM/Ray bind can land there too.
# ---------------------------------------------------------------------------
#    The port list is overridable via PF_PORTS so jobs that own no service port
#    (pure forward passes) or a different stack (sciworld/deepsearch) are not
#    aborted by a neighbour's unrelated listener on 36001/8081. Unset => the
#    historical default; PF_PORTS="" => skip the check entirely.
pf_check_ports() {
    echo "PREFLIGHT:        ephemeral port range: $(cat /proc/sys/net/ipv4/ip_local_port_range 2>/dev/null)"
    local port rc=0
    for port in ${PF_PORTS-36001 8081}; do
        local pids
        pids=$(lsof -ti:"$port" 2>/dev/null)
        if [ -z "$pids" ]; then
            _pf_ok "port $port free"
            continue
        fi
        local pid owner me args
        me=$(id -un)
        for pid in $pids; do
            owner=$(ps -p "$pid" -o user= 2>/dev/null | tr -d ' ')
            args=$(ps -p "$pid" -o args= 2>/dev/null)
            if [ "$owner" != "$me" ]; then
                _pf_err "port $port held by another user ('$owner'): ${args:0:60}"
                _pf_err "  -> this node cannot host our env service. Resubmit to land elsewhere."
                rc=1
            else
                _pf_warn "port $port held by our own stale process $pid: ${args:0:60}"
            fi
        done
    done
    return $rc
}

# ---------------------------------------------------------------------------
# 5. wandb. The login node has internet; compute nodes may not. Decide from a
#    real probe rather than assuming, so runs never block on a dead endpoint.
# ---------------------------------------------------------------------------
pf_set_wandb_mode() {
    if [ -n "${WANDB_MODE:-}" ]; then
        _pf_ok "WANDB_MODE=$WANDB_MODE (preset, not probing)"
        return 0
    fi
    if timeout 20 curl -sS -o /dev/null -w '%{http_code}' https://api.wandb.ai/ >/dev/null 2>&1; then
        export WANDB_MODE=online
        _pf_ok "WANDB_MODE=online (api.wandb.ai reachable)"
    else
        export WANDB_MODE=offline
        export WANDB_DIR="${WANDB_DIR:-${DUET_PROJECT_ROOT}/wandb}"
        mkdir -p "$WANDB_DIR"
        _pf_warn "api.wandb.ai unreachable -> WANDB_MODE=offline, WANDB_DIR=$WANDB_DIR (sync later)"
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Stale single-instance pidfile. The queue guards with logs/.h200_queue.pid, but
# that file lives on shared storage while the PID is node-local: after a requeue
# onto a different node, an unrelated process can hold the same PID and the
# queue would refuse to start. Clear it only when no queue is actually running here.
# ---------------------------------------------------------------------------
pf_clear_stale_pidfile() {
    local pf="${DUET_PROJECT_ROOT}/logs/.h200_queue.pid"
    [ -f "$pf" ] || return 0
    if pgrep -f "run_h200_rebuttal_queue.sh" >/dev/null 2>&1; then
        _pf_err "a rebuttal queue is already running on $(hostname) — refusing to start a second one"
        return 1
    fi
    _pf_warn "removing stale queue pidfile ($(cat "$pf" 2>/dev/null)) — no queue process on $(hostname)"
    rm -f "$pf"
    return 0
}

pf_run_all() {
    echo "==================== PREFLIGHT $(date '+%F %T') ===================="
    echo "PREFLIGHT:        host=$(hostname) job=${PBS_JOBID:-<none>}"
    pf_fix_gpus          || _pf_fail=1
    pf_pick_ray_tmp      || _pf_fail=1
    pf_check_ports       || _pf_fail=1
    pf_set_wandb_mode
    pf_clear_stale_pidfile || _pf_fail=1
    echo "PREFLIGHT:        disk on project volume: $(df -h /projects_vol/gp_wangwy/qisheng | tail -1)"
    echo "=================================================================="
    return $_pf_fail
}
