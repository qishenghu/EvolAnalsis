#!/usr/bin/env bash
# Isolated WebShop stack for teacher collection.
#
# Fixed endpoints:
#   AgentGym engine: 127.0.0.1:18013
#   env_service:     127.0.0.1:18093
#
# Existing services are adopted only after process-identity, root-health,
# wrapper-health, and full 6710-task profile checks.  This script never kills by
# port and never removes a Ray session directory.  `stop` signals only process
# groups whose PID/start-time identity was recorded by this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env_config.sh
source "$SCRIPT_DIR/env_config.sh"

AGENTGYM_PORT=18013
ENVSERVICE_PORT=18093
AGENTGYM_HOST=127.0.0.1
PYTHON_DUET="${CONDA_PATH}/envs/${CONDA_ENV_DUET}/bin/python"
WEBSHOP_BIN="${CONDA_PATH}/envs/${CONDA_ENV_WEBSHOP}/bin/webshop"
VERIFY_SCRIPT="$SCRIPT_DIR/scripts/verify_teacher_aux_env.sh"
HANDOFF_EXEC="$SCRIPT_DIR/scripts/teacher_aux_setsid_exec.sh"

# Keep this Ray cluster entirely separate from training and the ALFWorld aux
# cluster.  The env_service resource options are opt-in, so main launchers that
# do not set them retain their historical defaults.
export RAY_TMPDIR=/data/ray/envwsaux
export CUDA_VISIBLE_DEVICES=""
unset RAY_ADDRESS
export ENV_SERVICE_RAY_NUM_CPUS=8
export ENV_SERVICE_RAY_OBJECT_STORE_MEMORY=2147483648
export ENV_SERVICE_RAY_INCLUDE_DASHBOARD=false
export WEBSHOP_SERVER_URL="http://${AGENTGYM_HOST}:${AGENTGYM_PORT}"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

LOG_DIR="$SCRIPT_DIR/logs"
STATE_DIR="$LOG_DIR/teacher_webshop_aux_state"
LOCK_FILE="$LOG_DIR/.teacher_webshop_aux.lock"
ENGINE_LOG="$LOG_DIR/webshop_agentgym_aux.log"
WRAPPER_LOG="$LOG_DIR/webshop_envservice_aux.log"

mkdir -p "$LOG_DIR" "$STATE_DIR" "$RAY_TMPDIR"
umask 077

for required_command in curl flock lsof nohup setsid sort; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: $required_command" >&2
        exit 1
    fi
done

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "ERROR: another WebShop aux start/stop operation holds $LOCK_FILE" >&2
    exit 1
fi

state_file() {
    printf '%s/%s.pid\n' "$STATE_DIR" "$1"
}

state_field() {
    local file="$1"
    local field="$2"
    awk -F= -v wanted="$field" '$1 == wanted {print substr($0, length($1) + 2); exit}' "$file"
}

proc_stat_field() {
    local pid="$1"
    local requested="$2"
    local stat_line rest
    [ -r "/proc/$pid/stat" ] || return 1
    IFS= read -r stat_line <"/proc/$pid/stat" || return 1
    # Strip pid/comm (fields 1-2).  The remainder begins at state (field 3),
    # even when the parenthesized process name itself contains spaces.
    rest="${stat_line##*) }"
    set -- $rest
    case "$requested" in
        ppid) printf '%s\n' "${2}" ;;       # proc stat field 4
        pgid) printf '%s\n' "${3}" ;;       # proc stat field 5
        start_ticks) printf '%s\n' "${20}" ;; # proc stat field 22
        *) return 1 ;;
    esac
}

process_references_ray_root() {
    local pid="$1"
    local ray_root="$2"
    local argument
    [ -r "/proc/$pid/cmdline" ] || return 1
    while IFS= read -r -d '' argument; do
        if [[ "$argument" == "$ray_root" ||
            "$argument" == *"$ray_root/"* ||
            "$argument" == *"=$ray_root" ]]; then
            return 0
        fi
    done <"/proc/$pid/cmdline"
    return 1
}

find_dedicated_ray_root_processes() {
    local current_pid parent_pid cmdline_file pid
    local -A checker_ancestry=()

    # This function scans /proc directly.  Unlike `ps | awk -v marker=...`, it
    # never places the Ray-root marker in a helper process's argv.  Exclude both
    # the current Bash execution PID and its ancestors in case the operator's
    # invoking shell command itself contains the dedicated path.
    current_pid="${BASHPID:-$$}"
    checker_ancestry[$$]=1
    while [[ "$current_pid" =~ ^[0-9]+$ ]] && [ "$current_pid" -gt 1 ]; do
        checker_ancestry[$current_pid]=1
        parent_pid="$(proc_stat_field "$current_pid" ppid)" || break
        [ "$parent_pid" != "$current_pid" ] || break
        current_pid="$parent_pid"
    done

    for cmdline_file in /proc/[0-9]*/cmdline; do
        [ -r "$cmdline_file" ] || continue
        pid="${cmdline_file#/proc/}"
        pid="${pid%/cmdline}"
        if [ -n "${checker_ancestry[$pid]:-}" ]; then
            continue
        fi
        if process_references_ray_root "$pid" "$RAY_TMPDIR"; then
            printf '%s\n' "$pid"
        fi
    done
    return 0
}

process_has_exact_arg() {
    local pid="$1"
    local expected="$2"
    local argument
    [ -r "/proc/$pid/cmdline" ] || return 1
    while IFS= read -r -d '' argument; do
        if [ "$argument" = "$expected" ]; then
            return 0
        fi
    done <"/proc/$pid/cmdline"
    return 1
}

process_has_arg_basename() {
    local pid="$1"
    local expected="$2"
    local argument
    [ -r "/proc/$pid/cmdline" ] || return 1
    while IFS= read -r -d '' argument; do
        if [ "${argument##*/}" = "$expected" ]; then
            return 0
        fi
    done <"/proc/$pid/cmdline"
    return 1
}

process_has_launch_token() {
    local pid="$1"
    local expected_token="$2"
    local entry
    [ -r "/proc/$pid/environ" ] || return 1
    while IFS= read -r -d '' entry; do
        if [ "$entry" = "TEACHER_AUX_LAUNCH_TOKEN=${expected_token}" ]; then
            return 0
        fi
    done <"/proc/$pid/environ"
    return 1
}

expected_process() {
    local component="$1"
    local pid="$2"
    case "$component" in
        engine)
            process_has_arg_basename "$pid" webshop &&
                process_has_exact_arg "$pid" --host &&
                process_has_exact_arg "$pid" "$AGENTGYM_HOST" &&
                process_has_exact_arg "$pid" --port &&
                process_has_exact_arg "$pid" "$AGENTGYM_PORT"
            ;;
        wrapper)
            process_has_exact_arg "$pid" -m &&
                process_has_exact_arg "$pid" env_service.env_service &&
                process_has_exact_arg "$pid" --env &&
                process_has_exact_arg "$pid" webshop &&
                process_has_exact_arg "$pid" --port &&
                process_has_exact_arg "$pid" "$ENVSERVICE_PORT"
            ;;
        *) return 1 ;;
    esac
}

listener_pid() {
    local port="$1"
    local listeners count
    listeners="$(lsof -nP -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u || true)"
    if [ -z "$listeners" ]; then
        return 0
    fi
    count="$(printf '%s\n' "$listeners" | awk 'NF {count += 1} END {print count + 0}')"
    if [ "$count" -ne 1 ]; then
        echo "ERROR: port $port has $count listener processes" >&2
        return 1
    fi
    printf '%s\n' "$listeners"
}

engine_health() {
    local body
    body="$(curl --fail --silent --show-error --connect-timeout 2 --max-time 10 \
        "http://${AGENTGYM_HOST}:${AGENTGYM_PORT}/" 2>/dev/null)" || return 1
    [ "$body" = '"ok"' ]
}

wrapper_health() {
    local body
    body="$(curl --fail --silent --show-error --connect-timeout 2 --max-time 10 \
        "http://${AGENTGYM_HOST}:${ENVSERVICE_PORT}/healthz" 2>/dev/null)" || return 1
    [ "$body" = "OK" ]
}

write_owned_state() {
    local component="$1"
    local pid="$2"
    local port="$3"
    local launch_token="$4"
    local start_ticks pgid file temporary
    start_ticks="$(proc_stat_field "$pid" start_ticks)" || return 1
    pgid="$(proc_stat_field "$pid" pgid)" || return 1
    if [ "$pgid" != "$pid" ]; then
        echo "ERROR: $component PID $pid is not its own process-group leader" >&2
        return 1
    fi
    file="$(state_file "$component")"
    temporary="${file}.tmp.$$"
    {
        printf 'component=%s\n' "$component"
        printf 'pid=%s\n' "$pid"
        printf 'pgid=%s\n' "$pgid"
        printf 'start_ticks=%s\n' "$start_ticks"
        printf 'port=%s\n' "$port"
        printf 'launch_token=%s\n' "$launch_token"
    } >"$temporary"
    mv -f "$temporary" "$file"
}

# Return 0 for a live, identity-verified owned process; 2 for a dead stale
# record; and 1 for a suspicious/reused identity that must never be signalled.
validate_owned_state() {
    local component="$1"
    local file pid pgid start_ticks current_pgid current_start
    file="$(state_file "$component")"
    [ -f "$file" ] || return 2
    pid="$(state_field "$file" pid)"
    pgid="$(state_field "$file" pgid)"
    start_ticks="$(state_field "$file" start_ticks)"
    if [[ ! "$pid" =~ ^[0-9]+$ || ! "$pgid" =~ ^[0-9]+$ || ! "$start_ticks" =~ ^[0-9]+$ ]]; then
        echo "ERROR: malformed ownership record: $file" >&2
        return 1
    fi
    if [ ! -d "/proc/$pid" ]; then
        return 2
    fi
    current_start="$(proc_stat_field "$pid" start_ticks)" || return 1
    current_pgid="$(proc_stat_field "$pid" pgid)" || return 1
    if [ "$current_start" != "$start_ticks" ] || [ "$current_pgid" != "$pgid" ] || [ "$pgid" != "$pid" ]; then
        echo "ERROR: refusing PID $pid: ownership identity no longer matches $file" >&2
        return 1
    fi
    if ! expected_process "$component" "$pid"; then
        echo "ERROR: refusing PID $pid: command line is not the expected $component" >&2
        return 1
    fi
    return 0
}

STARTED_COMPONENTS=()
declare -A STARTED_HANDOFF_FILES=()

stop_verified_component() {
    local component="$1"
    local file pid pgid
    file="$(state_file "$component")"
    pid="$(state_field "$file" pid)"
    pgid="$(state_field "$file" pgid)"
    echo "Stopping owned WebShop aux $component process group $pgid..."
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 20); do
        if ! kill -0 -- "-$pgid" 2>/dev/null; then
            rm -f "$file"
            return 0
        fi
        sleep 1
    done
    echo "Owned $component process group $pgid did not exit after TERM; sending KILL." >&2
    kill -KILL -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 5); do
        if ! kill -0 -- "-$pgid" 2>/dev/null; then
            rm -f "$file"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: owned $component process group $pgid still exists" >&2
    return 1
}

cleanup_started() {
    local index component status handoff_file
    for ((index=${#STARTED_COMPONENTS[@]} - 1; index >= 0; index--)); do
        component="${STARTED_COMPONENTS[$index]}"
        if validate_owned_state "$component"; then
            stop_verified_component "$component" || true
        else
            status=$?
            handoff_file="${STARTED_HANDOFF_FILES[$component]:-}"
            if [ -n "$handoff_file" ] && [ -f "$handoff_file" ]; then
                rollback_handoff_launch "$component" "$handoff_file" || true
                rm -f "$(state_file "$component")"
            elif [ "$status" -ne 2 ]; then
                echo "WARNING: could not identity-verify newly started $component; not signalling it" >&2
            fi
        fi
    done
}

fail_start() {
    echo "ERROR: $*" >&2
    exit 1
}

cleanup_on_abnormal_exit() {
    local status=$?
    trap - EXIT INT TERM
    if [ "$status" -ne 0 ]; then
        cleanup_started
    fi
    exit "$status"
}

# Any unexpected shell error or interruption rolls back only process groups
# launched during this invocation.  A successful invocation clears the array
# before exiting so the durable services remain up.
trap cleanup_on_abnormal_exit EXIT
trap 'exit 130' INT TERM

validate_handoff_identity() {
    local component="$1"
    local handoff_file="$2"
    local recorded_component pid pgid start_ticks token current_pgid current_start
    [ -f "$handoff_file" ] || return 2
    recorded_component="$(state_field "$handoff_file" component)"
    pid="$(state_field "$handoff_file" pid)"
    pgid="$(state_field "$handoff_file" pgid)"
    start_ticks="$(state_field "$handoff_file" start_ticks)"
    token="$(state_field "$handoff_file" launch_token)"
    if [ "$recorded_component" != "$component" ] ||
        [[ ! "$pid" =~ ^[0-9]+$ || ! "$pgid" =~ ^[0-9]+$ || ! "$start_ticks" =~ ^[0-9]+$ ]] ||
        [ -z "$token" ]; then
        echo "ERROR: malformed launch handoff: $handoff_file" >&2
        return 1
    fi
    if [ ! -d "/proc/$pid" ]; then
        return 2
    fi
    current_start="$(proc_stat_field "$pid" start_ticks)" || return 1
    current_pgid="$(proc_stat_field "$pid" pgid)" || return 1
    if [ "$current_start" != "$start_ticks" ] || [ "$current_pgid" != "$pgid" ] || [ "$pgid" != "$pid" ]; then
        echo "ERROR: launch handoff identity no longer matches PID $pid" >&2
        return 1
    fi
    if ! process_has_launch_token "$pid" "$token"; then
        echo "ERROR: launch handoff token no longer matches PID $pid" >&2
        return 1
    fi
    return 0
}

rollback_handoff_launch() {
    local component="$1"
    local handoff_file="$2"
    local pid pgid start_ticks token current_start current_pgid
    if validate_handoff_identity "$component" "$handoff_file"; then
        :
    else
        case "$?" in
            2) rm -f "$handoff_file"; return 0 ;;
            *) echo "WARNING: refusing rollback because handoff identity is invalid" >&2; return 1 ;;
        esac
    fi
    pid="$(state_field "$handoff_file" pid)"
    pgid="$(state_field "$handoff_file" pgid)"
    start_ticks="$(state_field "$handoff_file" start_ticks)"
    token="$(state_field "$handoff_file" launch_token)"
    echo "Rolling back handoff-verified WebShop aux $component process group $pgid..." >&2
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 50); do
        if [ ! -d "/proc/$pid" ]; then
            rm -f "$handoff_file"
            return 0
        fi
        sleep 0.1
    done
    current_start="$(proc_stat_field "$pid" start_ticks)" || return 1
    current_pgid="$(proc_stat_field "$pid" pgid)" || return 1
    if [ "$current_start" != "$start_ticks" ] || [ "$current_pgid" != "$pgid" ] ||
        ! process_has_launch_token "$pid" "$token"; then
        echo "WARNING: refusing KILL because handoff identity changed" >&2
        return 1
    fi
    kill -KILL -- "-$pgid" 2>/dev/null || true
    rm -f "$handoff_file"
}

wait_for_handoff() {
    local component="$1"
    local handoff_file="$2"
    local status
    for _ in $(seq 1 100); do
        if [ -f "$handoff_file" ]; then
            if validate_handoff_identity "$component" "$handoff_file"; then
                return 0
            else
                status=$?
            fi
            if [ "$status" -eq 1 ]; then
                return 1
            fi
        fi
        sleep 0.1
    done
    echo "ERROR: setsid child did not produce a valid ownership handoff" >&2
    return 1
}

discover_launched_listener() {
    local component="$1"
    local port="$2"
    local handoff_file="$3"
    local listener handoff_pid
    handoff_pid="$(state_field "$handoff_file" pid)"
    for _ in $(seq 1 1200); do
        if ! listener="$(listener_pid "$port")"; then
            echo "ERROR: multiple listeners appeared on previously-free port $port" >&2
            return 1
        fi
        if [ -n "$listener" ]; then
            if [ "$listener" != "$handoff_pid" ]; then
                echo "ERROR: listener PID $listener does not match handoff PID $handoff_pid" >&2
                return 1
            fi
            if ! expected_process "$component" "$listener"; then
                echo "ERROR: an unexpected process won the race for port $port" >&2
                return 1
            fi
            if ! validate_handoff_identity "$component" "$handoff_file"; then
                echo "ERROR: listener PID $listener lost its handoff identity" >&2
                return 1
            fi
            printf '%s\n' "$listener"
            return 0
        fi
        if ! validate_handoff_identity "$component" "$handoff_file"; then
            echo "ERROR: handoff-owned $component process exited before listening" >&2
            return 1
        fi
        sleep 0.1
    done
    echo "ERROR: handoff-owned $component timed out before listening on port $port" >&2
    return 1
}

start_component() {
    local component="$1"
    local port="$2"
    local log_file="$3"
    shift 3
    local file preexisting_listener launch_token launcher_pid actual_pid handoff_file
    file="$(state_file "$component")"
    if validate_owned_state "$component"; then
        fail_start "owned $component is alive but is not listening on expected port $port"
    else
        case "$?" in
            1) fail_start "cannot safely replace suspicious ownership record $file" ;;
            2) : ;;
        esac
    fi

    STARTED_COMPONENTS+=("$component")
    if ! preexisting_listener="$(listener_pid "$port")"; then
        fail_start "cannot prove port $port has a unique listener state before launch"
    fi
    if [ -n "$preexisting_listener" ]; then
        fail_start "port $port ceased to be free before $component launch"
    fi

    if [ ! -r /proc/sys/kernel/random/uuid ]; then
        fail_start "kernel UUID source is unavailable for safe launch ownership"
    fi
    IFS= read -r launch_token </proc/sys/kernel/random/uuid
    if [ -z "$launch_token" ]; then
        fail_start "could not create a unique launch ownership token"
    fi
    handoff_file="$STATE_DIR/${component}.launch.${launch_token}"
    if [ -e "$handoff_file" ]; then
        fail_start "unique handoff path unexpectedly exists: $handoff_file"
    fi
    STARTED_HANDOFF_FILES[$component]="$handoff_file"

    echo "Starting isolated WebShop aux $component on port $port..."
    TEACHER_AUX_LAUNCH_TOKEN="$launch_token" \
        nohup setsid "$HANDOFF_EXEC" "$handoff_file" "$launch_token" \
        "$component" "$@" >>"$log_file" 2>&1 </dev/null &
    launcher_pid=$!

    # GNU setsid may fork when its incoming PID is already a process-group
    # leader.  `$!` can therefore be a transient launcher, not the service; it
    # is retained only for diagnostics.  The actual child writes an atomic
    # PID/start-ticks/PGID handoff before exec.
    echo "  transient launcher PID: $launcher_pid (not used as service ownership)"
    if ! wait_for_handoff "$component" "$handoff_file"; then
        fail_start "$component failed to produce a safe setsid handoff; see $log_file"
    fi
    if ! actual_pid="$(discover_launched_listener "$component" "$port" "$handoff_file")"; then
        fail_start "$component failed safe listener discovery; see $log_file"
    fi
    if ! write_owned_state "$component" "$actual_pid" "$port" "$launch_token"; then
        fail_start "could not record actual $component listener PID $actual_pid"
    fi
    rm -f "$handoff_file"
}

wait_for_health() {
    local component="$1"
    local attempts="$2"
    local health_function="$3"
    local file pid
    file="$(state_file "$component")"
    pid="$(state_field "$file" pid)"
    for _ in $(seq 1 "$attempts"); do
        if "$health_function"; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        sleep 1
    done
    return 1
}

stop_owned_stack() {
    local component status
    local -a verified=()
    local -a stale=()
    # Validate every state record before signalling anything, so malformed or
    # reused identities cannot produce a partially destructive stop.
    for component in wrapper engine; do
        if validate_owned_state "$component"; then
            verified+=("$component")
        else
            status=$?
            case "$status" in
                2) stale+=("$component") ;;
                *) echo "ERROR: stop aborted; no process was signalled" >&2; return 1 ;;
            esac
        fi
    done
    for component in "${stale[@]}"; do
        rm -f "$(state_file "$component")"
    done
    if [ "${#verified[@]}" -eq 0 ]; then
        echo "No script-owned WebShop aux services are running; adopted services were left untouched."
        return 0
    fi
    for component in "${verified[@]}"; do
        stop_verified_component "$component" || return 1
    done
}

case "${1:-start}" in
    stop)
        stop_owned_stack
        exit $?
        ;;
    verify|status)
        exec bash "$VERIFY_SCRIPT" webshop
        ;;
    start|"")
        ;;
    *)
        echo "Usage: $0 [start|stop|verify|status]" >&2
        exit 64
        ;;
esac

if [ ! -r "$VERIFY_SCRIPT" ]; then
    fail_start "read-only verifier is unavailable: $VERIFY_SCRIPT"
fi
if [ ! -x "$HANDOFF_EXEC" ]; then
    fail_start "setsid ownership handoff helper is unavailable: $HANDOFF_EXEC"
fi

if ! ENGINE_PID="$(listener_pid "$AGENTGYM_PORT")"; then
    fail_start "cannot resolve a unique listener on port $AGENTGYM_PORT"
fi
if ! WRAPPER_PID="$(listener_pid "$ENVSERVICE_PORT")"; then
    fail_start "cannot resolve a unique listener on port $ENVSERVICE_PORT"
fi

if [ -n "$ENGINE_PID" ]; then
    if ! expected_process engine "$ENGINE_PID"; then
        fail_start "port $AGENTGYM_PORT is occupied by a non-WebShop process (PID $ENGINE_PID)"
    fi
    if ! engine_health; then
        fail_start "expected WebShop process on port $AGENTGYM_PORT failed its exact root-health gate"
    fi
    echo "Adopting healthy existing WebShop engine PID $ENGINE_PID; leaving it running."
fi

if [ -n "$WRAPPER_PID" ]; then
    if [ -z "$ENGINE_PID" ]; then
        fail_start "wrapper port $ENVSERVICE_PORT is occupied while engine port $AGENTGYM_PORT is free"
    fi
    if ! expected_process wrapper "$WRAPPER_PID"; then
        fail_start "port $ENVSERVICE_PORT is occupied by a non-WebShop wrapper (PID $WRAPPER_PID)"
    fi
    if ! wrapper_health; then
        fail_start "expected wrapper on port $ENVSERVICE_PORT failed /healthz"
    fi
    if ! bash "$VERIFY_SCRIPT" webshop; then
        fail_start "existing WebShop stack failed the complete adoption gate"
    fi
    echo "Adopted the complete healthy WebShop aux stack; no process was restarted."
    exit 0
fi

if [ -z "$ENGINE_PID" ]; then
    if [ ! -x "$WEBSHOP_BIN" ]; then
        fail_start "WebShop executable is not available: $WEBSHOP_BIN"
    fi
    start_component engine "$AGENTGYM_PORT" "$ENGINE_LOG" \
        "$WEBSHOP_BIN" --host "$AGENTGYM_HOST" --port "$AGENTGYM_PORT"
    if ! wait_for_health engine 120 engine_health; then
        fail_start "WebShop engine did not become healthy; see $ENGINE_LOG"
    fi
fi

# A free wrapper port with leftover processes under the dedicated Ray root is
# ambiguous.  Do not remove or signal them; require explicit operator review.
ray_root_processes="$(find_dedicated_ray_root_processes)"
if [ -n "$ray_root_processes" ]; then
    fail_start "live processes already reference $RAY_TMPDIR while wrapper port is free"
fi

if [ ! -x "$PYTHON_DUET" ]; then
    fail_start "duet Python is not executable: $PYTHON_DUET"
fi
start_component wrapper "$ENVSERVICE_PORT" "$WRAPPER_LOG" \
    "$PYTHON_DUET" -m env_service.env_service \
    --env webshop --portal "$AGENTGYM_HOST" --port "$ENVSERVICE_PORT"
if ! wait_for_health wrapper 120 wrapper_health; then
    fail_start "WebShop env_service did not become healthy; see $WRAPPER_LOG"
fi

if ! bash "$VERIFY_SCRIPT" webshop; then
    fail_start "new WebShop stack failed engine/wrapper/6710-task profile verification"
fi

STARTED_COMPONENTS=()
echo "WebShop teacher aux stack is ready at engine=$AGENTGYM_PORT wrapper=$ENVSERVICE_PORT."
echo "Only services recorded in $STATE_DIR are eligible for '$0 stop'."
