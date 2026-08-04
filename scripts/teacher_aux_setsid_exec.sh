#!/usr/bin/env bash
# Internal exec shim for start_env_webshop_aux.sh.  After `setsid` has resolved
# any required fork, this process atomically hands its real PID/PGID/start-time
# to the parent and only then execs the requested service command.

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 HANDOFF_FILE TOKEN COMPONENT COMMAND [ARG ...]" >&2
    exit 64
fi

HANDOFF_FILE="$1"
LAUNCH_TOKEN="$2"
COMPONENT="$3"
shift 3

if [ -z "$LAUNCH_TOKEN" ] || [ -z "$COMPONENT" ]; then
    echo "ERROR: launch token and component must be non-empty" >&2
    exit 1
fi

proc_stat_field() {
    local requested="$1"
    local stat_line rest
    IFS= read -r stat_line <"/proc/$$/stat"
    rest="${stat_line##*) }"
    set -- $rest
    case "$requested" in
        pgid) printf '%s\n' "${3}" ;;
        start_ticks) printf '%s\n' "${20}" ;;
        *) return 1 ;;
    esac
}

PGID="$(proc_stat_field pgid)"
START_TICKS="$(proc_stat_field start_ticks)"
if [ "$PGID" != "$$" ]; then
    echo "ERROR: setsid child $$ is not its own process-group leader (PGID $PGID)" >&2
    exit 1
fi

umask 077
HANDOFF_TMP="${HANDOFF_FILE}.tmp.$$"
trap 'rm -f "$HANDOFF_TMP"' EXIT
{
    printf 'component=%s\n' "$COMPONENT"
    printf 'pid=%s\n' "$$"
    printf 'pgid=%s\n' "$PGID"
    printf 'start_ticks=%s\n' "$START_TICKS"
    printf 'launch_token=%s\n' "$LAUNCH_TOKEN"
} >"$HANDOFF_TMP"
mv -f "$HANDOFF_TMP" "$HANDOFF_FILE"
trap - EXIT

export TEACHER_AUX_LAUNCH_TOKEN="$LAUNCH_TOKEN"
exec "$@"
