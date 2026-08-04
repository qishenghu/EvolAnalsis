#!/usr/bin/env bash
# Read-only health/profile verifier for the isolated teacher environment stacks.
# It never starts, stops, signals, or adopts a process.  A successful exit is
# therefore safe to use as the only adoption gate in launch orchestration.

set -euo pipefail

usage() {
    echo "Usage: $0 {webshop|alfworld}" >&2
}

if [ "$#" -ne 1 ]; then
    usage
    exit 64
fi

ENV_NAME="$1"
case "$ENV_NAME" in
    webshop)
        ENGINE_PORT=18013
        WRAPPER_PORT=18093
        EXPECTED_ENGINE_BODY='"ok"'
        EXPECTED_PROFILE_SIZE=6710
        ;;
    alfworld)
        ENGINE_PORT=18011
        WRAPPER_PORT=18091
        EXPECTED_ENGINE_BODY='"This is environment AlfWorld."'
        EXPECTED_PROFILE_SIZE=2420
        ;;
    *)
        usage
        exit 64
        ;;
esac

CURL_BIN="${TEACHER_AUX_CURL_BIN:-curl}"
PYTHON_BIN="${TEACHER_AUX_PYTHON_BIN:-python3}"
ENGINE_URL="http://127.0.0.1:${ENGINE_PORT}"
WRAPPER_URL="http://127.0.0.1:${WRAPPER_PORT}"

engine_body="$($CURL_BIN --fail --silent --show-error \
    --connect-timeout 2 --max-time 10 "$ENGINE_URL/")" || {
    echo "ERROR: $ENV_NAME engine root is unavailable at $ENGINE_URL/" >&2
    exit 1
}
if [ "$engine_body" != "$EXPECTED_ENGINE_BODY" ]; then
    echo "ERROR: unexpected $ENV_NAME engine root response at $ENGINE_URL/" >&2
    exit 1
fi

wrapper_body="$($CURL_BIN --fail --silent --show-error \
    --connect-timeout 2 --max-time 10 "$WRAPPER_URL/healthz")" || {
    echo "ERROR: $ENV_NAME wrapper health is unavailable at $WRAPPER_URL/healthz" >&2
    exit 1
}
if [ "$wrapper_body" != "OK" ]; then
    echo "ERROR: unexpected $ENV_NAME wrapper health response" >&2
    exit 1
fi

profile_file="$(mktemp "${TMPDIR:-/tmp}/teacher-aux-profile.XXXXXX.json")"
trap 'rm -f "$profile_file"' EXIT

$CURL_BIN --fail --silent --show-error \
    --connect-timeout 2 --max-time 120 \
    -H 'Content-Type: application/json' \
    -X POST \
    --data "{\"env_type\":\"${ENV_NAME}\",\"params\":{\"split\":\"train\"}}" \
    "$WRAPPER_URL/get_env_profile" >"$profile_file" || {
        echo "ERROR: $ENV_NAME train profile request failed" >&2
        exit 1
    }

$PYTHON_BIN - "$profile_file" "$ENV_NAME" "$EXPECTED_PROFILE_SIZE" <<'PY'
import json
import pathlib
import sys

profile_path = pathlib.Path(sys.argv[1])
env_name = sys.argv[2]
expected_size = int(sys.argv[3])

try:
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"ERROR: invalid {env_name} profile JSON: {error}") from error

if payload.get("success") is not True:
    raise SystemExit(f"ERROR: {env_name} profile returned success != true")
profile = payload.get("data")
if not isinstance(profile, list):
    raise SystemExit(f"ERROR: {env_name} profile data is not a list")
if len(profile) != expected_size:
    raise SystemExit(
        f"ERROR: {env_name} train profile has {len(profile)} tasks; "
        f"expected {expected_size}"
    )
canonical_ids = [str(item) for item in profile]
if len(set(canonical_ids)) != expected_size:
    raise SystemExit(f"ERROR: {env_name} train profile contains duplicate task IDs")
PY

echo "OK: ${ENV_NAME} auxiliary stack engine=${ENGINE_PORT} wrapper=${WRAPPER_PORT} profile=${EXPECTED_PROFILE_SIZE}"
