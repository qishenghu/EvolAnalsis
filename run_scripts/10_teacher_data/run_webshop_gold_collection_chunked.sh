#!/bin/bash
# Chunked WebShop gold collection with service restart between chunks.
#
# This script:
#   1. starts the AgentGym WebShop server
#   2. starts env_service for WebShop
#   3. collects one chunk of tasks
#   4. stops both services to release memory
#   5. repeats until the full train split is covered
#
# Example:
#   bash run_webshop_gold_collection_chunked.sh
#   CHUNK_SIZE=200 INSTRUCTION_MATCH_POLICY=strict bash run_webshop_gold_collection_chunked.sh
# CHUNK_SIZE=200 INSTRUCTION_MATCH_POLICY=strict bash run_webshop_gold_collection_chunked.sh
# START_CHUNK_INDEX=10 bash run_webshop_gold_collection_chunked.sh


set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CHUNK_SIZE="${CHUNK_SIZE:-50}"
WEBSHOP_ENV_NAME="${WEBSHOP_ENV_NAME:-agentenv-webshop}"
COLLECT_ENV_NAME="${COLLECT_ENV_NAME:-agentevolver}"
WEBSHOP_HOST="${WEBSHOP_HOST:-0.0.0.0}"
WEBSHOP_PORT="${WEBSHOP_PORT:-36003}"
ENV_SERVICE_HOST="${ENV_SERVICE_HOST:-127.0.0.1}"
ENV_SERVICE_PORT="${ENV_SERVICE_PORT:-8083}"
ENV_URL="http://${ENV_SERVICE_HOST}:${ENV_SERVICE_PORT}"
INSTRUCTION_MATCH_POLICY="${INSTRUCTION_MATCH_POLICY:-strict}"
MAX_STEPS="${MAX_STEPS:-20}"
TARGET_ROLLOUTS_PER_TASK="${TARGET_ROLLOUTS_PER_TASK:-5}"
TARGET_MULTISEARCH_ROLLOUTS="${TARGET_MULTISEARCH_ROLLOUTS:-4}"
CHUNK_OUTPUT_DIR="${CHUNK_OUTPUT_DIR:-analysis_outputs/webshop_gold_chunks_full}"
MERGED_OUTPUT="${MERGED_OUTPUT:-analysis_outputs/webshop_gold_train_multisearch_full_chunked.jsonl}"
LOG_DIR="${LOG_DIR:-logs/webshop_gold_chunked}"
START_CHUNK_INDEX="${START_CHUNK_INDEX:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

WEBSHOP_PID=""
ENV_SERVICE_PID=""

wait_for_port() {
  local host="$1"
  local port="$2"
  local label="$3"
  local timeout_sec="${4:-120}"
  local deadline=$((SECONDS + timeout_sec))

  while (( SECONDS < deadline )); do
    if "$PYTHON_BIN" - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
raise SystemExit(0)
PY
    then
      echo "[ready] ${label} on ${host}:${port}"
      return 0
    fi
    sleep 1
  done

  echo "[error] Timed out waiting for ${label} on ${host}:${port}" >&2
  return 1
}

cleanup_services() {
  set +e

  if [[ -n "${ENV_SERVICE_PID}" ]]; then
    kill "${ENV_SERVICE_PID}" 2>/dev/null || true
    wait "${ENV_SERVICE_PID}" 2>/dev/null || true
    ENV_SERVICE_PID=""
  fi
  if [[ -n "${WEBSHOP_PID}" ]]; then
    kill "${WEBSHOP_PID}" 2>/dev/null || true
    wait "${WEBSHOP_PID}" 2>/dev/null || true
    WEBSHOP_PID=""
  fi

  pkill -f "env_service.env_service --env webshop --portal ${ENV_SERVICE_HOST} --port ${ENV_SERVICE_PORT}" 2>/dev/null || true
  pkill -f "webshop --host ${WEBSHOP_HOST} --port ${WEBSHOP_PORT}" 2>/dev/null || true
  sleep 2
  set -e
}

trap cleanup_services EXIT INT TERM

start_services() {
  local chunk_tag="$1"
  mkdir -p "$LOG_DIR"

  echo "[start] WebShop server for chunk ${chunk_tag}"
  conda run -n "${WEBSHOP_ENV_NAME}" --no-capture-output \
    webshop --host "${WEBSHOP_HOST}" --port "${WEBSHOP_PORT}" \
    > "${LOG_DIR}/webshop_${chunk_tag}.log" 2>&1 &
  WEBSHOP_PID=$!
  wait_for_port "127.0.0.1" "${WEBSHOP_PORT}" "webshop server"

  echo "[start] env_service for chunk ${chunk_tag}"
  (
    cd "${PROJECT_ROOT}"
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
    export WEBSHOP_SERVER_URL="http://127.0.0.1:${WEBSHOP_PORT}"
    conda run -n "${COLLECT_ENV_NAME}" --no-capture-output \
      python -m env_service.env_service --env webshop --portal "${ENV_SERVICE_HOST}" --port "${ENV_SERVICE_PORT}"
  ) > "${LOG_DIR}/env_service_${chunk_tag}.log" 2>&1 &
  ENV_SERVICE_PID=$!
  wait_for_port "${ENV_SERVICE_HOST}" "${ENV_SERVICE_PORT}" "env_service"
}

count_train_tasks() {
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

path = Path("env_service/environments/webshop/webshop_train.json")
data = json.loads(path.read_text(encoding="utf-8"))
print(len(data))
PY
}

chunk_expected_task_count() {
  local start="$1"
  local end="$2"
  echo $(( end - start ))
}

is_chunk_complete() {
  local chunk_file="$1"
  local expected="$2"

  if [[ ! -f "${chunk_file}" ]]; then
    return 1
  fi

  local completed
  completed=$("$PYTHON_BIN" - "${chunk_file}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
count = 0
for line in path.open("r", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    rec = json.loads(line)
    if rec.get("record_kind") == "task_summary":
        count += 1
print(count)
PY
)

  [[ "${completed}" == "${expected}" ]]
}

merge_chunk_outputs() {
  mkdir -p "$(dirname "${MERGED_OUTPUT}")"
  "$PYTHON_BIN" - "${CHUNK_OUTPUT_DIR}" "${MERGED_OUTPUT}" <<'PY'
from pathlib import Path
import sys

src_dir = Path(sys.argv[1])
dst = Path(sys.argv[2])
files = sorted(src_dir.glob("chunk_*.jsonl"))

with dst.open("w", encoding="utf-8") as wf:
    for path in files:
        wf.write(path.read_text(encoding="utf-8"))

print(f"merged_chunks={len(files)} output={dst}")
PY
}

run_chunk() {
  local start="$1"
  local end="$2"
  local chunk_id="$3"
  local chunk_tag
  chunk_tag=$(printf "%04d_%04d" "${start}" "$((end - 1))")
  local chunk_file="${CHUNK_OUTPUT_DIR}/chunk_${chunk_tag}.jsonl"
  local expected
  expected=$(chunk_expected_task_count "${start}" "${end}")

  if is_chunk_complete "${chunk_file}" "${expected}"; then
    echo "[skip] chunk ${chunk_id} already complete: ${chunk_file}"
    return 0
  fi

  if [[ -f "${chunk_file}" ]]; then
    echo "[reset] removing incomplete chunk file: ${chunk_file}"
    rm -f "${chunk_file}"
  fi

  start_services "${chunk_tag}"
  echo "[collect] chunk=${chunk_id} task_range=[${start}, ${end}) output=${chunk_file}"

  conda run -n "${COLLECT_ENV_NAME}" --no-capture-output \
    python scripts/collect_webshop_gold_trajectories.py \
    --env_url "${ENV_URL}" \
    --split train \
    --task_start "${start}" \
    --task_end "${end}" \
    --output "${chunk_file}" \
    --max_steps "${MAX_STEPS}" \
    --target_rollouts_per_task "${TARGET_ROLLOUTS_PER_TASK}" \
    --target_multisearch_rollouts "${TARGET_MULTISEARCH_ROLLOUTS}" \
    --instruction_match_policy "${INSTRUCTION_MATCH_POLICY}" \
    --abort_on_service_error

  cleanup_services

  if ! is_chunk_complete "${chunk_file}" "${expected}"; then
    echo "[error] chunk ${chunk_id} did not finish cleanly: ${chunk_file}" >&2
    return 1
  fi

  merge_chunk_outputs
}

main() {
  mkdir -p "${CHUNK_OUTPUT_DIR}" "${LOG_DIR}"

  local total_tasks
  total_tasks=$(count_train_tasks)
  local total_chunks=$(( (total_tasks + CHUNK_SIZE - 1) / CHUNK_SIZE ))

  echo "=== WebShop chunked full collection ==="
  echo "total_tasks=${total_tasks}"
  echo "chunk_size=${CHUNK_SIZE}"
  echo "total_chunks=${total_chunks}"
  echo "instruction_match_policy=${INSTRUCTION_MATCH_POLICY}"
  echo "chunk_output_dir=${CHUNK_OUTPUT_DIR}"
  echo "merged_output=${MERGED_OUTPUT}"

  local chunk_id
  for (( chunk_id=START_CHUNK_INDEX; chunk_id<total_chunks; chunk_id++ )); do
    local start=$(( chunk_id * CHUNK_SIZE ))
    local end=$(( start + CHUNK_SIZE ))
    if (( end > total_tasks )); then
      end=${total_tasks}
    fi
    run_chunk "${start}" "${end}" "${chunk_id}"
  done

  echo "[done] all chunks finished."
  merge_chunk_outputs
}

main "$@"
