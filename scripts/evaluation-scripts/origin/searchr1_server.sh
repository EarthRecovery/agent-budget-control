#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ragenv2

PROJECT_ROOT=${PROJECT_ROOT:-"$HOME/agent-budget-control"}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:$PWD/verl"

resolve_path() {
  local path="$1"
  if [[ "$path" == "~/"* ]]; then
    path="$HOME/${path#~/}"
  elif [[ "$path" == "~" ]]; then
    path="$HOME"
  fi
  if [[ "$path" != /* ]]; then
    path="$PROJECT_ROOT/$path"
  fi
  printf '%s\n' "$path"
}

check_python_module() {
  local module_name="$1"
  python - "$module_name" <<'PY'
import importlib.util
import sys

module_name = sys.argv[1]
sys.exit(0 if importlib.util.find_spec(module_name) is not None else 1)
PY
}

check_retrieval_server() {
  local quiet=${1:-false}
  local report_failure=${2:-true}
  python - "$RETRIEVAL_SERVER_URL" "$quiet" "$report_failure" <<'PY'
import sys
import urllib.request

server_url = sys.argv[1].rstrip("/")
quiet = sys.argv[2].lower() in {"1", "true", "yes", "on"}
report_failure = sys.argv[3].lower() in {"1", "true", "yes", "on"}
health_url = f"{server_url}/health"

try:
    with urllib.request.urlopen(health_url, timeout=5) as response:
        body = response.read().decode()
        if response.status != 200:
            raise RuntimeError(f"unexpected status {response.status}: {body}")
except Exception as exc:
    if report_failure:
        print(f"Retrieval health check failed for {health_url}: {exc}", file=sys.stderr)
    sys.exit(1)

if not quiet:
    print(f"Retrieval health check passed: {health_url}")
    print(body)
PY
}

parse_retrieval_server_endpoint() {
  python - "$RETRIEVAL_SERVER_URL" <<'PY'
from urllib.parse import urlparse
import sys

server_url = sys.argv[1]
parsed = urlparse(server_url)

host = parsed.hostname or "127.0.0.1"
port = parsed.port or 8000
path = parsed.path.rstrip("/")

if parsed.scheme not in {"", "http"}:
    raise SystemExit(f"Unsupported RETRIEVAL_SERVER_URL scheme for local startup: {server_url}")
if host not in {"127.0.0.1", "localhost", "0.0.0.0"}:
    raise SystemExit(f"Local startup only supports local RETRIEVAL_SERVER_URL values, got: {server_url}")
if path not in {"", "/"}:
    raise SystemExit(f"Local startup requires a root RETRIEVAL_SERVER_URL without a path, got: {server_url}")

print(host)
print(port)
PY
}

find_running_retrieval_server_pids() {
  local parsed host port
  mapfile -t parsed < <(parse_retrieval_server_endpoint)
  host=${parsed[0]}
  port=${parsed[1]}

  pgrep -f "scripts/retrieval/server.py .*--port ${port} .*--host ${host}" || true
}

format_bytes() {
  local bytes="$1"
  if command -v numfmt >/dev/null 2>&1; then
    numfmt --to=iec-i --suffix=B "$bytes"
  else
    printf '%s bytes\n' "$bytes"
  fi
}

format_percent() {
  local pos="$1"
  local size="$2"
  awk -v pos="$pos" -v size="$size" 'BEGIN { if (size <= 0) printf "0.0%%"; else printf "%.1f%%", (100 * pos / size) }'
}

print_pid_loading_progress() {
  local pid="$1"
  local found=0

  if [[ ! -d "/proc/$pid" ]]; then
    echo "pid $pid is no longer present."
    return
  fi

  for fd_path in /proc/"$pid"/fd/*; do
    [[ -e "$fd_path" ]] || continue

    local target fd pos size file_name
    target=$(readlink "$fd_path" 2>/dev/null || true)

    case "$target" in
      *"/corpus.json"|*"/e5_Flat.index")
        fd=$(basename "$fd_path")
        pos=$(awk '/^pos:/{print $2}' "/proc/$pid/fdinfo/$fd" 2>/dev/null || true)
        size=$(stat -c %s "$target" 2>/dev/null || true)
        file_name=$(basename "$target")
        if [[ -n "$pos" && -n "$size" ]]; then
          echo "pid $pid loading ${file_name}: $(format_bytes "$pos") / $(format_bytes "$size") ($(format_percent "$pos" "$size"))"
        else
          echo "pid $pid has ${file_name} open."
        fi
        found=1
        ;;
    esac
  done

  if (( found == 0 )); then
    echo "pid $pid is running but is not currently holding corpus/index open."
  fi
}

show_status() {
  local -a pids=()
  mapfile -t pids < <(find_running_retrieval_server_pids)

  echo "Retrieval server URL: $RETRIEVAL_SERVER_URL"
  if check_retrieval_server false false; then
    :
  else
    echo "Retrieval server health: not ready"
  fi

  if (( ${#pids[@]} == 0 )); then
    echo "Retrieval server process: not running"
  else
    local pid_csv
    pid_csv=$(IFS=,; echo "${pids[*]}")
    echo "Retrieval server process(es): ${pids[*]}"
    ps -o pid,etime,%cpu,%mem,rss,vsz,state,cmd -p "$pid_csv"
    for pid in "${pids[@]}"; do
      print_pid_loading_progress "$pid"
    done
  fi

  echo "Retrieval server log: $RETRIEVAL_SERVER_LOG_PATH"
  if [[ -f "$RETRIEVAL_SERVER_PID_FILE" ]]; then
    echo "Retrieval server pid file: $RETRIEVAL_SERVER_PID_FILE ($(cat "$RETRIEVAL_SERVER_PID_FILE"))"
  fi
}

tail_log() {
  if [[ -f "$RETRIEVAL_SERVER_LOG_PATH" ]]; then
    tail -n "${RETRIEVAL_SERVER_LOG_TAIL_LINES}" "$RETRIEVAL_SERVER_LOG_PATH"
  else
    echo "Retrieval server log does not exist yet: $RETRIEVAL_SERVER_LOG_PATH"
  fi
}

start_server_process() {
  local parsed host port
  mapfile -t parsed < <(parse_retrieval_server_endpoint)
  host=${parsed[0]}
  port=${parsed[1]}

  mkdir -p "$RETRIEVAL_SERVER_STATE_DIR"

  echo "Starting retrieval server in background..."
  echo "Retrieval server log: $RETRIEVAL_SERVER_LOG_PATH"

  PYTHONUNBUFFERED=1 \
  HOST="$host" \
  DEVICE="$RETRIEVAL_SERVER_DEVICE" \
  GPU_MEMORY_LIMIT_MB="$RETRIEVAL_SERVER_GPU_MEMORY_LIMIT_MB" \
  SEARCHR1_DATA_ROOT="$SEARCHR1_DATA_ROOT" \
  SEARCHR1_INDEX_DIR="$SEARCHR1_INDEX_DIR" \
  bash scripts/retrieval/launch_server.sh "$SEARCHR1_INDEX_DIR" "$port" \
    >"$RETRIEVAL_SERVER_LOG_PATH" 2>&1 &

  local pid=$!
  echo "$pid" > "$RETRIEVAL_SERVER_PID_FILE"
  echo "Started retrieval server pid $pid"
}

wait_for_server() {
  local elapsed=0
  local last_report=-999999
  local -a pids=()

  while (( elapsed < RETRIEVAL_SERVER_STARTUP_TIMEOUT_SECONDS )); do
    if check_retrieval_server false false; then
      return 0
    fi

    mapfile -t pids < <(find_running_retrieval_server_pids)
    if (( ${#pids[@]} == 0 )); then
      echo "Retrieval server process is not running." >&2
      echo "Recent retrieval server log output:" >&2
      tail_log >&2
      return 1
    fi

    if (( elapsed - last_report >= RETRIEVAL_SERVER_PROGRESS_REPORT_INTERVAL_SECONDS )); then
      echo "Waiting for retrieval server to become healthy (${elapsed}s elapsed)..."
      show_status
      last_report=$elapsed
    fi

    sleep "$RETRIEVAL_SERVER_POLL_INTERVAL_SECONDS"
    elapsed=$((elapsed + RETRIEVAL_SERVER_POLL_INTERVAL_SECONDS))
  done

  echo "Timed out waiting ${RETRIEVAL_SERVER_STARTUP_TIMEOUT_SECONDS}s for retrieval server health check." >&2
  echo "Current retrieval server status:" >&2
  show_status >&2
  echo "Recent retrieval server log output:" >&2
  tail_log >&2
  return 1
}

start_server() {
  local -a pids=()

  if ! check_python_module "sentence_transformers"; then
    echo "Missing Python dependency: sentence-transformers" >&2
    echo "Install it in the active environment with:" >&2
    echo "  python -m pip install sentence-transformers" >&2
    echo "Or run the search setup path:" >&2
    echo "  bash scripts/setup_ragen.sh --with-search" >&2
    exit 1
  fi

  if check_retrieval_server false false; then
    echo "Retrieval server is already healthy."
    return 0
  fi

  mapfile -t pids < <(find_running_retrieval_server_pids)
  if (( ${#pids[@]} > 0 )); then
    echo "Retrieval server process already exists (${pids[*]})."
  else
    start_server_process
  fi

  wait_for_server
}

stop_server() {
  local -a pids=()
  mapfile -t pids < <(find_running_retrieval_server_pids)

  if (( ${#pids[@]} == 0 )); then
    echo "Retrieval server is not running."
    rm -f "$RETRIEVAL_SERVER_PID_FILE"
    return 0
  fi

  echo "Stopping retrieval server process(es): ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
  sleep 1
  rm -f "$RETRIEVAL_SERVER_PID_FILE"

  mapfile -t pids < <(find_running_retrieval_server_pids)
  if (( ${#pids[@]} > 0 )); then
    echo "Some retrieval server process(es) are still alive: ${pids[*]}" >&2
    return 1
  fi
}

COMMAND=${1:-start}

SEARCHR1_DATA_ROOT=${SEARCHR1_DATA_ROOT:-/projects/bflz/searchr1_data}
SEARCHR1_INDEX_DIR=${SEARCHR1_INDEX_DIR:-${SEARCHR1_DATA_ROOT}/search_data/prebuilt_indices}
RETRIEVAL_SERVER_URL=${RETRIEVAL_SERVER_URL:-http://127.0.0.1:8000}
# Use a dedicated retrieval GPU by default. Override with values like
# `cpu`, `cuda`, or `cuda:1` if needed.
RETRIEVAL_SERVER_DEVICE=${RETRIEVAL_SERVER_DEVICE:-cuda:0}
RETRIEVAL_SERVER_GPU_MEMORY_LIMIT_MB=${RETRIEVAL_SERVER_GPU_MEMORY_LIMIT_MB:-6144}
RETRIEVAL_SERVER_STARTUP_TIMEOUT_SECONDS=${RETRIEVAL_SERVER_STARTUP_TIMEOUT_SECONDS:-1800}
RETRIEVAL_SERVER_POLL_INTERVAL_SECONDS=${RETRIEVAL_SERVER_POLL_INTERVAL_SECONDS:-5}
RETRIEVAL_SERVER_PROGRESS_REPORT_INTERVAL_SECONDS=${RETRIEVAL_SERVER_PROGRESS_REPORT_INTERVAL_SECONDS:-30}
RESULT_ROOT=${RESULT_ROOT:-"$PWD/results/estimation"}
RETRIEVAL_SERVER_STATE_DIR=${RETRIEVAL_SERVER_STATE_DIR:-"$RESULT_ROOT/searchr1-server"}
RETRIEVAL_SERVER_LOG_PATH=${RETRIEVAL_SERVER_LOG_PATH:-"$RETRIEVAL_SERVER_STATE_DIR/retrieval_server.log"}
RETRIEVAL_SERVER_PID_FILE=${RETRIEVAL_SERVER_PID_FILE:-"$RETRIEVAL_SERVER_STATE_DIR/retrieval_server.pid"}
RETRIEVAL_SERVER_LOG_TAIL_LINES=${RETRIEVAL_SERVER_LOG_TAIL_LINES:-80}

SEARCHR1_INDEX_DIR=$(resolve_path "$SEARCHR1_INDEX_DIR")

case "$COMMAND" in
  start)
    start_server
    ;;
  wait)
    wait_for_server
    ;;
  status)
    show_status
    ;;
  stop)
    stop_server
    ;;
  logs)
    tail_log
    ;;
  restart)
    stop_server || true
    start_server
    ;;
  *)
    echo "Usage: bash scripts/evaluation-scripts/origin/searchr1_server.sh [start|wait|status|stop|restart|logs]" >&2
    exit 1
    ;;
esac
