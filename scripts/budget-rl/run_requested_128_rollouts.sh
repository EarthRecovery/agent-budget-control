#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

ROLLOUTS_TEXT="${1:-${ROLLOUTS:-all}}"
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/data/budget-rl/requested-128-rollouts}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-128}"
VAL_GROUP_SIZE="${VAL_GROUP_SIZE:-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ragenv2}"

QWEN25_7B_MODEL="${QWEN25_7B_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
QWEN3_4B_MODEL="${QWEN3_4B_MODEL:-Qwen/Qwen3-4B}"
# Public Llama 3.1 instruct checkpoint is 8B. Override if you have a local/internal 7B path.
LLAMA31_7B_MODEL="${LLAMA31_7B_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

default_searchr1_data_root() {
  if [[ -n "${SEARCHR1_DATA_ROOT:-}" ]]; then
    printf '%s\n' "$SEARCHR1_DATA_ROOT"
  elif [[ -d "/projects/bflz" && -w "/projects/bflz" ]]; then
    printf '%s\n' "/projects/bflz/searchr1_data"
  elif [[ -d "/projects/e32695" && -w "/projects/e32695" ]]; then
    printf '%s\n' "/projects/e32695/searchr1_data"
  else
    printf '%s\n' "$PROJECT_ROOT/data/searchr1_data"
  fi
}
SEARCHR1_DATA_ROOT="$(default_searchr1_data_root)"
default_search_data_path() {
  if [[ -f "$SEARCHR1_DATA_ROOT/data/search/train.parquet" ]]; then
    printf '%s\n' "$SEARCHR1_DATA_ROOT/data/search/train.parquet"
  elif [[ -f "/projects/e32695/Search-R1/data/nq_search/train.parquet" ]]; then
    printf '%s\n' "/projects/e32695/Search-R1/data/nq_search/train.parquet"
  else
    printf '%s\n' "$SEARCHR1_DATA_ROOT/data/search/train.parquet"
  fi
}
SEARCH_DATA_PATH="${SEARCH_DATA_PATH:-$(default_search_data_path)}"
RETRIEVAL_SERVER_URL="${RETRIEVAL_SERVER_URL:-http://127.0.0.1:${SEARCH_SERVER_PORT:-8000}}"
SEARCH_MOCK_MODE="${SEARCH_MOCK_MODE:-false}"
START_SEARCH_SERVER="${START_SEARCH_SERVER:-0}"
SEARCH_SERVER_PID=""
SEARCHR1_ROLLOUT_GPUS="${SEARCHR1_ROLLOUT_GPUS:-1,2,3}"
SEARCHR1_ROLLOUT_TP_SIZE="${SEARCHR1_ROLLOUT_TP_SIZE:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/budget-rl/run_requested_128_rollouts.sh [all|searchr1_qwen25_7b|sokoban_qwen3_4b|sokoban_llama31_7b|searchr1_backend|name1,name2]

Runs exactly the requested 128-rollout jobs:
  1. searchr1_qwen25_7b: Qwen/Qwen2.5-7B-Instruct, SearchR1, 128 rollouts
  2. sokoban_qwen3_4b:   Qwen/Qwen3-4B, Sokoban 6x6 1box, 128 rollouts
  3. sokoban_llama31_7b: Llama 3.1, Sokoban 6x6 1box, 128 rollouts

Common overrides:
  OUTPUT_BASE=/path/to/output-root
  CUDA_VISIBLE_DEVICES=0
  TP_SIZE=1
  NUM_TRAJECTORIES=128
  DRY_RUN=1

Model overrides:
  QWEN25_7B_MODEL=/path/or/hf-id
  QWEN3_4B_MODEL=/path/or/hf-id
  LLAMA31_7B_MODEL=/path/or/hf-id

SearchR1:
  Defaults to three parallel single-GPU rollout shards on GPUs 1,2,3 so GPU 0
  can be reserved for the retrieval server:
    SEARCHR1_ROLLOUT_GPUS=1,2,3
    SEARCHR1_ROLLOUT_TP_SIZE=1

  Required:
    1. A parquet dataset at SEARCH_DATA_PATH.
    2. A healthy retrieval server at RETRIEVAL_SERVER_URL, or START_SEARCH_SERVER=1
       with SEARCHR1_INDEX_DIR containing corpus.json and e5_Flat.index.

  Start the RAGEN dense retrieval server automatically:
    START_SEARCH_SERVER=1 SEARCHR1_INDEX_DIR=/path/to/prebuilt_indices \
      bash scripts/budget-rl/run_requested_128_rollouts.sh searchr1_qwen25_7b

  Or start it manually:
    bash scripts/retrieval/launch_server.sh /path/to/prebuilt_indices 8000

  Backend-only mode; downloads missing backend data, then serves in foreground:
    bash scripts/budget-rl/run_requested_128_rollouts.sh searchr1_backend

  SEARCH_MOCK_MODE=true can be used for a command smoke test without retrieval.
  If START_SEARCH_SERVER=1 and backend files are missing, this script downloads
  wiki-18 corpus/index data and builds corpus.json + e5_Flat.index before launch.
EOF
}

contains_rollout() {
  local name="$1"
  [[ "$ROLLOUTS_TEXT" = "all" ]] && return 0
  [[ ",${ROLLOUTS_TEXT// /,}," == *",$name,"* ]]
}

search_mock_enabled() {
  [[ "${SEARCH_MOCK_MODE,,}" =~ ^(1|true|yes|on)$ ]]
}

search_server_healthy() {
  local health_url="${RETRIEVAL_SERVER_URL%/}/health"
  command -v curl >/dev/null 2>&1 && curl -fsS "$health_url" >/dev/null 2>&1
}

prepare_search_dataset_if_missing() {
  if [[ -f "$SEARCH_DATA_PATH" ]]; then
    return 0
  fi
  if [[ "${AUTO_DOWNLOAD_SEARCH_DATA:-1}" != "1" ]]; then
    return 1
  fi

  local output_dir
  output_dir="$(dirname "$SEARCH_DATA_PATH")"
  mkdir -p "$output_dir"
  echo "SearchR1 dataset missing; preparing HotpotQA parquet under $output_dir"
  python scripts/prepare_search_data.py --output_dir "$output_dir"
}

ensure_search_backend_data() {
  local index_dir="${SEARCHR1_INDEX_DIR:-$SEARCHR1_DATA_ROOT/search_data/prebuilt_indices}"
  if [[ -f "$index_dir/corpus.json" && -f "$index_dir/e5_Flat.index" ]]; then
    SEARCHR1_RESOLVED_INDEX_DIR="$index_dir"
    return 0
  fi
  if [[ "${AUTO_DOWNLOAD_SEARCH_BACKEND_DATA:-1}" != "1" ]]; then
    return 1
  fi

  local data_dir
  if [[ "$(basename "$index_dir")" = "prebuilt_indices" ]]; then
    data_dir="$(dirname "$index_dir")"
  else
    data_dir="$SEARCHR1_DATA_ROOT/search_data"
    index_dir="$data_dir/prebuilt_indices"
  fi

  mkdir -p "$data_dir" "$index_dir"
  echo "SearchR1 backend data missing under $index_dir; downloading to $data_dir"
  WIKI_CORPUS_REPO_ID="${WIKI_CORPUS_REPO_ID:-PeterJinGo/wiki-18-corpus}" \
  WIKI_E5_INDEX_REPO_ID="${WIKI_E5_INDEX_REPO_ID:-PeterJinGo/wiki-18-e5-index}" \
  PYTHONUNBUFFERED=1 \
    python -u scripts/download_search_index.py --data_dir "$data_dir"

  if [[ ! -f "$index_dir/e5_Flat.index" ]]; then
    if [[ -f "$index_dir/part_aa" && -f "$index_dir/part_ab" ]]; then
      echo "Merging FAISS shards into $index_dir/e5_Flat.index"
      cat "$index_dir/part_aa" "$index_dir/part_ab" > "$index_dir/e5_Flat.index"
    else
      echo "Missing FAISS shards in $index_dir; expected part_aa and part_ab" >&2
      return 1
    fi
  fi

  if [[ ! -f "$index_dir/corpus.json" ]]; then
    local wiki_jsonl="$data_dir/wikipedia/wiki-18.jsonl"
    if [[ ! -f "$wiki_jsonl" ]]; then
      echo "Missing Wikipedia corpus at $wiki_jsonl" >&2
      return 1
    fi
    echo "Converting $wiki_jsonl -> $index_dir/corpus.json"
    PYTHONUNBUFFERED=1 python -u - "$wiki_jsonl" "$index_dir/corpus.json" <<'PY'
import json
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
output_path.parent.mkdir(parents=True, exist_ok=True)

corpus = []
with input_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        doc = json.loads(line)
        text = doc.get("text") or doc.get("contents") or doc.get("content") or ""
        title = doc.get("title") or ""
        if title and text:
            corpus.append(f"{title} {text}")
        elif text:
            corpus.append(text)

with output_path.open("w", encoding="utf-8") as f:
    json.dump(corpus, f)
print(f"Wrote {len(corpus)} documents to {output_path}")
if not corpus:
    raise SystemExit("No corpus documents were written.")
PY
  fi

  if [[ ! -f "$index_dir/corpus.json" || ! -f "$index_dir/e5_Flat.index" ]]; then
    echo "SearchR1 backend data setup did not produce corpus.json and e5_Flat.index in $index_dir" >&2
    return 1
  fi

  SEARCHR1_RESOLVED_INDEX_DIR="$index_dir"
}

maybe_start_search_server() {
  if search_mock_enabled || [[ "$START_SEARCH_SERVER" != "1" ]]; then
    return 0
  fi
  if search_server_healthy; then
    echo "SearchR1 retrieval server is already healthy at $RETRIEVAL_SERVER_URL"
    return 0
  fi

  if ! ensure_search_backend_data; then
    echo "Failed to prepare SearchR1 backend data." >&2
    exit 2
  fi
  local index_dir="$SEARCHR1_RESOLVED_INDEX_DIR"

  mkdir -p "$OUTPUT_BASE"
  local log_file="$OUTPUT_BASE/searchr1_server.log"
  echo "Starting SearchR1 retrieval server; log: $log_file"
  HOST="${SEARCH_SERVER_HOST:-127.0.0.1}" \
  DEVICE="${SEARCH_SERVER_DEVICE:-cpu}" \
  GPU_MEMORY_LIMIT_MB="${SEARCH_SERVER_GPU_MEMORY_LIMIT_MB:-1024}" \
    bash "$PROJECT_ROOT/scripts/retrieval/launch_server.sh" "$index_dir" "${SEARCH_SERVER_PORT:-8000}" \
    >"$log_file" 2>&1 &
  SEARCH_SERVER_PID="$!"

  for _ in $(seq 1 90); do
    if search_server_healthy; then
      echo "SearchR1 retrieval server is healthy."
      return 0
    fi
    sleep 2
  done

  echo "SearchR1 retrieval server did not become healthy. See $log_file" >&2
  exit 2
}

preflight_searchr1() {
  if [[ ! -f "$SEARCH_DATA_PATH" ]] && ! prepare_search_dataset_if_missing; then
    cat >&2 <<EOF
SearchR1 dataset not found:
  SEARCH_DATA_PATH=$SEARCH_DATA_PATH

Set SEARCH_DATA_PATH to an existing parquet file, or prepare HotpotQA data:
  python scripts/prepare_search_data.py --output_dir "\$SEARCHR1_DATA_ROOT/data/search"

This machine currently has an older Search-R1/NQ parquet here if you want to use it:
  /projects/e32695/Search-R1/data/nq_search/train.parquet
EOF
    exit 2
  fi

  if search_mock_enabled || [[ "${DRY_RUN:-0}" = "1" ]]; then
    return 0
  fi

  if [[ "$START_SEARCH_SERVER" = "1" ]]; then
    maybe_start_search_server
    return 0
  fi

  if ! search_server_healthy; then
    cat >&2 <<EOF
SearchR1 retrieval server is not healthy:
  RETRIEVAL_SERVER_URL=$RETRIEVAL_SERVER_URL

Start the RAGEN dense retrieval backend first:
  bash scripts/retrieval/launch_server.sh /path/to/prebuilt_indices ${SEARCH_SERVER_PORT:-8000}

Or let this script start it, if SEARCHR1_INDEX_DIR contains corpus.json and e5_Flat.index:
  START_SEARCH_SERVER=1 SEARCHR1_INDEX_DIR=/path/to/prebuilt_indices \\
    bash scripts/budget-rl/run_requested_128_rollouts.sh searchr1_qwen25_7b

For a no-backend smoke test only:
  SEARCH_MOCK_MODE=true bash scripts/budget-rl/run_requested_128_rollouts.sh searchr1_qwen25_7b
EOF
    exit 2
  fi
}

cleanup() {
  if [[ -n "$SEARCH_SERVER_PID" ]]; then
    kill "$SEARCH_SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

run_searchr1_backend() {
  if search_server_healthy; then
    echo "SearchR1 retrieval server is already healthy at $RETRIEVAL_SERVER_URL"
    return 0
  fi
  if ! ensure_search_backend_data; then
    echo "Failed to prepare SearchR1 backend data." >&2
    exit 2
  fi
  echo "Starting SearchR1 retrieval server in foreground: $SEARCHR1_RESOLVED_INDEX_DIR"
  HOST="${SEARCH_SERVER_HOST:-127.0.0.1}" \
  DEVICE="${SEARCH_SERVER_DEVICE:-cpu}" \
  GPU_MEMORY_LIMIT_MB="${SEARCH_SERVER_GPU_MEMORY_LIMIT_MB:-1024}" \
    exec bash "$PROJECT_ROOT/scripts/retrieval/launch_server.sh" \
      "$SEARCHR1_RESOLVED_INDEX_DIR" "${SEARCH_SERVER_PORT:-8000}"
}

run_rollout() {
  local name="$1"
  local task="$2"
  local model="$3"
  shift 3

  if ! contains_rollout "$name"; then
    echo "skip $name"
    return 0
  fi

  local output_dir="$OUTPUT_BASE/$name"
  local output_jsonl="$output_dir/rollouts.jsonl"
  mkdir -p "$output_dir"

  echo
  echo "=== $name ==="
  echo "model: $model"
  echo "task: $task"
  echo "output: $output_jsonl"

  env \
    PROJECT_ROOT="$PROJECT_ROOT" \
    CONDA_ENV_NAME="$CONDA_ENV_NAME" \
    TASK="$task" \
    ROLLOUT_MODEL="$model" \
    NUM_TRAJECTORIES="$NUM_TRAJECTORIES" \
    VAL_GROUP_SIZE="$VAL_GROUP_SIZE" \
    RUN_NAME="$name" \
    OUTPUT_DIR="$output_dir" \
    OUTPUT_JSONL="$output_jsonl" \
    "$@" \
    bash "$SCRIPT_DIR/run_model_rollout.sh" "$task"
}

run_parallel_rollout() {
  local name="$1"
  local task="$2"
  local model="$3"
  local gpu_csv="$4"
  local tp_size="$5"
  shift 5

  if ! contains_rollout "$name"; then
    echo "skip $name"
    return 0
  fi

  local -a parsed_gpus=()
  local -a gpus=()
  IFS=',' read -r -a parsed_gpus <<< "$gpu_csv"
  local gpu
  for gpu in "${parsed_gpus[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    [[ -n "$gpu" ]] && gpus+=("$gpu")
  done

  if (( ${#gpus[@]} <= 1 )); then
    run_rollout \
      "$name" \
      "$task" \
      "$model" \
      CUDA_VISIBLE_DEVICES="${gpus[0]:-${CUDA_VISIBLE_DEVICES:-0}}" \
      TP_SIZE="$tp_size" \
      "$@"
    return 0
  fi

  local output_dir="$OUTPUT_BASE/$name"
  local output_jsonl="$output_dir/rollouts.jsonl"
  local shard_root="$output_dir/shards"
  mkdir -p "$shard_root"

  local total_groups
  if [[ -n "${VAL_GROUPS:-}" ]]; then
    total_groups="$VAL_GROUPS"
  else
    total_groups=$(((NUM_TRAJECTORIES + VAL_GROUP_SIZE - 1) / VAL_GROUP_SIZE))
  fi
  if (( total_groups <= 0 )); then
    echo "No rollout groups requested for $name" >&2
    exit 2
  fi

  echo
  echo "=== $name ==="
  echo "model: $model"
  echo "task: $task"
  echo "parallel GPUs: ${gpus[*]}"
  echo "tensor parallel size per shard: $tp_size"
  echo "total groups: $total_groups"
  echo "output: $output_jsonl"

  local shard_count="${#gpus[@]}"
  local base_groups=$((total_groups / shard_count))
  local remainder=$((total_groups % shard_count))
  local start_group=0
  local -a pids=()
  local -a shard_logs=()
  local -a shard_files=()

  local idx shard_groups shard_dir shard_file shard_log shard_name shard_trajectories
  for idx in "${!gpus[@]}"; do
    shard_groups="$base_groups"
    if (( idx < remainder )); then
      shard_groups=$((shard_groups + 1))
    fi
    if (( shard_groups == 0 )); then
      continue
    fi

    shard_name="${name}_shard${idx}"
    shard_dir="$shard_root/$shard_name"
    shard_file="$shard_dir/rollouts.jsonl"
    shard_log="$shard_dir/run.log"
    shard_trajectories=$((shard_groups * VAL_GROUP_SIZE))
    mkdir -p "$shard_dir"

    echo "  shard $idx: gpu=${gpus[$idx]} start_group_index=$start_group groups=$shard_groups log=$shard_log"
    env \
      PROJECT_ROOT="$PROJECT_ROOT" \
      CONDA_ENV_NAME="$CONDA_ENV_NAME" \
      TASK="$task" \
      ROLLOUT_MODEL="$model" \
      NUM_TRAJECTORIES="$shard_trajectories" \
      VAL_GROUPS="$shard_groups" \
      VAL_GROUP_SIZE="$VAL_GROUP_SIZE" \
      VAL_START_GROUP_INDEX="$start_group" \
      CUDA_VISIBLE_DEVICES="${gpus[$idx]}" \
      TP_SIZE="$tp_size" \
      RUN_NAME="$shard_name" \
      OUTPUT_DIR="$shard_dir" \
      OUTPUT_JSONL="$shard_file" \
      "$@" \
      bash "$SCRIPT_DIR/run_model_rollout.sh" "$task" \
      >"$shard_log" 2>&1 &

    pids+=("$!")
    shard_logs+=("$shard_log")
    shard_files+=("$shard_file")
    start_group=$((start_group + shard_groups))
  done

  local failed=0
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      failed=1
      echo "Shard $idx failed. Recent log from ${shard_logs[$idx]}:" >&2
      tail -n 80 "${shard_logs[$idx]}" >&2 || true
    fi
  done

  if (( failed != 0 )); then
    echo "At least one $name shard failed; not merging outputs." >&2
    exit 2
  fi

  if [[ "${DRY_RUN:-0}" = "1" ]]; then
    echo "DRY_RUN=1; shard commands were rendered in their run.log files, skipping merge."
    return 0
  fi

  local shard_file_path
  for shard_file_path in "${shard_files[@]}"; do
    if [[ ! -f "$shard_file_path" ]]; then
      echo "Expected shard output was not written: $shard_file_path" >&2
      exit 2
    fi
  done

  local tmp_output="${output_jsonl}.tmp"
  : > "$tmp_output"
  for shard_file_path in "${shard_files[@]}"; do
    cat "$shard_file_path" >> "$tmp_output"
  done
  mv "$tmp_output" "$output_jsonl"
  echo "Merged ${#shard_files[@]} shard outputs into $output_jsonl"
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  searchr1_backend|backend)
    run_searchr1_backend
    exit 0
    ;;
esac

if contains_rollout searchr1_qwen25_7b; then
  preflight_searchr1
fi

run_parallel_rollout \
  searchr1_qwen25_7b \
  searchr1 \
  "$QWEN25_7B_MODEL" \
  "$SEARCHR1_ROLLOUT_GPUS" \
  "$SEARCHR1_ROLLOUT_TP_SIZE" \
  SEARCH_DATA_PATH="$SEARCH_DATA_PATH" \
  RETRIEVAL_SERVER_URL="$RETRIEVAL_SERVER_URL" \
  SEARCH_MOCK_MODE="$SEARCH_MOCK_MODE"

run_rollout \
  sokoban_qwen3_4b \
  sokoban \
  "$QWEN3_4B_MODEL" \
  SOKOBAN_DIM_X=6 \
  SOKOBAN_DIM_Y=6 \
  SOKOBAN_NUM_BOXES=1

run_rollout \
  sokoban_llama31_7b \
  sokoban \
  "$LLAMA31_7B_MODEL" \
  SOKOBAN_DIM_X=6 \
  SOKOBAN_DIM_Y=6 \
  SOKOBAN_NUM_BOXES=1
