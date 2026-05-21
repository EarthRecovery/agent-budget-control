#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

STAGES_TEXT="${1:-${STAGES:-all}}"
DRY_RUN="${DRY_RUN:-0}"
RUN_EXPERIMENTS="${RUN_EXPERIMENTS:-all}"

export VENV_PATH="${VENV_PATH:-/workspace/.venv}"
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/budget-rl}"
export SFT_ABLATION="${SFT_ABLATION:-sft_interval_pct30}"
export SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-5}"
export RL_KL="${RL_KL:-0.05}"
export NGPUS="${NGPUS:-1}"
export TP_SIZE="${TP_SIZE:-1}"
export NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-128}"
export VAL_GROUP_SIZE="${VAL_GROUP_SIZE:-1}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\",\"wandb\"]}"
export SFT_TRAIN_BS="${SFT_TRAIN_BS:-8}"
export SFT_MICRO_BS="${SFT_MICRO_BS:-1}"
export RL_BATCH_SIZE="${RL_BATCH_SIZE:-16}"
export RL_PPO_MICRO_BS="${RL_PPO_MICRO_BS:-1}"
export RL_LOGPROB_MICRO_BS="${RL_LOGPROB_MICRO_BS:-1}"
export RL_ROLLOUT_N="${RL_ROLLOUT_N:-8}"

QWEN3_4B_MODEL="${QWEN3_4B_MODEL:-Qwen/Qwen3-4B}"
# Llama 3.1's public HF instruct checkpoint is 8B. Override this if you have an internal 7B path.
LLAMA31_MODEL="${LLAMA31_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
QWEN25_7B_MODEL="${QWEN25_7B_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

SEARCH_SERVER_PID=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/budget-rl/run_required_experiments.sh [all|rollout|prepare|sft|rl|rollout,prepare,...]

Runs the five requested budget-RL experiments:
  1. sokoban_qwen3_4b
  2. sokoban_llama31_8b
  3. warehouse_qwen25_7b
  4. searchr1_qwen25_7b
  5. swebench_qwen25_7b

Common controls:
  DRY_RUN=1                         print commands without heavy execution
  RUN_EXPERIMENTS=name1,name2        run a subset
  VENV_PATH=/workspace/.venv         runtime env
  NGPUS=1 TP_SIZE=1                  GPU layout
  NUM_TRAJECTORIES=128               rollout count for live RAGEN tasks
  WANDB_API_KEY=...                  required for real runs unless WANDB_MODE=offline

External rollout sources for non-live tasks:
  WAREHOUSE_ROLLOUT_SOURCE=/path/to/warehouse_rollouts.json-or-dir
  SWE_INPUT_DIR=/path/to/swebench_origin_dir
  SWE_ROLLOUT_SOURCE=/path/to/preconverted_swe_dialogues.json-or-dir

SearchR1:
  START_SEARCH_SERVER=1              launch scripts/retrieval/launch_server.sh
  SEARCHR1_DATA_ROOT=/path/to/searchr1_data
  SEARCH_MOCK_MODE=true              skip retrieval server for smoke rollouts
EOF
}

contains_experiment() {
  local name="$1"
  [[ "$RUN_EXPERIMENTS" = "all" ]] && return 0
  [[ ",${RUN_EXPERIMENTS// /,}," == *",$name,"* ]]
}

normalize_path() {
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

login_wandb_if_needed() {
  if [[ "$DRY_RUN" = "1" || "${WANDB_MODE:-}" = "offline" ]]; then
    return 0
  fi
  : "${WANDB_API_KEY:?WANDB_API_KEY is required for real runs. Export it or set WANDB_MODE=offline.}"
  if [[ "${AUTO_WANDB_LOGIN:-1}" = "1" ]]; then
    if [[ -x "$VENV_PATH/bin/wandb" ]]; then
      "$VENV_PATH/bin/wandb" login --relogin "$WANDB_API_KEY"
    elif command -v wandb >/dev/null 2>&1; then
      wandb login --relogin "$WANDB_API_KEY"
    else
      echo "wandb CLI not found; continuing with WANDB_API_KEY in environment."
    fi
  fi
}

search_server_healthy() {
  local url="${RETRIEVAL_SERVER_URL:-http://127.0.0.1:${SEARCH_SERVER_PORT:-8000}}/health"
  command -v curl >/dev/null 2>&1 && curl -fsS "$url" >/dev/null 2>&1
}

maybe_start_search_server() {
  local exp_base="$1"
  if [[ "${SEARCH_MOCK_MODE:-false}" =~ ^(1|true|yes|on)$ ]]; then
    return 0
  fi
  if [[ "${START_SEARCH_SERVER:-1}" != "1" || "$DRY_RUN" = "1" ]]; then
    return 0
  fi
  if search_server_healthy; then
    echo "SearchR1 retrieval server is already healthy at ${RETRIEVAL_SERVER_URL:-http://127.0.0.1:${SEARCH_SERVER_PORT:-8000}}"
    return 0
  fi

  local data_root="${SEARCHR1_DATA_ROOT:-/projects/bflz/searchr1_data}"
  local index_dir="${SEARCHR1_INDEX_DIR:-$data_root/search_data/prebuilt_indices}"
  index_dir="$(normalize_path "$index_dir")"
  if [[ ! -f "$index_dir/corpus.json" || ! -f "$index_dir/e5_Flat.index" ]]; then
    echo "SearchR1 index files not found under $index_dir" >&2
    echo "Run: SEARCHR1_DATA_ROOT=$data_root bash scripts/setup_ragen.sh --with-search" >&2
    exit 2
  fi

  mkdir -p "$exp_base"
  local log_file="$exp_base/searchr1_server.log"
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

cleanup() {
  if [[ -n "$SEARCH_SERVER_PID" ]]; then
    kill "$SEARCH_SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

add_source_env() {
  local var_name="$1"
  local value="$2"
  local -n out_ref="$3"
  [[ -z "$value" ]] && return 0
  local path
  path="$(normalize_path "$value")"
  if [[ -d "$path" ]]; then
    out_ref+=("ROLLOUT_SOURCE_DIR=$path")
  else
    out_ref+=("$var_name=$path")
  fi
}

run_one() {
  local name="$1"
  local task="$2"
  local model="$3"

  if ! contains_experiment "$name"; then
    echo "skip $name"
    return 0
  fi

  local exp_base="$DATA_ROOT/$name"
  local -a envs=(
    "TASK=$task"
    "ROLLOUT_MODEL=$model"
    "LEARNER_MODEL=$model"
    "TOKENIZER=$model"
    "EXP_NAME=$name"
    "EXP_BASE=$exp_base"
    "WANDB_RUN_GROUP=$name"
    "DRY_RUN=$DRY_RUN"
  )

  case "$task" in
    warehouse)
      add_source_env ROLLOUT_SOURCE_JSON "${WAREHOUSE_ROLLOUT_SOURCE:-}" envs
      ;;
    swebench)
      if [[ -n "${SWE_INPUT_DIR:-}" ]]; then
        envs+=("SWE_INPUT_DIR=$(normalize_path "$SWE_INPUT_DIR")")
      else
        add_source_env ROLLOUT_SOURCE_JSON "${SWE_ROLLOUT_SOURCE:-}" envs
      fi
      ;;
    searchr1)
      envs+=(
        "SEARCH_DATA_PATH=${SEARCH_DATA_PATH:-${SEARCHR1_DATA_ROOT:-/projects/bflz/searchr1_data}/data/search/train.parquet}"
        "RETRIEVAL_SERVER_URL=${RETRIEVAL_SERVER_URL:-http://127.0.0.1:${SEARCH_SERVER_PORT:-8000}}"
        "SEARCH_MOCK_MODE=${SEARCH_MOCK_MODE:-false}"
      )
      maybe_start_search_server "$exp_base"
      ;;
  esac

  echo
  echo "=== $name ($task, $model): $STAGES_TEXT ==="
  env "${envs[@]}" bash "$SCRIPT_DIR/run_budget_rl_pipeline.sh" "$STAGES_TEXT"
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

login_wandb_if_needed

run_one sokoban_qwen3_4b sokoban "$QWEN3_4B_MODEL"
run_one sokoban_llama31_8b sokoban "$LLAMA31_MODEL"
run_one warehouse_qwen25_7b warehouse "$QWEN25_7B_MODEL"
run_one searchr1_qwen25_7b searchr1 "$QWEN25_7B_MODEL"
run_one swebench_qwen25_7b swebench "$QWEN25_7B_MODEL"
