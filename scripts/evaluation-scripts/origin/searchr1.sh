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

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

check_retrieval_server() {
  local quiet=${1:-false}
  python - "$RETRIEVAL_SERVER_URL" "$quiet" <<'PY'
import sys
import urllib.request

server_url = sys.argv[1].rstrip("/")
quiet = sys.argv[2].lower() in {"1", "true", "yes", "on"}
health_url = f"{server_url}/health"

try:
    with urllib.request.urlopen(health_url, timeout=5) as response:
        body = response.read().decode()
        if response.status != 200:
            raise RuntimeError(f"unexpected status {response.status}: {body}")
except Exception as exc:
    print(f"Retrieval health check failed for {health_url}: {exc}", file=sys.stderr)
    sys.exit(1)

if not quiet:
    print(f"Retrieval health check passed: {health_url}")
    print(body)
PY
}

# Default model is OpenAI GPT-5.2 Instant.
: "${OPENAI_API_KEY:?Please export OPENAI_API_KEY before running this benchmark.}"

RUN_NAME=${RUN_NAME:-search_r1_api_eval_estimation}
MODEL_NAME=${MODEL_NAME:-OpenAI-5.2-Instant}
VAL_GROUPS=${VAL_GROUPS:-4}
VAL_START_GROUP_INDEX=${VAL_START_GROUP_INDEX:-50}
VAL_ROLLOUT_CHUNK_SIZE=${VAL_ROLLOUT_CHUNK_SIZE:-0}
SEARCH_ENV_TAG=${SEARCH_ENV_TAG:-SearchQA}
SEARCHR1_DATA_ROOT=${SEARCHR1_DATA_ROOT:-/projects/bflz/searchr1_data}
SEARCH_DATA_DIR=${SEARCH_DATA_DIR:-${SEARCHR1_DATA_ROOT}/data/search}
SEARCH_DATA_PATH=${SEARCH_DATA_PATH:-${SEARCH_DATA_DIR}/train.parquet}
SEARCH_MOCK_MODE=${SEARCH_MOCK_MODE:-False}
RETRIEVAL_SERVER_URL=${RETRIEVAL_SERVER_URL:-http://127.0.0.1:8000}
MAX_TURN=${MAX_TURN:-10}
MAX_ACTIONS_PER_TURN=${MAX_ACTIONS_PER_TURN:-1}
MAX_ACTIONS_PER_TRAJ=${MAX_ACTIONS_PER_TRAJ:-10}
MAX_TOKENS=${MAX_TOKENS:-2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-50000}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-50000}
PROMPT_TOKEN_MARGIN=${PROMPT_TOKEN_MARGIN:-1024}
TRUNCATION_MODE=${TRUNCATION_MODE:-token}
NO_BUDGET_PROMPT=${NO_BUDGET_PROMPT:-True}
MAX_CONTEXT_TOKEN=${MAX_CONTEXT_TOKEN:-2500}
RESULT_ROOT=${RESULT_ROOT:-"$PWD/results/estimation"}
OUTPUT_DIR=${OUTPUT_DIR:-"$RESULT_ROOT/searchr1-origin-gpt5.2-instant-4-test4"}
HYDRA_DIR=${HYDRA_DIR:-"$OUTPUT_DIR/hydra/$RUN_NAME"}

DEFAULT_SEARCH_DATA_PATH=$(resolve_path "${SEARCH_DATA_DIR}/train.parquet")
SEARCH_DATA_PATH=$(resolve_path "$SEARCH_DATA_PATH")

if [[ ! -f "$SEARCH_DATA_PATH" ]]; then
  if [[ "$SEARCH_DATA_PATH" == "$DEFAULT_SEARCH_DATA_PATH" ]]; then
    echo "Search dataset not found at $SEARCH_DATA_PATH. Preparing HotpotQA parquet files under $SEARCH_DATA_DIR ..."
    python scripts/prepare_search_data.py --output_dir "$SEARCH_DATA_DIR"
  else
    echo "Search dataset not found at $SEARCH_DATA_PATH." >&2
    echo "Set SEARCH_DATA_PATH to an existing parquet file, or run: python scripts/prepare_search_data.py --output_dir $SEARCH_DATA_DIR" >&2
    exit 1
  fi
fi

if ! is_truthy "$SEARCH_MOCK_MODE"; then
  if ! check_retrieval_server; then
    echo "Retrieval server is required when SEARCH_MOCK_MODE=False." >&2
    echo "Start it separately with:" >&2
    echo "  bash scripts/evaluation-scripts/origin/searchr1_server.sh start" >&2
    echo "Inspect progress/status with:" >&2
    echo "  bash scripts/evaluation-scripts/origin/searchr1_server.sh status" >&2
    echo "View server logs with:" >&2
    echo "  bash scripts/evaluation-scripts/origin/searchr1_server.sh logs" >&2
    echo "Or rerun with SEARCH_MOCK_MODE=True to use mock retrieval." >&2
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR" "$HYDRA_DIR"

python -m ragen.eval_api --config-name evaluate_api_llm \
  model_config.model_name="${MODEL_NAME}" \
  agent_proxy.enable_think=True \
  ++agent_proxy.truncation_mode="${TRUNCATION_MODE}" \
  ++agent_proxy.max_context_token=${MAX_CONTEXT_TOKEN} \
  agent_proxy.max_turn=${MAX_TURN} \
  agent_proxy.max_actions_per_turn=${MAX_ACTIONS_PER_TURN} \
  agent_proxy.no_budget_prompt=${NO_BUDGET_PROMPT} \
  es_manager.val.env_groups=${VAL_GROUPS} \
  es_manager.val.group_size=1 \
  es_manager.val.start_group_index=${VAL_START_GROUP_INDEX} \
  es_manager.val.rollout_chunk_size=${VAL_ROLLOUT_CHUNK_SIZE} \
  "es_manager.val.env_configs.tags=[${SEARCH_ENV_TAG}]" \
  "es_manager.val.env_configs.n_groups=[${VAL_GROUPS}]" \
  "custom_envs.${SEARCH_ENV_TAG}.env_instruction='You are a search agent answering questions by searching for information. Choose exactly one valid next action. Use search[<query>] to retrieve evidence and finish[<answer>] only when you are ready to submit the final answer. Inside <answer>, output exactly one action string only.'" \
  ++custom_envs.${SEARCH_ENV_TAG}.env_config.train_path="${SEARCH_DATA_PATH}" \
  ++custom_envs.${SEARCH_ENV_TAG}.env_config.mock_mode=${SEARCH_MOCK_MODE} \
  ++custom_envs.${SEARCH_ENV_TAG}.env_config.retrieval_server_url="${RETRIEVAL_SERVER_URL}" \
  custom_envs.${SEARCH_ENV_TAG}.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
  custom_envs.${SEARCH_ENV_TAG}.max_tokens=${MAX_TOKENS} \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS} \
  actor_rollout_ref.rollout.response_length=${MAX_TOKENS} \
  model_config.prompt_token_margin=${PROMPT_TOKEN_MARGIN} \
  "output.dir='${OUTPUT_DIR}'" \
  "output.filename='${RUN_NAME}.pkl'" \
  "output.append_timestamp=True" \
  "hydra.run.dir='${HYDRA_DIR}'" \
  "hydra.output_subdir=null"
