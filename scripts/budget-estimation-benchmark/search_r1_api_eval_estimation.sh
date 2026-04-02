#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ragen

PROJECT_ROOT=${PROJECT_ROOT:-"$HOME/RAGEN-v2"}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:$PWD/verl"

# Default model is OpenAI GPT-5.2 Thinking.
: "${OPENAI_API_KEY:?Please export OPENAI_API_KEY before running this benchmark.}"

RUN_NAME=${RUN_NAME:-search_r1_api_eval_estimation}
MODEL_NAME=${MODEL_NAME:-OpenAI-5.2-Thinking}
VAL_GROUPS=${VAL_GROUPS:-512}
VAL_START_GROUP_INDEX=${VAL_START_GROUP_INDEX:-0}
VAL_ROLLOUT_CHUNK_SIZE=${VAL_ROLLOUT_CHUNK_SIZE:-0}
SEARCH_ENV_TAG=${SEARCH_ENV_TAG:-SearchQA}
SEARCH_DATA_PATH=${SEARCH_DATA_PATH:-data/search/train.parquet}
SEARCH_MOCK_MODE=${SEARCH_MOCK_MODE:-False}
RETRIEVAL_SERVER_URL=${RETRIEVAL_SERVER_URL:-http://127.0.0.1:8000}
MAX_TURN=${MAX_TURN:-5}
MAX_ACTIONS_PER_TURN=${MAX_ACTIONS_PER_TURN:-1}
MAX_ACTIONS_PER_TRAJ=${MAX_ACTIONS_PER_TRAJ:-10}
MAX_TOKENS=${MAX_TOKENS:-2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-5000}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-5000}
PROMPT_TOKEN_MARGIN=${PROMPT_TOKEN_MARGIN:-1024}
RESULT_ROOT=${RESULT_ROOT:-"$PWD/results/budget-estimation-benchmark"}
OUTPUT_DIR=${OUTPUT_DIR:-"$RESULT_ROOT/search-r1-512-gpt5.2-Thinking"}
HYDRA_DIR=${HYDRA_DIR:-"$OUTPUT_DIR/hydra/$RUN_NAME"}

mkdir -p "$OUTPUT_DIR" "$HYDRA_DIR"

python -m ragen.eval_api --config-name evaluate_api_llm \
  model_config.model_name="${MODEL_NAME}" \
  agent_proxy.enable_think=True \
  "agent_proxy.eval-estimation-single=False" \
  "agent_proxy.eval-estimation-multi=True" \
  agent_proxy.max_turn=${MAX_TURN} \
  agent_proxy.max_actions_per_turn=${MAX_ACTIONS_PER_TURN} \
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
  output.dir="${OUTPUT_DIR}" \
  output.filename="${RUN_NAME}.pkl" \
  output.append_timestamp=True \
  hydra.run.dir="${HYDRA_DIR}" \
  hydra.output_subdir=null
