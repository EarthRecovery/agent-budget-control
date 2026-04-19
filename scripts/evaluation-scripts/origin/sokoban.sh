#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ragenv2

PROJECT_ROOT=${PROJECT_ROOT:-"$HOME/agent-budget-control"}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:$PWD/verl"

RUN_NAME=${RUN_NAME:-sokoban_api_eval_estimation}
MODEL_NAME=${MODEL_NAME:-OpenRouter-Gemini-3.1-Pro-Preview}
VAL_GROUPS=${VAL_GROUPS:-128}
VAL_ROLLOUT_CHUNK_SIZE=${VAL_ROLLOUT_CHUNK_SIZE:-32}
MAX_TURN=${MAX_TURN:-10}
MAX_ACTIONS_PER_TURN=${MAX_ACTIONS_PER_TURN:-3}
MAX_ACTIONS_PER_TRAJ=${MAX_ACTIONS_PER_TRAJ:-30}
SOKOBAN_DIM_X=${SOKOBAN_DIM_X:-8}
SOKOBAN_DIM_Y=${SOKOBAN_DIM_Y:-8}
SOKOBAN_NUM_BOXES=${SOKOBAN_NUM_BOXES:-2}
SOKOBAN_SEARCH_DEPTH=${SOKOBAN_SEARCH_DEPTH:-30}
VAL_START_GROUP_INDEX=${VAL_START_GROUP_INDEX:-0}
MAX_TOKENS=${MAX_TOKENS:-800}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-81920}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-81920}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-4}
API_BATCH_SIZE=${API_BATCH_SIZE:-16}
API_CONNECT_TIMEOUT_SECONDS=${API_CONNECT_TIMEOUT_SECONDS:-10}
API_REQUEST_TIMEOUT_SECONDS=${API_REQUEST_TIMEOUT_SECONDS:-180}
TRUNCATION_MODE=${TRUNCATION_MODE:-token}
NO_BUDGET_PROMPT=${NO_BUDGET_PROMPT:-True}
RESULT_ROOT=${RESULT_ROOT:-"$PWD/results/estimation"}
OUTPUT_DIR=${OUTPUT_DIR:-"$RESULT_ROOT/sokoban-origin-gemini-3.1-pro-preview-128-main"}
HYDRA_DIR=${HYDRA_DIR:-"$OUTPUT_DIR/hydra/$RUN_NAME"}
MAX_CONTEXT_TOKEN=${MAX_CONTEXT_TOKEN:-2500}

require_api_key_for_model() {
  case "$1" in
    OpenRouter-*|openrouter/*|deepseek/deepseek-v3.2|minimax/minimax-m2.5)
      : "${OPENROUTER_API_KEY:?Please export OPENROUTER_API_KEY before running this benchmark.}"
      ;;
    OpenAI-*|gpt-*|o*)
      : "${OPENAI_API_KEY:?Please export OPENAI_API_KEY before running this benchmark.}"
      ;;
    Claude-*|claude-*)
      : "${ANTHROPIC_API_KEY:?Please export ANTHROPIC_API_KEY before running this benchmark.}"
      ;;
    Gemini-*|gemini-*)
      : "${GEMINI_API_KEY:?Please export GEMINI_API_KEY before running this benchmark.}"
      ;;
  esac
}

require_api_key_for_model "$MODEL_NAME"

mkdir -p "$OUTPUT_DIR" "$HYDRA_DIR"

python -m ragen.eval_api --config-name evaluate_api_llm \
  model_config.model_name="${MODEL_NAME}" \
  model_config.max_concurrency=${MAX_CONCURRENCY} \
  model_config.api_batch_size=${API_BATCH_SIZE} \
  model_config.api_connect_timeout_seconds=${API_CONNECT_TIMEOUT_SECONDS} \
  model_config.api_request_timeout_seconds=${API_REQUEST_TIMEOUT_SECONDS} \
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
  "es_manager.val.env_configs.tags=[CoordSokoban]" \
  "es_manager.val.env_configs.n_groups=[${VAL_GROUPS}]" \
  "custom_envs.CoordSokoban.env_instruction='You are solving the Sokoban puzzle. Push all boxes to targets. You are given the grid and zero-indexed coordinates of the player, boxes, and targets. You can push but not pull boxes, and cannot push a box through a wall.'" \
  ++custom_envs.CoordSokoban.env_config.dim_x=${SOKOBAN_DIM_X} \
  ++custom_envs.CoordSokoban.env_config.dim_y=${SOKOBAN_DIM_Y} \
  ++custom_envs.CoordSokoban.env_config.num_boxes=${SOKOBAN_NUM_BOXES} \
  ++custom_envs.CoordSokoban.env_config.search_depth=${SOKOBAN_SEARCH_DEPTH} \
  custom_envs.CoordSokoban.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
  custom_envs.CoordSokoban.max_tokens=${MAX_TOKENS} \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS} \
  actor_rollout_ref.rollout.response_length=${MAX_TOKENS} \
  "output.dir='${OUTPUT_DIR}'" \
  "output.filename='${RUN_NAME}.pkl'" \
  "output.append_timestamp=True" \
  "hydra.run.dir='${HYDRA_DIR}'" \
  "hydra.output_subdir=null"
