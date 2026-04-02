#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ragen

PROJECT_ROOT=${PROJECT_ROOT:-"$HOME/RAGEN-v2"}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:$PWD/verl"

# Default model is OpenAI GPT-5.2 Thinking.
: "${OPENAI_API_KEY:?Please export OPENAI_API_KEY before running this benchmark.}"

RUN_NAME=${RUN_NAME:-webshop_api_eval_estimation}
MODEL_NAME=${MODEL_NAME:-OpenAI-5.2-Thinking}
VAL_GROUPS=${VAL_GROUPS:-512}
VAL_START_GROUP_INDEX=${VAL_START_GROUP_INDEX:-0}
VAL_ROLLOUT_CHUNK_SIZE=${VAL_ROLLOUT_CHUNK_SIZE:-0}
WEBSHOP_TAG=${WEBSHOP_TAG:-WebShop}
WEBSHOP_DATASET=${WEBSHOP_DATASET:-small}
WEBSHOP_LIMIT_GOALS=${WEBSHOP_LIMIT_GOALS:--1}
MAX_TURN=${MAX_TURN:-9}
MAX_ACTIONS_PER_TURN=${MAX_ACTIONS_PER_TURN:-1}
MAX_ACTIONS_PER_TRAJ=${MAX_ACTIONS_PER_TRAJ:-9}
MAX_TOKENS=${MAX_TOKENS:-2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-15000}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-15000}
PROMPT_TOKEN_MARGIN=${PROMPT_TOKEN_MARGIN:-2048}
RESULT_ROOT=${RESULT_ROOT:-"$PWD/results/budget-estimation-benchmark"}
OUTPUT_DIR=${OUTPUT_DIR:-"$RESULT_ROOT/webshop-512-gpt5.2-Thinking"}
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
  "es_manager.val.env_configs.tags=[${WEBSHOP_TAG}]" \
  "es_manager.val.env_configs.n_groups=[${VAL_GROUPS}]" \
  "custom_envs.${WEBSHOP_TAG}.env_instruction='You are browsing an online shop. Based on the user instruction, choose exactly one valid next action from the provided action list. Use search[<keywords>] when on the search page, click[<item>] to inspect a result or option, and click[buy now] only after selecting the needed product options. Inside <answer>, output exactly one action string only.'" \
  ++custom_envs.${WEBSHOP_TAG}.env_config.dataset="${WEBSHOP_DATASET}" \
  ++custom_envs.${WEBSHOP_TAG}.env_config.limit_goals=${WEBSHOP_LIMIT_GOALS} \
  custom_envs.${WEBSHOP_TAG}.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
  custom_envs.${WEBSHOP_TAG}.max_tokens=${MAX_TOKENS} \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS} \
  actor_rollout_ref.rollout.response_length=${MAX_TOKENS} \
  model_config.prompt_token_margin=${PROMPT_TOKEN_MARGIN} \
  output.dir="${OUTPUT_DIR}" \
  output.filename="${RUN_NAME}.pkl" \
  output.append_timestamp=True \
  hydra.run.dir="${HYDRA_DIR}" \
  hydra.output_subdir=null
