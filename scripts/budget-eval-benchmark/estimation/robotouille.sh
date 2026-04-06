#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ragenv2

PROJECT_ROOT=${PROJECT_ROOT:-"$HOME/agent-budget-control"}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:$PWD/verl"
# Default model is OpenAI GPT-4o.
: "${OPENAI_API_KEY:?Please export OPENAI_API_KEY before running this benchmark.}"

RUN_NAME=${RUN_NAME:-robotouille_api_eval_estimation}
MODEL_NAME=${MODEL_NAME:-OpenAI-5.2-Instant}
VAL_GROUPS=${VAL_GROUPS:-1}
VAL_START_GROUP_INDEX=${VAL_START_GROUP_INDEX:-0} # 0-based: 128 means start from the 129th validation sample
VAL_ROLLOUT_CHUNK_SIZE=${VAL_ROLLOUT_CHUNK_SIZE:-0}
ENV_NAME=${ENV_NAME:-synchronous/5_double_cheeseburger}
MAX_TURN=${MAX_TURN:-15}
MAX_ACTIONS_PER_TURN=${MAX_ACTIONS_PER_TURN:-5}
MAX_ACTIONS_PER_TRAJ=${MAX_ACTIONS_PER_TRAJ:-30}
ENV_MAX_STEPS=${ENV_MAX_STEPS:-100}
MAX_TOKENS=${MAX_TOKENS:-768}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
PROMPT_TOKEN_MARGIN=${PROMPT_TOKEN_MARGIN:-1024}
CONTEXT_WINDOW_MODE=${CONTEXT_WINDOW_MODE:-limited_multi_turn} # full | single_turn | limited_multi_turn
MAX_CONTEXT_WINDOW=${MAX_CONTEXT_WINDOW:-3} # -1 keeps full history, 1 keeps only current turn
MAX_ACTION_POINTS=${MAX_ACTION_POINTS:-30}
API_CONNECT_TIMEOUT_SECONDS=${API_CONNECT_TIMEOUT_SECONDS:-10}
RESULT_ROOT=${RESULT_ROOT:-"$PWD/results/budget-estimation-benchmark"}
OUTPUT_DIR=${OUTPUT_DIR:-"$RESULT_ROOT/robotouille-1-gpt5.2-Instant-eval-estimation-test"}
HYDRA_DIR=${HYDRA_DIR:-"$OUTPUT_DIR/hydra/$RUN_NAME"}

mkdir -p "$OUTPUT_DIR" "$HYDRA_DIR"

python -m ragen.eval_api --config-name evaluate_api_llm \
  model_config.model_name="${MODEL_NAME}" \
  agent_proxy.enable_think=True \
  "agent_proxy.eval-estimation-single=False" \
  "agent_proxy.eval-estimation-multi=False" \
  "agent_proxy.eval-estimation-toolcall=True" \
  agent_proxy.context_window_mode=${CONTEXT_WINDOW_MODE} \
  agent_proxy.max_context_window=${MAX_CONTEXT_WINDOW} \
  agent_proxy.max_turn=${MAX_TURN} \
  agent_proxy.max_actions_per_turn=${MAX_ACTIONS_PER_TURN} \
  "es_manager.train.env_configs.tags=[Robotouille]" \
  "es_manager.train.env_configs.n_groups=[0]" \
  es_manager.val.env_groups=${VAL_GROUPS} \
  es_manager.val.group_size=1 \
  es_manager.val.start_group_index=${VAL_START_GROUP_INDEX} \
  es_manager.val.rollout_chunk_size=${VAL_ROLLOUT_CHUNK_SIZE} \
  "es_manager.val.env_configs.tags=[Robotouille]" \
  "es_manager.val.env_configs.n_groups=[${VAL_GROUPS}]" \
  "custom_envs.Robotouille.env_instruction='You are controlling a kitchen robot. Choose between 1 and ${MAX_ACTIONS_PER_TURN} actions from the provided Valid Actions list for this turn. Think about the next state changes, then inside <answer> output only the exact action string or strings. If you choose multiple actions, separate them with ||, for example: action1 || action2. Do not output more than ${MAX_ACTIONS_PER_TURN} actions or any explanation inside <answer>.'" \
  ++custom_envs.Robotouille.env_config.env_name="${ENV_NAME}" \
  ++custom_envs.Robotouille.env_config.max_steps=${ENV_MAX_STEPS} \
  ++custom_envs.Robotouille.env_config.enable_action_budget=True \
  ++custom_envs.Robotouille.env_config.max_action_points=${MAX_ACTION_POINTS} \
  custom_envs.Robotouille.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS} \
  model_config.api_connect_timeout_seconds=${API_CONNECT_TIMEOUT_SECONDS} \
  actor_rollout_ref.rollout.response_length=${MAX_TOKENS} \
  model_config.prompt_token_margin=${PROMPT_TOKEN_MARGIN} \
  output.dir="${OUTPUT_DIR}" \
  output.filename="${RUN_NAME}.pkl" \
  output.append_timestamp=True \
  hydra.run.dir="${HYDRA_DIR}" \
  hydra.output_subdir=null
