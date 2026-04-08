#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ragenv2

PROJECT_ROOT=${PROJECT_ROOT:-"$HOME/agent-budget-control"}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:$PWD/verl"

# Default model is OpenAI GPT-5.2 Thinking.
: "${OPENAI_API_KEY:?Please export OPENAI_API_KEY before running this benchmark.}"

RUN_NAME=${RUN_NAME:-frozen_lake_api_eval_estimation}
MODEL_NAME=${MODEL_NAME:-OpenAI-5.2-Thinking}
VAL_GROUPS=${VAL_GROUPS:-1}
VAL_START_GROUP_INDEX=${VAL_START_GROUP_INDEX:-0}
VAL_ROLLOUT_CHUNK_SIZE=${VAL_ROLLOUT_CHUNK_SIZE:-0}
SUCCESS_RATE=${SUCCESS_RATE:-1.0}
MAX_TURN=${MAX_TURN:-10}
MAX_ACTIONS_PER_TURN=${MAX_ACTIONS_PER_TURN:-1}
MAX_ACTIONS_PER_TRAJ=${MAX_ACTIONS_PER_TRAJ:-10}
MAX_TOKENS=${MAX_TOKENS:-1024}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
PROMPT_TOKEN_MARGIN=${PROMPT_TOKEN_MARGIN:-512}
RESULT_ROOT=${RESULT_ROOT:-"$PWD/results/budget-estimation-benchmark"}
OUTPUT_DIR=${OUTPUT_DIR:-"$RESULT_ROOT/frozenlake-compliance-turn-gpt5.2-thinking-1-test"}
HYDRA_DIR=${HYDRA_DIR:-"$OUTPUT_DIR/hydra/$RUN_NAME"}

mkdir -p "$OUTPUT_DIR" "$HYDRA_DIR"

python -m ragen.eval_api --config-name evaluate_api_llm \
  model_config.model_name="${MODEL_NAME}" \
  agent_proxy.enable_think=True \
  "agent_proxy.eval-estimation-single=False" \
  "agent_proxy.eval-estimation-multi=False" \
  "agent_proxy.eval_compliance_turn=True" \
  "agent_proxy.eval_compliance_turn_scope=[2,4,6,8,10]" \
  agent_proxy.max_turn=${MAX_TURN} \
  agent_proxy.max_actions_per_turn=${MAX_ACTIONS_PER_TURN} \
  es_manager.val.env_groups=${VAL_GROUPS} \
  es_manager.val.group_size=1 \
  es_manager.val.start_group_index=${VAL_START_GROUP_INDEX} \
  es_manager.val.rollout_chunk_size=${VAL_ROLLOUT_CHUNK_SIZE} \
  "es_manager.val.env_configs.tags=[CoordFrozenLake]" \
  "es_manager.val.env_configs.n_groups=[${VAL_GROUPS}]" \
  "custom_envs.CoordFrozenLake.env_instruction='You are solving the FrozenLake puzzle. The observation includes a symbol grid and zero-indexed coordinates for the start, goal, player, and holes. The ice may be slippery, so the executed move can differ from the intended one. Choose exactly one next move from Left, Down, Right, or Up. Inside <answer>, output that single action only.'" \
  custom_envs.CoordFrozenLake.env_config.success_rate=${SUCCESS_RATE} \
  custom_envs.CoordFrozenLake.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ} \
  custom_envs.CoordFrozenLake.max_tokens=${MAX_TOKENS} \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS} \
  actor_rollout_ref.rollout.response_length=${MAX_TOKENS} \
  model_config.prompt_token_margin=${PROMPT_TOKEN_MARGIN} \
  "output.dir='${OUTPUT_DIR}'" \
  "output.filename='${RUN_NAME}.pkl'" \
  "output.append_timestamp=True" \
  "hydra.run.dir='${HYDRA_DIR}'" \
  "hydra.output_subdir=null"
