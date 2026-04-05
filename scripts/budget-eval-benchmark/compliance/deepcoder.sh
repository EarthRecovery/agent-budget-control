#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ragenv2

PROJECT_ROOT=${PROJECT_ROOT:-"$HOME/agent-budget-control"}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:$PWD/verl"

# Default model is OpenAI GPT-4o.
: "${OPENAI_API_KEY:?Please export OPENAI_API_KEY before running this benchmark.}"

RUN_NAME=${RUN_NAME:-deepcoder_api_eval_estimation}
MODEL_NAME=${MODEL_NAME:-OpenAI-5.2-Instant}
VAL_GROUPS=${VAL_GROUPS:-1}
VAL_START_GROUP_INDEX=${VAL_START_GROUP_INDEX:-0} # 0-based: 128 means start from the 129th validation sample
VAL_ROLLOUT_CHUNK_SIZE=${VAL_ROLLOUT_CHUNK_SIZE:-0}
MAX_TURN=${MAX_TURN:-1}
MAX_ACTIONS_PER_TURN=${MAX_ACTIONS_PER_TURN:-1}
MAX_TOKENS=${MAX_TOKENS:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-15000}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-12000}
PROMPT_TOKEN_MARGIN=${PROMPT_TOKEN_MARGIN:-512}
RESULT_ROOT=${RESULT_ROOT:-"$PWD/results/budget-estimation-benchmark"}
OUTPUT_DIR=${OUTPUT_DIR:-"$RESULT_ROOT/deepcoder-compliance-token-gpt5.2-instant-1-test"}
HYDRA_DIR=${HYDRA_DIR:-"$OUTPUT_DIR/hydra/$RUN_NAME"}

mkdir -p "$OUTPUT_DIR" "$HYDRA_DIR"

python -m ragen.eval_api --config-name evaluate_api_llm \
  model_config.model_name="${MODEL_NAME}" \
  agent_proxy.enable_think=True \
  "agent_proxy.eval-estimation-single=False" \
  "agent_proxy.eval-estimation-multi=False" \
  "agent_proxy.eval_compliance_token=True" \
  "agent_proxy.eval_compliance_token_scope=[400,800,1200,1600,2000]" \
  agent_proxy.max_turn=${MAX_TURN} \
  agent_proxy.max_actions_per_turn=${MAX_ACTIONS_PER_TURN} \
  es_manager.val.env_groups=${VAL_GROUPS} \
  es_manager.val.group_size=1 \
  es_manager.val.start_group_index=${VAL_START_GROUP_INDEX} \
  es_manager.val.rollout_chunk_size=${VAL_ROLLOUT_CHUNK_SIZE} \
  "es_manager.val.env_configs.tags=[DeepCoder]" \
  "es_manager.val.env_configs.n_groups=[${VAL_GROUPS}]" \
  "custom_envs.DeepCoder.env_instruction='You are solving a coding task. You are given a problem description and you need to write a function that satisfies the requirements. Inside <answer>, output raw Python code only. Do not use Markdown code fences, triple backticks, backticks, or any explanation.'" \
  custom_envs.DeepCoder.max_tokens=${MAX_TOKENS} \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS} \
  actor_rollout_ref.rollout.response_length=${MAX_TOKENS} \
  model_config.prompt_token_margin=${PROMPT_TOKEN_MARGIN} \
  output.dir="${OUTPUT_DIR}" \
  output.filename="${RUN_NAME}.pkl" \
  output.append_timestamp=True \
  hydra.run.dir="${HYDRA_DIR}" \
  hydra.output_subdir=null
