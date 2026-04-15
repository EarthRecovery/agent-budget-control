#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-ragenv2}
conda activate "${CONDA_ENV_NAME}"

PROJECT_ROOT=${PROJECT_ROOT:-"$HOME/RAGEN-v2"}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:$PWD/verl"

MODEL_PATH=${MODEL_PATH:-/projects/e32695/Qwen3-32B}
INSTANT_MODEL_TAG=${INSTANT_MODEL_TAG:-qwen-32b-instant}
THINKING_MODEL_TAG=${THINKING_MODEL_TAG:-qwen-32b-thinking}
CUDA_VISIBLE_DEVICES_VALUE=${CUDA_VISIBLE_DEVICES_VALUE:-0,1,2,3}
NUM_GPUS=${NUM_GPUS:-4}
TP_SIZE=${TP_SIZE:-4}
DTYPE=${DTYPE:-bfloat16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.25}
RESULT_ROOT=${RESULT_ROOT:-"$PWD/results/budget-estimation-benchmark"}
DRY_RUN=${DRY_RUN:-0}
VLLM_USE_V1=${VLLM_USE_V1:-1}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-False}
ROLLOUT_ENABLE_PREFIX_CACHING=${ROLLOUT_ENABLE_PREFIX_CACHING:-False}
ROLLOUT_DISABLE_LOG_STATS=${ROLLOUT_DISABLE_LOG_STATS:-True}
ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE=${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-True}
ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE=${ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE:-True}
ROLLOUT_SKIP_TOKENIZER_INIT=${ROLLOUT_SKIP_TOKENIZER_INIT:-False}
ROLLOUT_TRUST_REMOTE_CODE=${ROLLOUT_TRUST_REMOTE_CODE:-False}
ROLLOUT_DO_SAMPLE=${ROLLOUT_DO_SAMPLE:-False}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
ROLLOUT_LOGPROBS=${ROLLOUT_LOGPROBS:-null}

VAL_GROUPS=${VAL_GROUPS:-512}
VAL_START_GROUP_INDEX=${VAL_START_GROUP_INDEX:-0}

SOKOBAN_VAL_GROUPS=${SOKOBAN_VAL_GROUPS:-${VAL_GROUPS}}
SOKOBAN_VAL_START_GROUP_INDEX=${SOKOBAN_VAL_START_GROUP_INDEX:-${VAL_START_GROUP_INDEX}}
SOKOBAN_MAX_TURN=${SOKOBAN_MAX_TURN:-10}
SOKOBAN_MAX_ACTIONS_PER_TURN=${SOKOBAN_MAX_ACTIONS_PER_TURN:-1}
SOKOBAN_MAX_TOKENS=${SOKOBAN_MAX_TOKENS:-1024}
SOKOBAN_MAX_MODEL_LEN=${SOKOBAN_MAX_MODEL_LEN:-4096}
SOKOBAN_MAX_BATCHED_TOKENS=${SOKOBAN_MAX_BATCHED_TOKENS:-4096}

WEBSHOP_VAL_GROUPS=${WEBSHOP_VAL_GROUPS:-${VAL_GROUPS}}
WEBSHOP_VAL_START_GROUP_INDEX=${WEBSHOP_VAL_START_GROUP_INDEX:-${VAL_START_GROUP_INDEX}}
WEBSHOP_TAG=${WEBSHOP_TAG:-WebShop}
WEBSHOP_DATASET=${WEBSHOP_DATASET:-small}
WEBSHOP_LIMIT_GOALS=${WEBSHOP_LIMIT_GOALS:--1}
WEBSHOP_MAX_TURN=${WEBSHOP_MAX_TURN:-9}
WEBSHOP_MAX_ACTIONS_PER_TURN=${WEBSHOP_MAX_ACTIONS_PER_TURN:-1}
WEBSHOP_MAX_ACTIONS_PER_TRAJ=${WEBSHOP_MAX_ACTIONS_PER_TRAJ:-9}
WEBSHOP_MAX_TOKENS=${WEBSHOP_MAX_TOKENS:-2048}
WEBSHOP_MAX_MODEL_LEN=${WEBSHOP_MAX_MODEL_LEN:-15000}
WEBSHOP_MAX_BATCHED_TOKENS=${WEBSHOP_MAX_BATCHED_TOKENS:-15000}

FROZEN_LAKE_VAL_GROUPS=${FROZEN_LAKE_VAL_GROUPS:-${VAL_GROUPS}}
FROZEN_LAKE_VAL_START_GROUP_INDEX=${FROZEN_LAKE_VAL_START_GROUP_INDEX:-${VAL_START_GROUP_INDEX}}
FROZEN_LAKE_SUCCESS_RATE=${FROZEN_LAKE_SUCCESS_RATE:-1.0}
FROZEN_LAKE_MAX_TURN=${FROZEN_LAKE_MAX_TURN:-10}
FROZEN_LAKE_MAX_ACTIONS_PER_TURN=${FROZEN_LAKE_MAX_ACTIONS_PER_TURN:-1}
FROZEN_LAKE_MAX_ACTIONS_PER_TRAJ=${FROZEN_LAKE_MAX_ACTIONS_PER_TRAJ:-10}
FROZEN_LAKE_MAX_TOKENS=${FROZEN_LAKE_MAX_TOKENS:-1024}
FROZEN_LAKE_MAX_MODEL_LEN=${FROZEN_LAKE_MAX_MODEL_LEN:-8192}
FROZEN_LAKE_MAX_BATCHED_TOKENS=${FROZEN_LAKE_MAX_BATCHED_TOKENS:-8192}

DEEPCODER_VAL_GROUPS=${DEEPCODER_VAL_GROUPS:-${VAL_GROUPS}}
DEEPCODER_VAL_START_GROUP_INDEX=${DEEPCODER_VAL_START_GROUP_INDEX:-${VAL_START_GROUP_INDEX}}
DEEPCODER_MAX_TURN=${DEEPCODER_MAX_TURN:-1}
DEEPCODER_MAX_ACTIONS_PER_TURN=${DEEPCODER_MAX_ACTIONS_PER_TURN:-1}
DEEPCODER_MAX_TOKENS=${DEEPCODER_MAX_TOKENS:-1024}
DEEPCODER_MAX_MODEL_LEN=${DEEPCODER_MAX_MODEL_LEN:-4096}
DEEPCODER_MAX_BATCHED_TOKENS=${DEEPCODER_MAX_BATCHED_TOKENS:-4096}

SEARCHR1_VAL_GROUPS=${SEARCHR1_VAL_GROUPS:-${VAL_GROUPS}}
SEARCHR1_VAL_START_GROUP_INDEX=${SEARCHR1_VAL_START_GROUP_INDEX:-${VAL_START_GROUP_INDEX}}
SEARCHR1_TAG=${SEARCHR1_TAG:-SearchQA}
SEARCHR1_DATA_ROOT=${SEARCHR1_DATA_ROOT:-/projects/bflz/searchr1_data}
SEARCHR1_DATA_PATH=${SEARCHR1_DATA_PATH:-${SEARCHR1_DATA_ROOT}/data/search/train.parquet}
SEARCHR1_MOCK_MODE=${SEARCHR1_MOCK_MODE:-False}
SEARCHR1_RETRIEVAL_SERVER_URL=${SEARCHR1_RETRIEVAL_SERVER_URL:-http://127.0.0.1:8000}
SEARCHR1_MAX_TURN=${SEARCHR1_MAX_TURN:-5}
SEARCHR1_MAX_ACTIONS_PER_TURN=${SEARCHR1_MAX_ACTIONS_PER_TURN:-1}
SEARCHR1_MAX_ACTIONS_PER_TRAJ=${SEARCHR1_MAX_ACTIONS_PER_TRAJ:-10}
SEARCHR1_MAX_TOKENS=${SEARCHR1_MAX_TOKENS:-2048}
SEARCHR1_MAX_MODEL_LEN=${SEARCHR1_MAX_MODEL_LEN:-5000}
SEARCHR1_MAX_BATCHED_TOKENS=${SEARCHR1_MAX_BATCHED_TOKENS:-5000}

GPQA_VAL_GROUPS=${GPQA_VAL_GROUPS:-${VAL_GROUPS}}
GPQA_VAL_START_GROUP_INDEX=${GPQA_VAL_START_GROUP_INDEX:-${VAL_START_GROUP_INDEX}}
GPQA_SPLIT=${GPQA_SPLIT:-train}
GPQA_MAX_TURN=${GPQA_MAX_TURN:-1}
GPQA_MAX_ACTIONS_PER_TURN=${GPQA_MAX_ACTIONS_PER_TURN:-1}
GPQA_MAX_TOKENS=${GPQA_MAX_TOKENS:-1024}
GPQA_MAX_MODEL_LEN=${GPQA_MAX_MODEL_LEN:-4096}
GPQA_MAX_BATCHED_TOKENS=${GPQA_MAX_BATCHED_TOKENS:-4096}

find_nvcc() {
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    printf '%s\n' "${CUDA_HOME}/bin/nvcc"
    return 0
  fi
  command -v nvcc 2>/dev/null || true
}

configure_vllm_sampler() {
  if [[ -n "${VLLM_USE_FLASHINFER_SAMPLER:-}" ]]; then
    echo "==> Using preset VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}"
    return
  fi

  local nvcc_path
  nvcc_path=$(find_nvcc)
  if [[ -n "$nvcc_path" ]]; then
    echo "==> Detected nvcc at ${nvcc_path}"
    return
  fi

  export VLLM_USE_FLASHINFER_SAMPLER=0
  echo "==> nvcc not found; exporting VLLM_USE_FLASHINFER_SAMPLER=0 to avoid FlashInfer JIT"
}

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "MODEL_PATH does not look like a merged HF model directory: ${MODEL_PATH}" >&2
  exit 1
fi

export VLLM_USE_V1

configure_vllm_sampler

run_eval() {
  local env_name=$1
  local mode_name=$2
  local enable_think=$3
  local qwen_enable_thinking=$4
  local model_tag=$5
  shift 5

  local run_name="${env_name}_${model_tag}_rollout"
  local output_dir="${RESULT_ROOT}/${env_name}-${model_tag}"
  local hydra_dir="${output_dir}/hydra/${run_name}"

  mkdir -p "$output_dir" "$hydra_dir"

  local -a cmd=(
    python -m ragen.llm_agent.agent_proxy --config-name eval
    "model_path=${MODEL_PATH}"
    "system.CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES_VALUE}\""
    "trainer.n_gpus_per_node=${NUM_GPUS}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"
    "actor_rollout_ref.rollout.dtype=${DTYPE}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}"
    "actor_rollout_ref.rollout.enable_sleep_mode=${ROLLOUT_ENABLE_SLEEP_MODE}"
    "actor_rollout_ref.rollout.enable_prefix_caching=${ROLLOUT_ENABLE_PREFIX_CACHING}"
    "actor_rollout_ref.rollout.disable_log_stats=${ROLLOUT_DISABLE_LOG_STATS}"
    "actor_rollout_ref.rollout.disable_custom_all_reduce=${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE}"
    "actor_rollout_ref.rollout.disable_mm_preprocessor_cache=${ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE}"
    "actor_rollout_ref.rollout.skip_tokenizer_init=${ROLLOUT_SKIP_TOKENIZER_INIT}"
    "actor_rollout_ref.rollout.trust_remote_code=${ROLLOUT_TRUST_REMOTE_CODE}"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=${ROLLOUT_DO_SAMPLE}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${ROLLOUT_TEMPERATURE}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${ROLLOUT_TOP_P}"
    "actor_rollout_ref.rollout.val_kwargs.top_k=${ROLLOUT_TOP_K}"
    "actor_rollout_ref.rollout.val_kwargs.logprobs=${ROLLOUT_LOGPROBS}"
    "agent_proxy.enable_think=${enable_think}"
    "agent_proxy.qwen_enable_thinking=${qwen_enable_thinking}"
    "trainer.local_log_dir=${output_dir}"
    "trainer.experiment_name=${run_name}"
    "output.dir=${output_dir}"
    "output.filename=${run_name}.pkl"
    "output.append_timestamp=True"
    "hydra.run.dir=${hydra_dir}"
    "hydra.output_subdir=null"
  )

  cmd+=("$@")

  echo "==> Running ${env_name} in ${mode_name} mode with local model ${MODEL_PATH} on GPUs ${CUDA_VISIBLE_DEVICES_VALUE}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}"
}

run_sokoban() {
  local mode_name=$1
  local enable_think=$2
  local qwen_enable_thinking=$3
  local model_tag=$4
  run_eval "sokoban" "${mode_name}" "${enable_think}" "${qwen_enable_thinking}" "${model_tag}" \
    "agent_proxy.eval-estimation-single=False" \
    "agent_proxy.eval-estimation-multi=True" \
    "agent_proxy.max_turn=${SOKOBAN_MAX_TURN}" \
    "agent_proxy.max_actions_per_turn=${SOKOBAN_MAX_ACTIONS_PER_TURN}" \
    "es_manager.val.env_groups=${SOKOBAN_VAL_GROUPS}" \
    "es_manager.val.group_size=1" \
    "es_manager.val.start_group_index=${SOKOBAN_VAL_START_GROUP_INDEX}" \
    "es_manager.val.env_configs.tags=[CoordSokoban]" \
    "es_manager.val.env_configs.n_groups=[${SOKOBAN_VAL_GROUPS}]" \
    "custom_envs.CoordSokoban.env_instruction='You are solving the Sokoban puzzle. Push all boxes to targets. You are given the grid and zero-indexed coordinates of the player, boxes, and targets. You can push but not pull boxes, and cannot push a box through a wall.'" \
    "custom_envs.CoordSokoban.max_tokens=${SOKOBAN_MAX_TOKENS}" \
    "actor_rollout_ref.rollout.max_model_len=${SOKOBAN_MAX_MODEL_LEN}" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=${SOKOBAN_MAX_BATCHED_TOKENS}" \
    "actor_rollout_ref.rollout.response_length=${SOKOBAN_MAX_TOKENS}"
}

run_webshop() {
  local mode_name=$1
  local enable_think=$2
  local qwen_enable_thinking=$3
  local model_tag=$4
  run_eval "webshop" "${mode_name}" "${enable_think}" "${qwen_enable_thinking}" "${model_tag}" \
    "agent_proxy.eval-estimation-single=False" \
    "agent_proxy.eval-estimation-multi=True" \
    "agent_proxy.max_turn=${WEBSHOP_MAX_TURN}" \
    "agent_proxy.max_actions_per_turn=${WEBSHOP_MAX_ACTIONS_PER_TURN}" \
    "es_manager.val.env_groups=${WEBSHOP_VAL_GROUPS}" \
    "es_manager.val.group_size=1" \
    "es_manager.val.start_group_index=${WEBSHOP_VAL_START_GROUP_INDEX}" \
    "es_manager.val.env_configs.tags=[${WEBSHOP_TAG}]" \
    "es_manager.val.env_configs.n_groups=[${WEBSHOP_VAL_GROUPS}]" \
    "custom_envs.${WEBSHOP_TAG}.env_instruction='You are browsing an online shop. Based on the user instruction, choose exactly one valid next action from the provided action list. Use search[<keywords>] when on the search page, click[<item>] to inspect a result or option, and click[buy now] only after selecting the needed product options. Inside <answer>, output exactly one action string only.'" \
    "++custom_envs.${WEBSHOP_TAG}.env_config.dataset=\"${WEBSHOP_DATASET}\"" \
    "++custom_envs.${WEBSHOP_TAG}.env_config.limit_goals=${WEBSHOP_LIMIT_GOALS}" \
    "custom_envs.${WEBSHOP_TAG}.max_actions_per_traj=${WEBSHOP_MAX_ACTIONS_PER_TRAJ}" \
    "custom_envs.${WEBSHOP_TAG}.max_tokens=${WEBSHOP_MAX_TOKENS}" \
    "actor_rollout_ref.rollout.max_model_len=${WEBSHOP_MAX_MODEL_LEN}" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=${WEBSHOP_MAX_BATCHED_TOKENS}" \
    "actor_rollout_ref.rollout.response_length=${WEBSHOP_MAX_TOKENS}"
}

run_frozen_lake() {
  local mode_name=$1
  local enable_think=$2
  local qwen_enable_thinking=$3
  local model_tag=$4
  run_eval "frozen-lake" "${mode_name}" "${enable_think}" "${qwen_enable_thinking}" "${model_tag}" \
    "agent_proxy.eval-estimation-single=False" \
    "agent_proxy.eval-estimation-multi=True" \
    "agent_proxy.max_turn=${FROZEN_LAKE_MAX_TURN}" \
    "agent_proxy.max_actions_per_turn=${FROZEN_LAKE_MAX_ACTIONS_PER_TURN}" \
    "es_manager.val.env_groups=${FROZEN_LAKE_VAL_GROUPS}" \
    "es_manager.val.group_size=1" \
    "es_manager.val.start_group_index=${FROZEN_LAKE_VAL_START_GROUP_INDEX}" \
    "es_manager.val.env_configs.tags=[CoordFrozenLake]" \
    "es_manager.val.env_configs.n_groups=[${FROZEN_LAKE_VAL_GROUPS}]" \
    "custom_envs.CoordFrozenLake.env_instruction='You are solving the FrozenLake puzzle. The observation includes a symbol grid and zero-indexed coordinates for the start, goal, player, and holes. The ice may be slippery, so the executed move can differ from the intended one. Choose exactly one next move from Left, Down, Right, or Up. Inside <answer>, output that single action only.'" \
    "custom_envs.CoordFrozenLake.env_config.success_rate=${FROZEN_LAKE_SUCCESS_RATE}" \
    "custom_envs.CoordFrozenLake.max_actions_per_traj=${FROZEN_LAKE_MAX_ACTIONS_PER_TRAJ}" \
    "custom_envs.CoordFrozenLake.max_tokens=${FROZEN_LAKE_MAX_TOKENS}" \
    "actor_rollout_ref.rollout.max_model_len=${FROZEN_LAKE_MAX_MODEL_LEN}" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=${FROZEN_LAKE_MAX_BATCHED_TOKENS}" \
    "actor_rollout_ref.rollout.response_length=${FROZEN_LAKE_MAX_TOKENS}"
}

run_deepcoder() {
  local mode_name=$1
  local enable_think=$2
  local qwen_enable_thinking=$3
  local model_tag=$4
  run_eval "deepcoder" "${mode_name}" "${enable_think}" "${qwen_enable_thinking}" "${model_tag}" \
    "agent_proxy.eval-estimation-single=True" \
    "agent_proxy.eval-estimation-multi=False" \
    "agent_proxy.max_turn=${DEEPCODER_MAX_TURN}" \
    "agent_proxy.max_actions_per_turn=${DEEPCODER_MAX_ACTIONS_PER_TURN}" \
    "es_manager.val.env_groups=${DEEPCODER_VAL_GROUPS}" \
    "es_manager.val.group_size=1" \
    "es_manager.val.start_group_index=${DEEPCODER_VAL_START_GROUP_INDEX}" \
    "es_manager.val.env_configs.tags=[DeepCoder]" \
    "es_manager.val.env_configs.n_groups=[${DEEPCODER_VAL_GROUPS}]" \
    "custom_envs.DeepCoder.env_instruction='You are solving a coding task. You are given a problem description and you need to write a function that satisfies the requirements. Inside <answer>, output raw Python code only. Do not use Markdown code fences, triple backticks, backticks, or any explanation.'" \
    "custom_envs.DeepCoder.max_tokens=${DEEPCODER_MAX_TOKENS}" \
    "actor_rollout_ref.rollout.max_model_len=${DEEPCODER_MAX_MODEL_LEN}" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=${DEEPCODER_MAX_BATCHED_TOKENS}" \
    "actor_rollout_ref.rollout.response_length=${DEEPCODER_MAX_TOKENS}"
}

run_search_r1() {
  local mode_name=$1
  local enable_think=$2
  local qwen_enable_thinking=$3
  local model_tag=$4
  run_eval "search-r1" "${mode_name}" "${enable_think}" "${qwen_enable_thinking}" "${model_tag}" \
    "agent_proxy.eval-estimation-single=False" \
    "agent_proxy.eval-estimation-multi=True" \
    "agent_proxy.max_turn=${SEARCHR1_MAX_TURN}" \
    "agent_proxy.max_actions_per_turn=${SEARCHR1_MAX_ACTIONS_PER_TURN}" \
    "es_manager.val.env_groups=${SEARCHR1_VAL_GROUPS}" \
    "es_manager.val.group_size=1" \
    "es_manager.val.start_group_index=${SEARCHR1_VAL_START_GROUP_INDEX}" \
    "es_manager.val.env_configs.tags=[${SEARCHR1_TAG}]" \
    "es_manager.val.env_configs.n_groups=[${SEARCHR1_VAL_GROUPS}]" \
    "custom_envs.${SEARCHR1_TAG}.env_instruction='You are a search agent answering questions by searching for information. Choose exactly one valid next action. Use search[<query>] to retrieve evidence and finish[<answer>] only when you are ready to submit the final answer. Inside <answer>, output exactly one action string only.'" \
    "++custom_envs.${SEARCHR1_TAG}.env_config.train_path=\"${SEARCHR1_DATA_PATH}\"" \
    "++custom_envs.${SEARCHR1_TAG}.env_config.mock_mode=${SEARCHR1_MOCK_MODE}" \
    "++custom_envs.${SEARCHR1_TAG}.env_config.retrieval_server_url=\"${SEARCHR1_RETRIEVAL_SERVER_URL}\"" \
    "custom_envs.${SEARCHR1_TAG}.max_actions_per_traj=${SEARCHR1_MAX_ACTIONS_PER_TRAJ}" \
    "custom_envs.${SEARCHR1_TAG}.max_tokens=${SEARCHR1_MAX_TOKENS}" \
    "actor_rollout_ref.rollout.max_model_len=${SEARCHR1_MAX_MODEL_LEN}" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=${SEARCHR1_MAX_BATCHED_TOKENS}" \
    "actor_rollout_ref.rollout.response_length=${SEARCHR1_MAX_TOKENS}"
}

run_gpqa_main() {
  local mode_name=$1
  local enable_think=$2
  local qwen_enable_thinking=$3
  local model_tag=$4
  run_eval "gpqa-main" "${mode_name}" "${enable_think}" "${qwen_enable_thinking}" "${model_tag}" \
    "agent_proxy.eval-estimation-single=True" \
    "agent_proxy.eval-estimation-multi=False" \
    "agent_proxy.max_turn=${GPQA_MAX_TURN}" \
    "agent_proxy.max_actions_per_turn=${GPQA_MAX_ACTIONS_PER_TURN}" \
    "es_manager.val.env_groups=${GPQA_VAL_GROUPS}" \
    "es_manager.val.group_size=1" \
    "es_manager.val.start_group_index=${GPQA_VAL_START_GROUP_INDEX}" \
    "es_manager.val.env_configs.tags=[GPQAMain]" \
    "es_manager.val.env_configs.n_groups=[${GPQA_VAL_GROUPS}]" \
    "custom_envs.GPQAMain.env_instruction='You are answering a multiple-choice science question. Think briefly, then inside <answer> output exactly one letter: A, B, C, or D. Do not include any explanation inside <answer>.'" \
    "++custom_envs.GPQAMain.env_config.split=${GPQA_SPLIT}" \
    "custom_envs.GPQAMain.max_tokens=${GPQA_MAX_TOKENS}" \
    "actor_rollout_ref.rollout.max_model_len=${GPQA_MAX_MODEL_LEN}" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=${GPQA_MAX_BATCHED_TOKENS}" \
    "actor_rollout_ref.rollout.response_length=${GPQA_MAX_TOKENS}"
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/budget-estimation-benchmark/4-qwen3-32b.sh [target...]

Targets:
  instant
  thinking
  sokoban_{instant,thinking}
  webshop_{instant,thinking}
  frozen_lake_{instant,thinking}
  deepcoder_{instant,thinking}
  search_r1_{instant,thinking}
  gpqa_main_{instant,thinking}

Aliases:
  frozen-lake_{instant,thinking}
  searchr1_{instant,thinking}
  search-r1_{instant,thinking}
  gpqa-main_{instant,thinking}

Behavior:
  - No positional args: run 12 jobs in sequence:
    6 envs with qwen-32b-instant, then 6 envs with qwen-32b-thinking.
  - Default local model path: /projects/e32695/Qwen3-32B
  - Default 4x A100: CUDA_VISIBLE_DEVICES_VALUE=0,1,2,3 NUM_GPUS=4 TP_SIZE=4
  - DRY_RUN=1: print commands without executing them.
EOF
}

run_instant_suite() {
  run_sokoban "instant" "True" "False" "${INSTANT_MODEL_TAG}"
  run_webshop "instant" "True" "False" "${INSTANT_MODEL_TAG}"
  run_frozen_lake "instant" "True" "False" "${INSTANT_MODEL_TAG}"
  run_deepcoder "instant" "True" "False" "${INSTANT_MODEL_TAG}"
  run_search_r1 "instant" "True" "False" "${INSTANT_MODEL_TAG}"
  run_gpqa_main "instant" "True" "False" "${INSTANT_MODEL_TAG}"
}

run_thinking_suite() {
  run_sokoban "thinking" "True" "True" "${THINKING_MODEL_TAG}"
  run_webshop "thinking" "True" "True" "${THINKING_MODEL_TAG}"
  run_frozen_lake "thinking" "True" "True" "${THINKING_MODEL_TAG}"
  run_deepcoder "thinking" "True" "True" "${THINKING_MODEL_TAG}"
  run_search_r1 "thinking" "True" "True" "${THINKING_MODEL_TAG}"
  run_gpqa_main "thinking" "True" "True" "${THINKING_MODEL_TAG}"
}

main() {
  local -a targets
  if [[ $# -eq 0 ]]; then
    targets=(
      sokoban_instant
      webshop_instant
      frozen_lake_instant
      deepcoder_instant
      search_r1_instant
      gpqa_main_instant
      sokoban_thinking
      webshop_thinking
      frozen_lake_thinking
      deepcoder_thinking
      search_r1_thinking
      gpqa_main_thinking
    )
  else
    targets=("$@")
  fi

  local target
  for target in "${targets[@]}"; do
    case "$target" in
      instant|all_instant)
        run_instant_suite
        ;;
      thinking|all_thinking)
        run_thinking_suite
        ;;
      sokoban_instant)
        run_sokoban "instant" "True" "False" "${INSTANT_MODEL_TAG}"
        ;;
      webshop_instant)
        run_webshop "instant" "True" "False" "${INSTANT_MODEL_TAG}"
        ;;
      frozen_lake_instant|frozen-lake_instant)
        run_frozen_lake "instant" "True" "False" "${INSTANT_MODEL_TAG}"
        ;;
      deepcoder_instant)
        run_deepcoder "instant" "True" "False" "${INSTANT_MODEL_TAG}"
        ;;
      search_r1_instant|searchr1_instant|search-r1_instant)
        run_search_r1 "instant" "True" "False" "${INSTANT_MODEL_TAG}"
        ;;
      gpqa_main_instant|gpqa-main_instant)
        run_gpqa_main "instant" "True" "False" "${INSTANT_MODEL_TAG}"
        ;;
      sokoban_thinking)
        run_sokoban "thinking" "True" "True" "${THINKING_MODEL_TAG}"
        ;;
      webshop_thinking)
        run_webshop "thinking" "True" "True" "${THINKING_MODEL_TAG}"
        ;;
      frozen_lake_thinking|frozen-lake_thinking)
        run_frozen_lake "thinking" "True" "True" "${THINKING_MODEL_TAG}"
        ;;
      deepcoder_thinking)
        run_deepcoder "thinking" "True" "True" "${THINKING_MODEL_TAG}"
        ;;
      search_r1_thinking|searchr1_thinking|search-r1_thinking)
        run_search_r1 "thinking" "True" "True" "${THINKING_MODEL_TAG}"
        ;;
      gpqa_main_thinking|gpqa-main_thinking)
        run_gpqa_main "thinking" "True" "True" "${THINKING_MODEL_TAG}"
        ;;
      -h|--help|help)
        usage
        return 0
        ;;
      *)
        echo "Unknown target: $target" >&2
        usage >&2
        return 1
        ;;
    esac
  done
}

main "$@"
