#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-ragenv2}
conda activate "${CONDA_ENV_NAME}"

PROJECT_ROOT=${PROJECT_ROOT:-"$HOME/agent-budget-control"}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:$PWD/verl"

MODEL_PATH=${MODEL_PATH:-/projects/e32695/Qwen3-32B}
MODEL_TAG=${MODEL_TAG:-qwen3-32b-thinking-instance}
CUDA_VISIBLE_DEVICES_VALUE=${CUDA_VISIBLE_DEVICES_VALUE:-0,1,2,3}
NUM_GPUS=${NUM_GPUS:-4}
TP_SIZE=${TP_SIZE:-4}
DTYPE=${DTYPE:-bfloat16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
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

RUN_NAME=${RUN_NAME:-gpqa_main_qwen3_eval_compliance}
VAL_GROUPS=${VAL_GROUPS:-512}
VAL_START_GROUP_INDEX=${VAL_START_GROUP_INDEX:-0}
GPQA_SPLIT=${GPQA_SPLIT:-train}
MAX_TURN=${MAX_TURN:-1}
MAX_ACTIONS_PER_TURN=${MAX_ACTIONS_PER_TURN:-1}
MAX_TOKENS=${MAX_TOKENS:-2000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
PROMPT_TOKEN_MARGIN=${PROMPT_TOKEN_MARGIN:-512}
RESULT_ROOT=${RESULT_ROOT:-"$PWD/results/budget-estimation-benchmark"}
OUTPUT_DIR=${OUTPUT_DIR:-"$RESULT_ROOT/gpqa-main-compliance-token-qwen3-32b-thinking-512-test"}
HYDRA_DIR=${HYDRA_DIR:-"$OUTPUT_DIR/hydra/$RUN_NAME"}
DRY_RUN=${DRY_RUN:-0}

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

mkdir -p "$OUTPUT_DIR" "$HYDRA_DIR"

cmd=(
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
  "agent_proxy.enable_think=True"
  "agent_proxy.qwen_enable_thinking=True"
  "agent_proxy.eval-estimation-single=False"
  "agent_proxy.eval-estimation-multi=False"
  "agent_proxy.eval_compliance_token=True"
  "agent_proxy.eval_compliance_token_scope=[100,200,300,400,500]"
  "agent_proxy.max_turn=${MAX_TURN}"
  "agent_proxy.max_actions_per_turn=${MAX_ACTIONS_PER_TURN}"
  "es_manager.val.env_groups=${VAL_GROUPS}"
  "es_manager.val.group_size=1"
  "es_manager.val.start_group_index=${VAL_START_GROUP_INDEX}"
  "es_manager.val.env_configs.tags=[GPQAMain]"
  "es_manager.val.env_configs.n_groups=[${VAL_GROUPS}]"
  "custom_envs.GPQAMain.env_instruction='You are answering a multiple-choice science question. Think briefly, then inside <answer> output exactly one letter: A, B, C, or D. Do not include any explanation inside <answer>.'"
  "++custom_envs.GPQAMain.env_config.split=\"${GPQA_SPLIT}\""
  "custom_envs.GPQAMain.max_tokens=${MAX_TOKENS}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS}"
  "actor_rollout_ref.rollout.response_length=${MAX_TOKENS}"
  "++model_config.prompt_token_margin=${PROMPT_TOKEN_MARGIN}"
  "trainer.local_log_dir=${OUTPUT_DIR}"
  "trainer.experiment_name=${RUN_NAME}"
  "output.dir=${OUTPUT_DIR}"
  "output.filename=${RUN_NAME}.pkl"
  "output.append_timestamp=True"
  "hydra.run.dir=${HYDRA_DIR}"
  "hydra.output_subdir=null"
)

echo "==> Running GPQA compliance with local Qwen3-32B thinking model at ${MODEL_PATH}"
if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
