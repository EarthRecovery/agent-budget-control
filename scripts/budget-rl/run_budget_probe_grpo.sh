#!/bin/bash
set -euo pipefail
set -x

# Budget Probe GRPO Training Script
# Model: Qwen2.5-7B-Instruct
# Task: Budget estimation for Sokoban rollouts
#
# Prerequisites:
#   1. Prepare data:
#      python prepare_budget_probe.py \
#          --input qwen-2.5-7b-sokoban-128.jsonl \
#          --output-dir ./budget_probe_data \
#          --max-tokens 32768 \
#          --probe-every-n 1 \
#          --margin 0.1
#
#   2. Reward function: budget_probe_reward.py (in this directory)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/budget_probe_data_qwen25}"
REWARD_FN="${SCRIPT_DIR}/budget_probe_reward.py"

if [ "${SKIP_ENV_ACTIVATE:-0}" != "1" ]; then
  if [ -n "${VENV_PATH:-}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${VENV_PATH}/bin/activate"
  elif [ -n "${CONDA_ENV_NAME:-ragenv2}" ]; then
    if command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)"
      conda activate "${CONDA_ENV_NAME:-ragenv2}"
    elif [ -n "${CONDA_BASE:-}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1090
      source "${CONDA_BASE}/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV_NAME:-ragenv2}"
    fi
  fi
fi
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"

# H200 is SM 9.0 (Hopper). Required for megatron-core JIT compilation at import time.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
# vLLM's CuMemAllocator memory pool is incompatible with PyTorch expandable
# segments, so keep this unset for RL rollout/training.
unset PYTORCH_CUDA_ALLOC_CONF

# Configurable parameters
NGPUS=${NGPUS:-8}
NNODES=${NNODES:-1}
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
LR=${LR:-5e-7}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
ROLLOUT_N=${ROLLOUT_N:-16}
TP_SIZE=${TP_SIZE:-4}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.05}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
RL_MODEL_DTYPE=${RL_MODEL_DTYPE:-bfloat16}
RL_ACTOR_PARAM_OFFLOAD=${RL_ACTOR_PARAM_OFFLOAD:-False}
RL_ACTOR_OPTIMIZER_OFFLOAD=${RL_ACTOR_OPTIMIZER_OFFLOAD:-True}
RL_REF_PARAM_OFFLOAD=${RL_REF_PARAM_OFFLOAD:-True}
PROJECT_NAME=${PROJECT_NAME:-budget_probe_grpo}
TRAIN_FILES="${DATA_DIR}/rl/train.parquet"
VAL_FILES="${DATA_DIR}/rl/test.parquet"

if [ "${BUDGET_RL_DISABLE_LOG:-0}" != "1" ] && [ "${BUDGET_RL_LOG_STARTED:-0}" != "1" ]; then
    BUDGET_RL_LOG_FILE="${BUDGET_RL_LOG_FILE:-${DATA_DIR}/grpo_run.log}"
    mkdir -p "$(dirname "$BUDGET_RL_LOG_FILE")"
    if [ "${BUDGET_RL_APPEND_LOG:-0}" != "1" ]; then
        : >"$BUDGET_RL_LOG_FILE"
    fi
    export BUDGET_RL_LOG_FILE
    export BUDGET_RL_LOG_STARTED=1
    exec > >(tee -a "$BUDGET_RL_LOG_FILE") 2>&1
fi

echo "Budget Probe GRPO"
echo "  log file: ${BUDGET_RL_LOG_FILE:-disabled}"
echo "  data dir: ${DATA_DIR}"
echo "  model: ${MODEL}"
echo "  ngpus: ${NGPUS}"
echo "  tp_size: ${TP_SIZE}"
echo "  train_batch_size: ${TRAIN_BATCH_SIZE}"
echo "  rollout_n: ${ROLLOUT_N}"
echo "  lr: ${LR}"
echo "  total_epochs: ${TOTAL_EPOCHS}"
echo "  max_prompt_length: 8192"
echo "  max_response_length: 1024"
echo "  entropy_coeff: 0"
echo "  use_kl_in_reward: False"
echo "  kl_loss_coef: ${KL_LOSS_COEF}"

if [ ! -f "$VAL_FILES" ]; then
    VAL_FILES="$TRAIN_FILES"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=${RL_MODEL_DTYPE} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${RL_ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${RL_ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=${RL_MODEL_DTYPE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${RL_REF_PARAM_OFFLOAD} \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="${REWARD_FN}" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    "trainer.logger=${TRAINER_LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME:-qwen2.5_7b_budget_probe_run5_sftwarmup}" \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.resume_mode=${RESUME_MODE:-disable} \
    trainer.total_epochs=${TOTAL_EPOCHS} "$@"
