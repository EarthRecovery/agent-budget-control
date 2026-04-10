#!/bin/bash
# Robotouille plain RL runner.
# This script disables mixed toolcall/turn/token budget training and is intended
# for single-node launches such as 4xH200 sbatch jobs.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

# The vendored VERL repo is laid out as <repo>/verl/verl, so add <repo>/verl
# to PYTHONPATH unless the environment has already installed verl.
VERL_PYTHON_ROOT="$REPO_ROOT/verl"
if [ -d "$VERL_PYTHON_ROOT/verl" ]; then
    export PYTHONPATH="$VERL_PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

CONFIG="_9_robotouille"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
PROJECT_NAME="mixed-budget-training"
ALGO="PPO"

STEPS=100
SAVE_FREQ=50
MAX_TURN=20
MAX_ACTIONS_PER_TURN=1
MAX_ACTIONS_PER_TRAJ=10
ENV_NAME="synchronous/4_cheeseburger"
ENV_MAX_STEPS=100
CONTEXT_WINDOW_MODE="limited_multi_turn"
MAX_CONTEXT_WINDOW=3
MAX_ACTION_POINTS=25
ALLOCATED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
GPUS="${GPUS:-${ALLOCATED_CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
GPUS_EXPLICITLY_SET=0
GPU_MEMORY_UTILIZATION=0.2
MAX_MODEL_LEN="10000"
MAX_NUM_BATCHED_TOKENS="10000"

NUM_GROUPS=8
GROUP_SIZE=16
VAL_ENV_GROUPS=128
VAL_GROUP_SIZE=1
MICRO_BATCH_SIZE_PER_GPU=1
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
PPO_MINI_BATCH_SIZE=32

RUN_NAME=""
LOGGER="['console','wandb']"
OUTPUT_ROOT="logs/training"
CHECKPOINT_ROOT="/projects/bflz/model_saving/robotouille-origin"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

usage() {
    cat <<'EOF'
Usage: bash scripts/training/robotouille/mixed-budget-training-robotouille-origin.sh [options]

Robotouille plain RL:
  - mixed toolcall budget: disabled
  - mixed turn budget: disabled
  - mixed token budget: disabled
  - fixed action budget: enabled
  - recommended hardware: 4 GPUs (for example 4xH200)

Options:
  --gpus LIST                    Comma-separated GPU ids. Default: Slurm allocation or 0,1,2,3
  --steps N                      Total training steps. Default: 100
  --save-freq N                  Checkpoint save frequency. Default: 50
  --config NAME                  Hydra config name. Default: _9_robotouille
  --model-path PATH              Model path. Default: Qwen/Qwen2.5-7B-Instruct
  --project-name NAME            W&B/Ray project name. Default: mixed-budget-training
  --algo NAME                    PPO or GRPO. Default: PPO
  --env-name NAME                Robotouille env name. Default: synchronous/4_cheeseburger
  --env-max-steps N              Robotouille max steps. Default: 100
  --context-window-mode MODE     agent_proxy.context_window_mode. Default: limited_multi_turn
  --max-context-window N         agent_proxy.max_context_window. Default: 3
  --max-action-points N          Fixed Robotouille action budget. Default: 25
  --max-turn N                   agent_proxy.max_turn. Default: 20
  --max-actions-per-turn N       agent_proxy.max_actions_per_turn. Default: 1
  --max-actions-per-traj N       custom_envs.Robotouille.max_actions_per_traj. Default: 10
  --num-groups N                 Train env groups. Default: 8
  --group-size N                 Train group size. Default: 16
  --val-env-groups N             Validation env groups. Default: 128
  --val-group-size N             Validation group size. Default: 1
  --micro-batch-size N           micro_batch_size_per_gpu. Default: 1
  --log-prob-micro-batch-size N  log_prob_micro_batch_size_per_gpu. Default: 1
  --ppo-mini-batch-size N        ppo_mini_batch_size. Default: 32
  --gpu-memory-utilization V     rollout gpu_memory_utilization. Default: 0.2
  --max-model-len N              Override actor_rollout_ref.rollout.max_model_len
  --max-num-batched-tokens N     Override actor_rollout_ref.rollout.max_num_batched_tokens
  --checkpoint-root PATH         Checkpoint root. Default: /projects/bflz/model_saving/robotouille-origin
  --run-name NAME                Explicit experiment name
  -h, --help                     Show this help
EOF
}

normalize_algo() {
    local value
    value=$(echo "$1" | tr '[:lower:]' '[:upper:]')
    case "$value" in
        PPO|GRPO)
            echo "$value"
            ;;
        *)
            echo "Error: unsupported algo '$1'. Use PPO or GRPO." >&2
            exit 1
            ;;
    esac
}

model_tag() {
    local tag
    tag=$(basename "$MODEL_PATH")
    tag=${tag//\//-}
    echo "$tag"
}

build_run_name() {
    echo "robotouille-origin-${ALGO}-${GPUS_PER_EXP}gpu-$(model_tag)-${NUM_GROUPS}x${GROUP_SIZE}-turn${MAX_TURN}-budget${MAX_ACTION_POINTS}-origin"
}

get_algo_overrides() {
    case "$1" in
        PPO)
            echo \
                "algorithm.adv_estimator=gae" \
                "actor_rollout_ref.actor.loss_agg_mode=token-mean"
            ;;
        GRPO)
            echo \
                "algorithm.adv_estimator=grpo" \
                "algorithm.norm_adv_by_std_in_grpo=True" \
                "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean"
            ;;
    esac
}

query_gpu_name() {
    local gpu_id="$1"
    local gpu_name=""
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$gpu_id" 2>/dev/null || true)
    gpu_name=$(printf '%s\n' "$gpu_name" | head -n 1 | tr -d '\r')
    case "$gpu_name" in
        NVIDIA-SMI\ has\ failed*)
            gpu_name=""
            ;;
    esac
    printf '%s\n' "$gpu_name"
}

detect_gpu_names() {
    DETECTED_GPU_NAMES=()
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return
    fi

    local gpu_ids=()
    local gpu_id gpu_name
    IFS=',' read -r -a gpu_ids <<< "$GPUS"
    for gpu_id in "${gpu_ids[@]}"; do
        gpu_id="${gpu_id// /}"
        gpu_name=$(query_gpu_name "$gpu_id")
        if [ -n "$gpu_name" ]; then
            DETECTED_GPU_NAMES+=("$gpu_name")
        fi
    done
}

normalize_gpu_list() {
    echo "${1// /}"
}

resolve_gpu_selection() {
    GPUS=$(normalize_gpu_list "$GPUS")

    if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "$ALLOCATED_CUDA_VISIBLE_DEVICES" ]; then
        local slurm_gpus
        slurm_gpus=$(normalize_gpu_list "$ALLOCATED_CUDA_VISIBLE_DEVICES")
        if [ "$GPUS_EXPLICITLY_SET" -eq 1 ] && [ "$GPUS" != "$slurm_gpus" ]; then
            echo "Error: under Slurm, --gpus (${GPUS}) must match allocated CUDA_VISIBLE_DEVICES (${slurm_gpus}), or omit --gpus." >&2
            exit 1
        fi
        GPUS="$slurm_gpus"
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        --gpus) GPUS="$2"; GPUS_EXPLICITLY_SET=1; shift 2 ;;
        --gpus=*) GPUS="${1#*=}"; GPUS_EXPLICITLY_SET=1; shift ;;
        --steps) STEPS="$2"; shift 2 ;;
        --steps=*) STEPS="${1#*=}"; shift ;;
        --save-freq) SAVE_FREQ="$2"; shift 2 ;;
        --save-freq=*) SAVE_FREQ="${1#*=}"; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        --config=*) CONFIG="${1#*=}"; shift ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --model-path=*) MODEL_PATH="${1#*=}"; shift ;;
        --project-name) PROJECT_NAME="$2"; shift 2 ;;
        --project-name=*) PROJECT_NAME="${1#*=}"; shift ;;
        --algo) ALGO=$(normalize_algo "$2"); shift 2 ;;
        --algo=*) ALGO=$(normalize_algo "${1#*=}"); shift ;;
        --env-name) ENV_NAME="$2"; shift 2 ;;
        --env-name=*) ENV_NAME="${1#*=}"; shift ;;
        --env-max-steps) ENV_MAX_STEPS="$2"; shift 2 ;;
        --env-max-steps=*) ENV_MAX_STEPS="${1#*=}"; shift ;;
        --context-window-mode) CONTEXT_WINDOW_MODE="$2"; shift 2 ;;
        --context-window-mode=*) CONTEXT_WINDOW_MODE="${1#*=}"; shift ;;
        --max-context-window) MAX_CONTEXT_WINDOW="$2"; shift 2 ;;
        --max-context-window=*) MAX_CONTEXT_WINDOW="${1#*=}"; shift ;;
        --max-action-points) MAX_ACTION_POINTS="$2"; shift 2 ;;
        --max-action-points=*) MAX_ACTION_POINTS="${1#*=}"; shift ;;
        --max-turn) MAX_TURN="$2"; shift 2 ;;
        --max-turn=*) MAX_TURN="${1#*=}"; shift ;;
        --max-actions-per-turn) MAX_ACTIONS_PER_TURN="$2"; shift 2 ;;
        --max-actions-per-turn=*) MAX_ACTIONS_PER_TURN="${1#*=}"; shift ;;
        --max-actions-per-traj) MAX_ACTIONS_PER_TRAJ="$2"; shift 2 ;;
        --max-actions-per-traj=*) MAX_ACTIONS_PER_TRAJ="${1#*=}"; shift ;;
        --num-groups) NUM_GROUPS="$2"; shift 2 ;;
        --num-groups=*) NUM_GROUPS="${1#*=}"; shift ;;
        --group-size) GROUP_SIZE="$2"; shift 2 ;;
        --group-size=*) GROUP_SIZE="${1#*=}"; shift ;;
        --val-env-groups) VAL_ENV_GROUPS="$2"; shift 2 ;;
        --val-env-groups=*) VAL_ENV_GROUPS="${1#*=}"; shift ;;
        --val-group-size) VAL_GROUP_SIZE="$2"; shift 2 ;;
        --val-group-size=*) VAL_GROUP_SIZE="${1#*=}"; shift ;;
        --micro-batch-size) MICRO_BATCH_SIZE_PER_GPU="$2"; shift 2 ;;
        --micro-batch-size=*) MICRO_BATCH_SIZE_PER_GPU="${1#*=}"; shift ;;
        --log-prob-micro-batch-size) LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="$2"; shift 2 ;;
        --log-prob-micro-batch-size=*) LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${1#*=}"; shift ;;
        --ppo-mini-batch-size) PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
        --ppo-mini-batch-size=*) PPO_MINI_BATCH_SIZE="${1#*=}"; shift ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --gpu-memory-utilization=*) GPU_MEMORY_UTILIZATION="${1#*=}"; shift ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-model-len=*) MAX_MODEL_LEN="${1#*=}"; shift ;;
        --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
        --max-num-batched-tokens=*) MAX_NUM_BATCHED_TOKENS="${1#*=}"; shift ;;
        --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;;
        --checkpoint-root=*) CHECKPOINT_ROOT="${1#*=}"; shift ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --run-name=*) RUN_NAME="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

ALGO=$(normalize_algo "$ALGO")
resolve_gpu_selection

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
GPUS_PER_EXP=${#GPU_IDS[@]}
detect_gpu_names

if [ -z "$RUN_NAME" ]; then
    RUN_NAME=$(build_run_name)
fi

TASK_LOG_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${TASK_LOG_DIR}/train.log"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"

mkdir -p "$TASK_LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

ROBOTOUILLE_INSTRUCTION="You are controlling a kitchen robot. Choose between 1 and ${MAX_ACTIONS_PER_TURN} actions from the provided Valid Actions list for this turn. Think about the next state changes, then inside <answer> output only the exact action string or strings. If you choose multiple actions, separate them with ||, for example: action1 || action2. Do not output more than ${MAX_ACTIONS_PER_TURN} actions or any explanation inside <answer>."

COMMON_OVERRIDES=(
    "model_path=${MODEL_PATH}"
    "micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}"
    "log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    "ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "agent_proxy.max_turn=${MAX_TURN}"
    "agent_proxy.max_actions_per_turn=${MAX_ACTIONS_PER_TURN}"
    "agent_proxy.context_window_mode=${CONTEXT_WINDOW_MODE}"
    "agent_proxy.max_context_window=${MAX_CONTEXT_WINDOW}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.total_training_steps=${STEPS}"
    "trainer.experiment_name=${RUN_NAME}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.default_local_dir=${CHECKPOINT_DIR}"
    "trainer.logger=${LOGGER}"
    "trainer.val_before_train=True"
    "trainer.n_gpus_per_node=${GPUS_PER_EXP}"
    "system.CUDA_VISIBLE_DEVICES='${GPUS}'"
    "es_manager.train.env_groups=${NUM_GROUPS}"
    "es_manager.train.group_size=${GROUP_SIZE}"
    "es_manager.train.env_configs.tags=[Robotouille]"
    "es_manager.train.env_configs.n_groups=[${NUM_GROUPS}]"
    "es_manager.val.env_groups=${VAL_ENV_GROUPS}"
    "es_manager.val.group_size=${VAL_GROUP_SIZE}"
    "es_manager.val.env_configs.tags=[Robotouille]"
    "es_manager.val.env_configs.n_groups=[${VAL_ENV_GROUPS}]"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.rollout_filter_strategy=top_p"
    "actor_rollout_ref.rollout.rollout_filter_value=1.0"
    "actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode=linear"
    "actor_rollout_ref.rollout.rollout_filter_type=largest"
    "actor_rollout_ref.rollout.rollout_filter_metric=reward_variance"
    "actor_rollout_ref.rollout.rollout_filter_include_zero=True"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.actor.checkpoint.save_contents=[model]"
    "critic.checkpoint.save_contents=[model]"
    "agent_proxy.mixed_turn_budget.enabled=False"
    "agent_proxy.mixed_turn_budget.mixed_budget=False"
    "agent_proxy.mixed_token_budget.enabled=False"
    "agent_proxy.mixed_token_budget.mixed_budget=False"
    "agent_proxy.mixed_toolcall_budget.enabled=False"
    "agent_proxy.mixed_toolcall_budget.mixed_budget=False"
    "custom_envs.Robotouille.env_instruction='${ROBOTOUILLE_INSTRUCTION}'"
    "++custom_envs.Robotouille.env_config.env_name=${ENV_NAME}"
    "++custom_envs.Robotouille.env_config.max_steps=${ENV_MAX_STEPS}"
    "++custom_envs.Robotouille.env_config.enable_action_budget=True"
    "++custom_envs.Robotouille.env_config.max_action_points=${MAX_ACTION_POINTS}"
    "custom_envs.Robotouille.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ}"
)

if [ -n "$MAX_MODEL_LEN" ]; then
    COMMON_OVERRIDES+=("actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}")
fi

if [ -n "$MAX_NUM_BATCHED_TOKENS" ]; then
    COMMON_OVERRIDES+=("actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}")
fi

ALGO_OVERRIDES=()
read -r -a ALGO_OVERRIDES <<< "$(get_algo_overrides "$ALGO")"

echo "Repo root: $REPO_ROOT"
echo "Run name: $RUN_NAME"
echo "Mode: plain RL (mixed toolcall/turn/token budget disabled)"
echo "Robotouille env: ${ENV_NAME}"
echo "Context window mode: ${CONTEXT_WINDOW_MODE}"
echo "Max context window: ${MAX_CONTEXT_WINDOW}"
echo "Fixed action budget: ${MAX_ACTION_POINTS}"
echo "Algorithm: $ALGO"
echo "GPUs: $GPUS"
echo "Detected GPUs: ${DETECTED_GPU_NAMES[*]}"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Log path: $LOG_PATH"

CMD=(
    python train.py
    --config-name "$CONFIG"
    "${COMMON_OVERRIDES[@]}"
    "${ALGO_OVERRIDES[@]}"
)

printf 'Command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

if [ "$PREFLIGHT_ONLY" = "1" ]; then
    echo "Preflight only: command constructed successfully; skipping training launch."
    exit 0
fi

CUDA_VISIBLE_DEVICES="$GPUS" "${CMD[@]}" 2>&1 | tee "$LOG_PATH"
