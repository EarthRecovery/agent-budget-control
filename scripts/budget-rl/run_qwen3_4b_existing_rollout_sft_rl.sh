#!/usr/bin/env bash
set -euo pipefail

# Train Qwen3-4B budget probe with the existing requested 128 Sokoban rollouts.
# This intentionally skips live rollout generation by default.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/budget-rl/run_qwen3_4b_existing_rollout_sft_rl.sh [prepare|sft|rl|prepare,sft,rl]

Default:
  Uses the existing Qwen3-4B Sokoban rollout JSONL, then runs prepare + SFT + RL.

Common overrides:
  DRY_RUN=1
  NGPUS=1
  TP_SIZE=1
  MODEL=Qwen/Qwen3-4B
  ROLLOUT_JSONL=/path/to/sokoban_qwen3_4b.jsonl
  EXP_BASE=/path/to/output_dir
  SFT_TOTAL_EPOCHS=5
  RL_TOTAL_EPOCHS=5
  WANDB_MODE=offline
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

MODEL="${MODEL:-${QWEN3_4B_MODEL:-Qwen/Qwen3-4B}}"
EXP_NAME="${EXP_NAME:-sokoban_qwen3_4b_existing_rollout_sft_rl}"
EXP_BASE="${EXP_BASE:-$PROJECT_ROOT/data/budget-rl/$EXP_NAME}"
ROLLOUT_JSONL="${ROLLOUT_JSONL:-$PROJECT_ROOT/data/budget-rl/requested-128-rollouts/sokoban_qwen3_4b/sokoban_qwen3_4b.jsonl}"
STAGES_TEXT="${1:-${STAGES:-prepare,sft,rl}}"

# Keep "all" aligned with this script's purpose: consume existing rollout data.
if [[ "$STAGES_TEXT" = "all" ]]; then
  STAGES_TEXT="prepare,sft,rl"
fi
if [[ ",${STAGES_TEXT// /,}," == *",rollout,"* && "${INCLUDE_ROLLOUT:-0}" != "1" ]]; then
  echo "This script is for existing rollouts and will not run rollout by default." >&2
  echo "Use STAGES=prepare,sft,rl, or set INCLUDE_ROLLOUT=1 if you really want rollout." >&2
  exit 2
fi

if [[ ! -f "$ROLLOUT_JSONL" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "Missing rollout JSONL: $ROLLOUT_JSONL" >&2
  exit 2
fi

export TASK="${TASK:-sokoban}"
export ROLLOUT_MODEL="${ROLLOUT_MODEL:-$MODEL}"
export LEARNER_MODEL="${LEARNER_MODEL:-$MODEL}"
export TOKENIZER="${TOKENIZER:-$MODEL}"
export EXP_NAME
export EXP_BASE
export ROLLOUT_JSONL

# Defaults sized for a single large GPU; override for multi-GPU runs.
export NGPUS="${NGPUS:-1}"
export TP_SIZE="${TP_SIZE:-1}"
export SFT_NGPUS="${SFT_NGPUS:-$NGPUS}"
export RL_NGPUS="${RL_NGPUS:-$NGPUS}"
export RL_TP_SIZE="${RL_TP_SIZE:-$TP_SIZE}"

export SFT_ABLATION="${SFT_ABLATION:-sft_interval_pct30}"
export SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-5}"
export RL_TOTAL_EPOCHS="${RL_TOTAL_EPOCHS:-5}"
export RL_KL="${RL_KL:-0.05}"

export SFT_TRAIN_BS="${SFT_TRAIN_BS:-8}"
export SFT_MICRO_BS="${SFT_MICRO_BS:-1}"
export RL_BATCH_SIZE="${RL_BATCH_SIZE:-16}"
export RL_PPO_MICRO_BS="${RL_PPO_MICRO_BS:-1}"
export RL_LOGPROB_MICRO_BS="${RL_LOGPROB_MICRO_BS:-1}"
export RL_ROLLOUT_N="${RL_ROLLOUT_N:-8}"
export RL_GPU_UTIL="${RL_GPU_UTIL:-0.4}"

export MAX_TOKENS="${MAX_TOKENS:-8192}"
export SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-9216}"
export RL_MAX_MODEL_LEN="${RL_MAX_MODEL_LEN:-8192}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\",\"wandb\"]}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$EXP_NAME}"

# The local ragenv2 flash-attn wheel requires GLIBC_2.32 on some machines.
# Use PyTorch SDPA by default so Qwen3 can load without importing flash_attn.
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"

append_arg_once() {
  local var_name="$1"
  local key="$2"
  local arg="$3"
  local current="${!var_name:-}"

  case " $current " in
    *" $key="*|*" +$key="*)
      ;;
    *)
      printf -v "$var_name" '%s%s%s' "$current" "${current:+ }" "$arg"
      export "$var_name"
      ;;
  esac
}

if [[ "${DISABLE_FLASH_ATTN_WORKAROUND:-0}" != "1" ]]; then
  append_arg_once SFT_EXTRA_ARGS \
    "model.override_config.attn_implementation" \
    "+model.override_config.attn_implementation=$ATTN_IMPLEMENTATION"
  append_arg_once SFT_EXTRA_ARGS \
    "model.use_remove_padding" \
    "model.use_remove_padding=$USE_REMOVE_PADDING"
  append_arg_once RL_EXTRA_ARGS \
    "actor_rollout_ref.model.override_config.attn_implementation" \
    "+actor_rollout_ref.model.override_config.attn_implementation=$ATTN_IMPLEMENTATION"
  append_arg_once RL_EXTRA_ARGS \
    "actor_rollout_ref.model.use_remove_padding" \
    "actor_rollout_ref.model.use_remove_padding=$USE_REMOVE_PADDING"
fi

SFT_CKPT="${SFT_CKPT:-$EXP_BASE/checkpoints/$SFT_ABLATION/huggingface_e$SFT_TOTAL_EPOCHS}"
RL_EXPERIMENT_NAME="${RL_EXPERIMENT_NAME:-${EXP_NAME}_rl_${SFT_ABLATION}_e${SFT_TOTAL_EPOCHS}_kl$(printf '%s' "$RL_KL" | tr -d '.')}"
RL_SAVE_DIR="${RL_SAVE_DIR:-$EXP_BASE/checkpoints/$RL_EXPERIMENT_NAME}"
export SFT_CKPT
export RL_EXPERIMENT_NAME
export RL_SAVE_DIR

print_model_params() {
  local latest_rl_checkpoint="not_created_yet"
  if [[ -d "$RL_SAVE_DIR" ]]; then
    latest_rl_checkpoint="$(
      find "$RL_SAVE_DIR" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null \
        | sort -V \
        | tail -n 1
    )"
    latest_rl_checkpoint="${latest_rl_checkpoint:-not_created_yet}"
  fi

  cat <<EOF

=== Qwen3-4B budget-RL model params ===
base_model=${MODEL}
rollout_model=${ROLLOUT_MODEL}
learner_model=${LEARNER_MODEL}
tokenizer=${TOKENIZER}
task=${TASK}
rollout_jsonl=${ROLLOUT_JSONL}
exp_base=${EXP_BASE}
sft_ablation=${SFT_ABLATION}
sft_epochs=${SFT_TOTAL_EPOCHS}
sft_checkpoint=${SFT_CKPT}
rl_epochs=${RL_TOTAL_EPOCHS}
rl_kl=${RL_KL}
rl_checkpoint_dir=${RL_SAVE_DIR}
rl_latest_checkpoint=${latest_rl_checkpoint}
ngpus=${NGPUS}
tp_size=${TP_SIZE}
sft_train_bs=${SFT_TRAIN_BS}
sft_micro_bs=${SFT_MICRO_BS}
rl_batch_size=${RL_BATCH_SIZE}
rl_rollout_n=${RL_ROLLOUT_N}
max_tokens=${MAX_TOKENS}
attn_implementation=${ATTN_IMPLEMENTATION}
use_remove_padding=${USE_REMOVE_PADDING}
EOF
}

summarize_rollout() {
  if [[ ! -f "$ROLLOUT_JSONL" ]]; then
    return 0
  fi
  python3 - "$ROLLOUT_JSONL" <<'PY'
import json
import sys

path = sys.argv[1]
total = 0
success = 0
turns = []

with open(path, "r", encoding="utf-8") as handle:
    for line in handle:
        obj = json.loads(line)
        total += 1
        meta = obj.get("metadata", {})
        success += int(bool(meta.get("success", False)))
        if isinstance(meta.get("num_turns"), (int, float)):
            turns.append(float(meta["num_turns"]))

mean_turns = sum(turns) / len(turns) if turns else 0.0
rate = (100.0 * success / total) if total else 0.0
print("Existing rollout summary")
print(f"  path: {path}")
print(f"  trajectories: {total}")
print(f"  rollout_success: {success}/{total} ({rate:.2f}%)")
print(f"  mean_num_turns: {mean_turns:.2f}")
PY
}

summarize_rollout
print_model_params

env \
  TASK="$TASK" \
  ROLLOUT_MODEL="$ROLLOUT_MODEL" \
  LEARNER_MODEL="$LEARNER_MODEL" \
  TOKENIZER="$TOKENIZER" \
  EXP_NAME="$EXP_NAME" \
  EXP_BASE="$EXP_BASE" \
  ROLLOUT_JSONL="$ROLLOUT_JSONL" \
  SFT_CKPT="$SFT_CKPT" \
  RL_EXPERIMENT_NAME="$RL_EXPERIMENT_NAME" \
  RL_SAVE_DIR="$RL_SAVE_DIR" \
  bash "$SCRIPT_DIR/run_budget_rl_pipeline.sh" "$STAGES_TEXT"

print_model_params
