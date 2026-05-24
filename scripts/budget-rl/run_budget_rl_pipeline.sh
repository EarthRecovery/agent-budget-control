#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/budget-rl}"

TASK="${TASK:-sokoban}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-Qwen/Qwen3-8B}"
LEARNER_MODEL="${LEARNER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER="${TOKENIZER:-$LEARNER_MODEL}"
SFT_ABLATION="${SFT_ABLATION:-sft_interval_pct30}"
SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-5}"
RL_KL="${RL_KL:-0.05}"

slug() {
  printf '%s' "$1" | tr '/:.' '---' | tr -cs 'A-Za-z0-9_-' '-'
}

ROLLOUT_SLUG="$(slug "$ROLLOUT_MODEL")"
LEARNER_SLUG="$(slug "$LEARNER_MODEL")"
EXP_NAME="${EXP_NAME:-${TASK}_${ROLLOUT_SLUG}_to_${LEARNER_SLUG}}"
EXP_BASE="${EXP_BASE:-$DATA_ROOT/$EXP_NAME}"
if [[ "$EXP_BASE" != /* ]]; then
  EXP_BASE="$PROJECT_ROOT/$EXP_BASE"
fi
ROLLOUT_JSONL="${ROLLOUT_JSONL:-$EXP_BASE/rollouts.jsonl}"
if [[ "$ROLLOUT_JSONL" != /* ]]; then
  ROLLOUT_JSONL="$PROJECT_ROOT/$ROLLOUT_JSONL"
fi
STAGES_TEXT="${1:-${STAGES:-all}}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/budget-rl/run_budget_rl_pipeline.sh [all|rollout|prepare|sft|rl|rollout,prepare,...]

This runs a local-model budget RL loop inside agent-budget-control:
  1. rollout: generate task trajectories with a HF/vLLM model via RAGEN
  2. prepare: convert trajectories into SFT and RL parquet data
  3. sft: supervised warm-up on budget-estimation probes
  4. rl: GRPO on the budget-estimation reward

Common overrides:
  TASK=sokoban
  TASK=searchr1
  TASK=warehouse          # requires ROLLOUT_SOURCE_JSON or ROLLOUT_SOURCE_DIR
  TASK=swebench           # requires SWE_INPUT_DIR, ROLLOUT_SOURCE_JSON, or ROLLOUT_SOURCE_DIR
  ROLLOUT_MODEL=Qwen/Qwen3-8B
  LEARNER_MODEL=Qwen/Qwen2.5-7B-Instruct
  NUM_TRAJECTORIES=128
  EXP_BASE=data/budget-rl/my_run
  NGPUS=8
  TP_SIZE=4
  SFT_TOTAL_EPOCHS=5
  RL_TOTAL_EPOCHS=5
  DRY_RUN=1

For SearchR1, start a retrieval server first or set SEARCH_MOCK_MODE=true.
For Warehouse/SWE, the rollout stage converts pre-existing rollout logs into
the JSONL format used by the shared SFT/RL data preparation.
EOF
}

contains_stage() {
  local wanted="$1"
  [[ "$STAGES_TEXT" = "all" ]] && return 0
  [[ ",${STAGES_TEXT// /,}," == *",$wanted,"* ]]
}

task_name_for_prepare() {
  case "$TASK" in
    sokoban|coord_sokoban) printf 'sokoban' ;;
    searchr1|search|searchqa) printf 'searchr1' ;;
    swe|swebench|swebanch|miniswebench) printf 'swebench' ;;
    warehouse|money|money_estimation) printf 'warehouse' ;;
    *) printf '%s' "$TASK" ;;
  esac
}

normalize_path() {
  local path="$1"
  if [[ "$path" == "~/"* ]]; then
    path="$HOME/${path#~/}"
  elif [[ "$path" == "~" ]]; then
    path="$HOME"
  fi
  if [[ "$path" != /* ]]; then
    path="$PROJECT_ROOT/$path"
  fi
  printf '%s\n' "$path"
}

is_source_rollout_task() {
  case "$TASK" in
    warehouse|money|money_estimation|swe|swebench|swebanch|miniswebench)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

combine_rollout_json_dir() {
  local source_dir="$1"
  local output_json="$2"
  python3 - "$source_dir" "$output_json" <<'PY'
import json
import sys
from pathlib import Path

source_dir = Path(sys.argv[1]).expanduser().resolve()
output_json = Path(sys.argv[2]).expanduser().resolve()
rollouts = []

def maybe_add(record):
    if isinstance(record, dict) and "turns" in record:
        rollouts.append(record)

for path in sorted(source_dir.rglob("*.json")):
    if path.resolve() == output_json:
        continue
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        for item in payload:
            maybe_add(item)
    elif isinstance(payload, dict):
        if isinstance(payload.get("rollouts"), list):
            for item in payload["rollouts"]:
                maybe_add(item)
        elif isinstance(payload.get("results"), list):
            for item in payload["results"]:
                maybe_add(item)
        else:
            maybe_add(payload)

output_json.parent.mkdir(parents=True, exist_ok=True)
with output_json.open("w", encoding="utf-8") as handle:
    json.dump(rollouts, handle, ensure_ascii=False, indent=2)
print(f"Combined {len(rollouts)} rollout(s) from {source_dir} -> {output_json}")
if not rollouts:
    raise SystemExit("No rollout records with a 'turns' field were found.")
PY
}

prepare_source_rollouts() {
  local source_json=""
  local source_dir=""
  local convert_json="$EXP_BASE/source_rollouts.json"
  local -a convert_cmd

  if [[ -n "${SWE_INPUT_DIR:-}" ]]; then
    source_dir="$(normalize_path "$SWE_INPUT_DIR")"
    source_json="$EXP_BASE/swebench_dialogues.json"
    local -a prepare_cmd=(
      python3 "$PROJECT_ROOT/scripts/budget-estimation-benchmark/prepare_swebench_dialogues.py"
      --input-dir "$source_dir"
      --output-json "$source_json"
    )
    if [[ -n "${MAX_ROLLOUTS:-}" ]]; then
      prepare_cmd+=(--max-rollouts "$MAX_ROLLOUTS")
    fi
    if [[ "${DRY_RUN:-0}" = "1" ]]; then
      printf '%q ' "${prepare_cmd[@]}"
      printf '\n'
    else
      "${prepare_cmd[@]}"
    fi
  elif [[ -n "${ROLLOUT_SOURCE_JSON:-}" ]]; then
    source_json="$(normalize_path "$ROLLOUT_SOURCE_JSON")"
  elif [[ -n "${ROLLOUT_SOURCE_DIR:-}" ]]; then
    source_dir="$(normalize_path "$ROLLOUT_SOURCE_DIR")"
    source_json="$convert_json"
    if [[ "${DRY_RUN:-0}" = "1" ]]; then
      echo "DRY_RUN: would combine rollout JSON files from $source_dir -> $source_json"
    else
      combine_rollout_json_dir "$source_dir" "$source_json"
    fi
  else
    if [[ "${DRY_RUN:-0}" = "1" ]]; then
      echo "DRY_RUN: source task $TASK requires ROLLOUT_SOURCE_JSON, ROLLOUT_SOURCE_DIR, or SWE_INPUT_DIR"
      return 0
    fi
    echo "TASK=$TASK uses pre-existing rollout data. Set ROLLOUT_SOURCE_JSON, ROLLOUT_SOURCE_DIR, or SWE_INPUT_DIR." >&2
    exit 2
  fi

  if [[ "${DRY_RUN:-0}" != "1" && ! -f "$source_json" ]]; then
    echo "Rollout source JSON not found: $source_json" >&2
    exit 2
  fi

  convert_cmd=(
    python3 "$SCRIPT_DIR/convert_estimation_dialogues.py"
    --input "$source_json"
    --output "$ROLLOUT_JSONL"
  )
  if [[ -n "${ROLLOUT_SOURCE_MAX_TURNS:-}" ]]; then
    convert_cmd+=(--max-turns "$ROLLOUT_SOURCE_MAX_TURNS")
  fi

  if [[ "${DRY_RUN:-0}" = "1" ]]; then
    printf '%q ' "${convert_cmd[@]}"
    printf '\n'
    return 0
  fi

  "${convert_cmd[@]}"
}

activate_runtime() {
  if [[ "${SKIP_ENV_ACTIVATE:-0}" = "1" ]]; then
    return 0
  fi
  if [[ -n "${VENV_PATH:-}" && -f "$VENV_PATH/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
    return 0
  fi
  if [[ -n "${CONDA_ENV_NAME:-ragenv2}" ]]; then
    if command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)"
      conda activate "${CONDA_ENV_NAME:-ragenv2}"
    elif [[ -n "${CONDA_BASE:-}" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$CONDA_BASE/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV_NAME:-ragenv2}"
    fi
  fi
  export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
}

run_rollout() {
  if is_source_rollout_task; then
    prepare_source_rollouts
    return 0
  fi

  ROLLOUT_MODEL="$ROLLOUT_MODEL" \
  TASK="$TASK" \
  DATA_ROOT="$DATA_ROOT" \
  OUTPUT_JSONL="$ROLLOUT_JSONL" \
  bash "$SCRIPT_DIR/run_model_rollout.sh" "$TASK"
}

run_prepare() {
  if [[ ! -f "$ROLLOUT_JSONL" ]]; then
    if [[ "${DRY_RUN:-0}" = "1" ]]; then
      echo "DRY_RUN: would require rollout JSONL at $ROLLOUT_JSONL"
    else
      echo "Missing rollout JSONL: $ROLLOUT_JSONL" >&2
      exit 2
    fi
  fi

  local -a cmd=(
    env
    "INPUT=$ROLLOUT_JSONL"
    "BASE=$EXP_BASE"
    "SEED=${SEED:-42}"
    "SFT_FRAC=${SFT_FRAC:-0.4}"
    "RL_FRAC=${RL_FRAC:-0.5}"
    "TEST_FRAC=${TEST_FRAC:-0.1}"
    "MAX_TOKENS=${MAX_TOKENS:-8192}"
    "PROBE_EVERY_N=${PROBE_EVERY_N:-1}"
    "TOKENIZER=$TOKENIZER"
    "TASK_NAME=$(task_name_for_prepare)"
    "SFT_VARIANTS=$SFT_ABLATION"
    "BUDGET_PROBE_CONTEXT_WINDOW_MODE=${BUDGET_PROBE_CONTEXT_WINDOW_MODE:-full}"
    "BUDGET_PROBE_MAX_CONTEXT_WINDOW=${BUDGET_PROBE_MAX_CONTEXT_WINDOW:--1}"
    "BUDGET_PROBE_MAX_PROMPT_TOKENS=${BUDGET_PROBE_MAX_PROMPT_TOKENS:-}"
    "BUDGET_PROBE_DROP_OVERLONG_PROMPTS=${BUDGET_PROBE_DROP_OVERLONG_PROMPTS:-0}"
    bash "$SCRIPT_DIR/prepare_all_ablations.sh"
  )
  if [[ "${DRY_RUN:-0}" = "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}"
}

run_sft() {
  local -a cmd=(
    env
    "BASE=$EXP_BASE"
    "MODEL=$LEARNER_MODEL"
    "NGPUS=${SFT_NGPUS:-${NGPUS:-8}}"
    "LR=${SFT_LR:-5e-6}"
    "TOTAL_EPOCHS=$SFT_TOTAL_EPOCHS"
    "TRAIN_BS=${SFT_TRAIN_BS:-16}"
    "MICRO_BS=${SFT_MICRO_BS:-2}"
    "PROJECT_NAME=${SFT_PROJECT_NAME:-budget_probe_sft}"
    "EXPERIMENT_NAME=${SFT_EXPERIMENT_NAME:-${EXP_NAME}_sft_${SFT_ABLATION}_e${SFT_TOTAL_EPOCHS}}"
    "WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-$EXP_NAME}"
    "WANDB_NAME=${WANDB_NAME:-${EXP_NAME}_sft_${SFT_ABLATION}_e${SFT_TOTAL_EPOCHS}}"
    bash "$SCRIPT_DIR/run_sft_ablation.sh" "$SFT_ABLATION"
  )
  if [[ -n "${SFT_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local extra=(${SFT_EXTRA_ARGS})
    cmd+=("${extra[@]}")
  fi
  if [[ "${DRY_RUN:-0}" = "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}"
}

run_rl() {
  local sft_ckpt rl_name rl_dir rl_model
  sft_ckpt="${SFT_CKPT:-$EXP_BASE/checkpoints/$SFT_ABLATION/huggingface_e$SFT_TOTAL_EPOCHS}"
  rl_name="${RL_EXPERIMENT_NAME:-${EXP_NAME}_rl_${SFT_ABLATION}_e${SFT_TOTAL_EPOCHS}_kl$(printf '%s' "$RL_KL" | tr -d '.')}"
  rl_dir="${RL_SAVE_DIR:-$EXP_BASE/checkpoints/$rl_name}"
  rl_model="${RL_INIT_MODEL:-$sft_ckpt}"

  if [[ -z "${RL_INIT_MODEL:-}" && ! -f "$sft_ckpt/config.json" ]]; then
    if [[ "${DRY_RUN:-0}" = "1" ]]; then
      echo "DRY_RUN: would require SFT checkpoint at $sft_ckpt/config.json"
    else
      echo "Missing SFT checkpoint: $sft_ckpt/config.json" >&2
      exit 3
    fi
  fi
  if [[ -n "${RL_INIT_MODEL:-}" && -e "$rl_model" && ! -f "$rl_model/config.json" ]]; then
    echo "RL_INIT_MODEL exists but has no config.json: $rl_model" >&2
    exit 3
  fi

  local -a cmd=(
    env
    "DATA_DIR=$EXP_BASE"
    "MODEL=$rl_model"
    "NGPUS=${RL_NGPUS:-${NGPUS:-8}}"
    "TP_SIZE=${RL_TP_SIZE:-${TP_SIZE:-4}}"
    "TRAIN_BATCH_SIZE=${RL_BATCH_SIZE:-64}"
    "LR=${RL_LR:-5e-7}"
    "TOTAL_EPOCHS=${RL_TOTAL_EPOCHS:-5}"
    "ROLLOUT_N=${RL_ROLLOUT_N:-16}"
    "PROJECT_NAME=${RL_PROJECT_NAME:-budget_probe_grpo}"
    "EXPERIMENT_NAME=$rl_name"
    "WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-$EXP_NAME}"
    "WANDB_NAME=$rl_name"
    "RESUME_MODE=${RESUME_MODE:-disable}"
    bash "$SCRIPT_DIR/run_budget_probe_grpo.sh"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${RL_BATCH_SIZE:-64}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${RL_PPO_MICRO_BS:-4}"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${RL_LOGPROB_MICRO_BS:-4}"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${RL_LOGPROB_MICRO_BS:-4}"
    "actor_rollout_ref.actor.kl_loss_coef=$RL_KL"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${RL_GPU_UTIL:-0.4}"
    "actor_rollout_ref.rollout.max_model_len=${RL_MAX_MODEL_LEN:-8192}"
    "trainer.default_local_dir=$rl_dir"
    "trainer.save_freq=${RL_SAVE_FREQ:-10}"
    'actor_rollout_ref.actor.checkpoint.save_contents=["model","extra"]'
    'actor_rollout_ref.actor.checkpoint.load_contents=["model","extra"]'
  )
  if [[ -n "${RL_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local extra=(${RL_EXTRA_ARGS})
    cmd+=("${extra[@]}")
  fi
  if [[ "${DRY_RUN:-0}" = "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}"
}

main() {
  case "${1:-}" in
    -h|--help)
      usage
      exit 0
      ;;
  esac

  mkdir -p "$EXP_BASE"
  if [[ "${BUDGET_RL_DISABLE_LOG:-0}" != "1" && "${BUDGET_RL_LOG_STARTED:-0}" != "1" ]]; then
    BUDGET_RL_LOG_FILE="${BUDGET_RL_LOG_FILE:-$EXP_BASE/run.log}"
    mkdir -p "$(dirname "$BUDGET_RL_LOG_FILE")"
    if [[ "${BUDGET_RL_APPEND_LOG:-0}" != "1" ]]; then
      : >"$BUDGET_RL_LOG_FILE"
    fi
    export BUDGET_RL_LOG_FILE
    export BUDGET_RL_LOG_STARTED=1
    exec > >(tee -a "$BUDGET_RL_LOG_FILE") 2>&1
  fi
  activate_runtime
  echo "Budget RL pipeline"
  echo "  stages: $STAGES_TEXT"
  echo "  task: $TASK"
  echo "  rollout model: $ROLLOUT_MODEL"
  echo "  learner model: $LEARNER_MODEL"
  echo "  exp base: $EXP_BASE"
  echo "  rollout jsonl: $ROLLOUT_JSONL"
  echo "  log file: ${BUDGET_RL_LOG_FILE:-disabled}"
  echo "  ngpus: ${NGPUS:-8}"
  echo "  tp_size: ${TP_SIZE:-4}"
  echo "  rl_batch_size: ${RL_BATCH_SIZE:-64}"
  echo "  rl_rollout_n: ${RL_ROLLOUT_N:-16}"
  echo "  rl_lr: ${RL_LR:-5e-7}"
  echo "  rl_epochs: ${RL_TOTAL_EPOCHS:-5}"
  echo "  rl_kl: $RL_KL"

  if contains_stage rollout; then
    echo
    echo "=== rollout ==="
    run_rollout
  fi
  if contains_stage prepare; then
    echo
    echo "=== prepare ==="
    run_prepare
  fi
  if contains_stage sft; then
    echo
    echo "=== sft ==="
    run_sft
  fi
  if contains_stage rl; then
    echo
    echo "=== rl ==="
    run_rl
  fi
}

main "$@"
