#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-swe_gpt52instant_existing_data}"
TASK="${TASK:-swebench}"
if [[ -z "${DATA_FILE:-}" ]]; then
  if [[ -f "$SCRIPT_DIR/SWE_gpt5.2instant.json" ]]; then
    DATA_FILE="$SCRIPT_DIR/SWE_gpt5.2instant.json"
  else
    DATA_FILE="$SCRIPT_DIR/swebench-origin-gpt5.2instant/openai__gpt-5.2.gpt52_instant.json"
  fi
fi
if [[ -z "${SWE_INPUT_DIR:-}" && -d "$SCRIPT_DIR/swebench-origin-gpt5.2instant" ]]; then
  SWE_INPUT_DIR="$SCRIPT_DIR/swebench-origin-gpt5.2instant"
fi
SOURCE_KIND="swe_summary"
LEARNER_MODEL="${LEARNER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-GPT5.2 instant}"
TOKENIZER="${TOKENIZER:-$LEARNER_MODEL}"
SYSTEM_PROMPT_EXTRA="${SYSTEM_PROMPT_EXTRA:-the rollout is generated from GPT5.2 instant}"

NGPUS="${NGPUS:-8}"
TP_SIZE="${TP_SIZE:-4}"
SFT_NGPUS="${SFT_NGPUS:-$NGPUS}"
RL_NGPUS="${RL_NGPUS:-$NGPUS}"
RL_TP_SIZE="${RL_TP_SIZE:-$TP_SIZE}"

swe_normalize_path() {
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

swe_quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

swe_contains_stage() {
  local stages_text="${1:-${STAGES:-prepare,sft,rl,compare}}"
  local wanted="$2"
  [[ "$stages_text" == "all" ]] && return 0
  [[ ",${stages_text// /,}," == *",$wanted,"* ]]
}

swe_stage_needs_rollouts() {
  local stages_text="${1:-${STAGES:-prepare,sft,rl,compare}}"
  for stage in prepare sft rl compare; do
    if swe_contains_stage "$stages_text" "$stage"; then
      return 0
    fi
  done
  return 1
}

merge_swe_rollouts() {
  local stages_text="${1:-${STAGES:-prepare,sft,rl,compare}}"
  if ! swe_stage_needs_rollouts "$stages_text"; then
    return 0
  fi
  if [[ -n "${SWE_ROLLOUT_SOURCE_JSON:-}" || -n "${ROLLOUT_SOURCE_JSON:-}" ]]; then
    echo "Using pre-merged SWE rollout JSON: ${SWE_ROLLOUT_SOURCE_JSON:-$ROLLOUT_SOURCE_JSON}"
    unset SWE_INPUT_DIR
    return 0
  fi
  if [[ -z "${SWE_INPUT_DIR:-}" ]]; then
    return 0
  fi

  local data_root exp_base source_dir merged_json
  data_root="${DATA_ROOT:-$PROJECT_ROOT/data/budget-rl/from-existing-data}"
  exp_base="${EXP_BASE:-$data_root/$EXPERIMENT_NAME}"
  exp_base="$(swe_normalize_path "$exp_base")"
  source_dir="$(swe_normalize_path "$SWE_INPUT_DIR")"
  merged_json="${SWE_MERGED_ROLLOUT_JSON:-$exp_base/swebench_dialogues.json}"
  merged_json="$(swe_normalize_path "$merged_json")"

  local -a cmd=(
    python3 "$PROJECT_ROOT/scripts/budget-estimation-benchmark/prepare_swebench_dialogues.py"
    --input-dir "$source_dir"
    --output-json "$merged_json"
  )
  if [[ -n "${MAX_ROLLOUTS:-}" ]]; then
    cmd+=(--max-rollouts "$MAX_ROLLOUTS")
  fi

  echo "=== merge dispersed SWE rollouts ==="
  echo "  input_dir: $source_dir"
  echo "  merged_json: $merged_json"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    swe_quote_cmd "${cmd[@]}"
  else
    mkdir -p "$(dirname "$merged_json")"
    "${cmd[@]}"
  fi

  export SWE_ROLLOUT_SOURCE_JSON="$merged_json"
  unset SWE_INPUT_DIR
}

export EXPERIMENT_NAME TASK DATA_FILE SOURCE_KIND LEARNER_MODEL ROLLOUT_MODEL TOKENIZER
export SYSTEM_PROMPT_EXTRA
export NGPUS TP_SIZE SFT_NGPUS RL_NGPUS RL_TP_SIZE

merge_swe_rollouts "${1:-${STAGES:-prepare,sft,rl,compare}}"
export SWE_INPUT_DIR

# shellcheck source=scripts/budget-rl-data/budget_rl_existing_data_common.inc
source "$SCRIPT_DIR/budget_rl_existing_data_common.inc"
budget_rl_data_main "$@"
