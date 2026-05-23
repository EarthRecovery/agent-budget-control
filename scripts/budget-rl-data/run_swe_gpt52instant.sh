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

export EXPERIMENT_NAME TASK DATA_FILE SOURCE_KIND LEARNER_MODEL ROLLOUT_MODEL TOKENIZER
export SWE_INPUT_DIR
export SYSTEM_PROMPT_EXTRA
export NGPUS TP_SIZE SFT_NGPUS RL_NGPUS RL_TP_SIZE

# shellcheck source=scripts/budget-rl-data/budget_rl_existing_data_common.inc
source "$SCRIPT_DIR/budget_rl_existing_data_common.inc"
budget_rl_data_main "$@"
