#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-warehouse_gpt52instant_existing_data}"
TASK="${TASK:-warehouse}"
if [[ -z "${DATA_FILE:-}" ]]; then
  if [[ -f "$SCRIPT_DIR/warehouse_gpt5.2instant.json" ]]; then
    DATA_FILE="$SCRIPT_DIR/warehouse_gpt5.2instant.json"
  else
    DATA_FILE="$SCRIPT_DIR/warehouse_gpt5.2.json"
  fi
fi
SOURCE_KIND="estimation_json"
LEARNER_MODEL="${LEARNER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-GPT5.2 instant}"
TOKENIZER="${TOKENIZER:-$LEARNER_MODEL}"
SYSTEM_PROMPT_EXTRA="${SYSTEM_PROMPT_EXTRA:-the rollout is generated from GPT5.2 instant}"
BUDGET_PROBE_CONTEXT_WINDOW_MODE="${BUDGET_PROBE_CONTEXT_WINDOW_MODE:-limited_multi_turn}"
BUDGET_PROBE_MAX_CONTEXT_WINDOW="${BUDGET_PROBE_MAX_CONTEXT_WINDOW:-3}"
BUDGET_PROBE_MAX_PROMPT_TOKENS="${BUDGET_PROBE_MAX_PROMPT_TOKENS:-8192}"
BUDGET_PROBE_DROP_OVERLONG_PROMPTS="${BUDGET_PROBE_DROP_OVERLONG_PROMPTS:-1}"

NGPUS="${NGPUS:-8}"
TP_SIZE="${TP_SIZE:-4}"
SFT_NGPUS="${SFT_NGPUS:-$NGPUS}"
RL_NGPUS="${RL_NGPUS:-$NGPUS}"
RL_TP_SIZE="${RL_TP_SIZE:-$TP_SIZE}"

export EXPERIMENT_NAME TASK DATA_FILE SOURCE_KIND LEARNER_MODEL ROLLOUT_MODEL TOKENIZER
export SYSTEM_PROMPT_EXTRA
export BUDGET_PROBE_CONTEXT_WINDOW_MODE BUDGET_PROBE_MAX_CONTEXT_WINDOW
export BUDGET_PROBE_MAX_PROMPT_TOKENS BUDGET_PROBE_DROP_OVERLONG_PROMPTS
export NGPUS TP_SIZE SFT_NGPUS RL_NGPUS RL_TP_SIZE

# shellcheck source=scripts/budget-rl-data/budget_rl_existing_data_common.inc
source "$SCRIPT_DIR/budget_rl_existing_data_common.inc"
budget_rl_data_main "$@"
