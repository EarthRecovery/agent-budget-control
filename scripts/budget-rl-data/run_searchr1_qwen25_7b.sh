#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-searchr1_qwen25_7b_existing_data}"
TASK="${TASK:-searchr1}"
DATA_FILE="${DATA_FILE:-$SCRIPT_DIR/searchr1_qwen25_7b.jsonl}"
SOURCE_KIND="jsonl"
LEARNER_MODEL="${LEARNER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER="${TOKENIZER:-$LEARNER_MODEL}"

NGPUS="${NGPUS:-8}"
TP_SIZE="${TP_SIZE:-4}"
SFT_NGPUS="${SFT_NGPUS:-$NGPUS}"
RL_NGPUS="${RL_NGPUS:-$NGPUS}"
RL_TP_SIZE="${RL_TP_SIZE:-$TP_SIZE}"

export EXPERIMENT_NAME TASK DATA_FILE SOURCE_KIND LEARNER_MODEL ROLLOUT_MODEL TOKENIZER
export NGPUS TP_SIZE SFT_NGPUS RL_NGPUS RL_TP_SIZE

# shellcheck source=scripts/budget-rl-data/budget_rl_existing_data_common.inc
source "$SCRIPT_DIR/budget_rl_existing_data_common.inc"
budget_rl_data_main "$@"
