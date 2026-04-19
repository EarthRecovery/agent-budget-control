#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate ragenv2
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$HOME/agent-budget-control"}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:$PWD/verl"

MODEL_NAME=${MODEL_NAME:-OpenAI-5.2-Instant}
REASONING_EFFORT=${REASONING_EFFORT:-}

case "$MODEL_NAME" in
  OpenAI-5.2-Instant)
    PROVIDER=openai
    MODEL_NAME=gpt-5.2
    REASONING_EFFORT=${REASONING_EFFORT:-none}
    ;;
  OpenAI-5.2-Thinking)
    PROVIDER=openai
    MODEL_NAME=gpt-5.2
    REASONING_EFFORT=${REASONING_EFFORT:-high}
    ;;
  Claude-Opus-Latest)
    PROVIDER=anthropic
    MODEL_NAME=claude-opus-4-7
    ;;
  Claude-Sonnet-Latest)
    PROVIDER=anthropic
    MODEL_NAME=claude-sonnet-4-6
    ;;
  Claude-Opus-4.7-low-thinking)
    PROVIDER=anthropic
    MODEL_NAME=claude-opus-4-7
    ANTHROPIC_OUTPUT_EFFORT=${ANTHROPIC_OUTPUT_EFFORT:-low}
    ;;
  Claude-Sonnet-4.6-low-thinking)
    PROVIDER=anthropic
    MODEL_NAME=claude-sonnet-4-6
    ANTHROPIC_OUTPUT_EFFORT=${ANTHROPIC_OUTPUT_EFFORT:-low}
    ;;
  Gemini-Pro-Latest)
    PROVIDER=gemini
    MODEL_NAME=gemini-2.5-pro
    ;;
  OpenRouter-Qwen3.6-Plus)
    PROVIDER=openrouter
    MODEL_NAME=qwen/qwen3.6-plus
    ;;
esac

case "$PROVIDER" in
  openai)
    : "${OPENAI_API_KEY:?Please export OPENAI_API_KEY before running this evaluation.}"
    ;;
  anthropic)
    : "${ANTHROPIC_API_KEY:?Please export ANTHROPIC_API_KEY before running this evaluation.}"
    ;;
  gemini)
    : "${GEMINI_API_KEY:?Please export GEMINI_API_KEY before running this evaluation.}"
    ;;
  openrouter)
    : "${OPENROUTER_API_KEY:?Please export OPENROUTER_API_KEY before running this evaluation.}"
    ;;
  together)
    : "${TOGETHER_API_KEY:?Please export TOGETHER_API_KEY before running this evaluation.}"
    ;;
  deepseek)
    : "${DEEPSEEK_API_KEY:?Please export DEEPSEEK_API_KEY before running this evaluation.}"
    ;;
  *)
    echo "Unsupported PROVIDER: $PROVIDER" >&2
    exit 1
    ;;
esac

RUN_NAME=${RUN_NAME:-warehouse-OpenAI-5.2-Instant_OpenAI-5.2-Instant-128-main}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT_ROOT/results/evaluation-scripts/eval"}
OUTPUT_DIR=${OUTPUT_DIR:-"$RESULT_ROOT/${RUN_NAME}"}
OUTPUT_JSON=${OUTPUT_JSON:-"$OUTPUT_DIR/${RUN_NAME}.json"}
TEMP_JSON=${TEMP_JSON:-"$OUTPUT_DIR/${RUN_NAME}_pairs.json"}
SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-"$SCRIPT_DIR/prompts/warehouse_estimation_system.txt"}
USER_PROMPT_FILE=${USER_PROMPT_FILE:-"$SCRIPT_DIR/prompts/warehouse_estimation_user.txt"}

MAX_SAMPLES=${MAX_SAMPLES:-}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-8}
REQUEST_BATCH_SIZE=${REQUEST_BATCH_SIZE:-32}
MAX_TOKENS=${MAX_TOKENS:-768}
TEMPERATURE=${TEMPERATURE:-0.0}
ANTHROPIC_THINKING_ENABLED=${ANTHROPIC_THINKING_ENABLED:-0}
ANTHROPIC_THINKING_BUDGET_TOKENS=${ANTHROPIC_THINKING_BUDGET_TOKENS:-}
ANTHROPIC_THINKING_ADAPTIVE=${ANTHROPIC_THINKING_ADAPTIVE:-0}
ANTHROPIC_THINKING_DISPLAY=${ANTHROPIC_THINKING_DISPLAY:-}
ANTHROPIC_OUTPUT_EFFORT=${ANTHROPIC_OUTPUT_EFFORT:-}
DRY_RUN=${DRY_RUN:-0}

# Budget controls for the money_estimation evaluator.
# Precedence rule:
# 1. If an absolute budget is set, it overrides the matching *_RATIO value.
# 2. Otherwise the runner derives the budget from each traj's final-turn value
#    multiplied by the selected ratio.
#
# BUDGET_PRESET=per_traj_final_turn_reachable makes the default question:
#   "From a partial prefix of each traj, can the rollout still reach a slightly
#   easier target cash within time equal to the realized rollout length and
#   warehouse / cost budgets that are slightly looser than the realized final
#   totals?"
# This keeps the realized final turn budget-feasible for successful rollouts.
#
# BUDGET_PRESET=half-reachable keeps the same auto-derived time / warehouse /
# cost budgets, but ignores AUTO_TARGET_CASH_RATIO and instead perturbs each
# rollout target cash so that roughly half of valid rollouts have an
# unreachable target.
BUDGET_PRESET=${BUDGET_PRESET:-half-reachable}
AUTO_TARGET_CASH_RATIO=${AUTO_TARGET_CASH_RATIO:-0.98}
AUTO_TIME_BUDGET_RATIO=${AUTO_TIME_BUDGET_RATIO:-1.0}
AUTO_WAREHOUSE_BUDGET_RATIO=${AUTO_WAREHOUSE_BUDGET_RATIO:-1.02}
AUTO_COST_BUDGET_RATIO=${AUTO_COST_BUDGET_RATIO:-1.02}
HALF_REACHABLE_TARGET_SEED=${HALF_REACHABLE_TARGET_SEED:-42}
#
# TARGET_CASH_*:
#   Minimum final cash required for the rollout to count as successful.
TARGET_CASH_USD=${TARGET_CASH_USD-}
TARGET_CASH_RATIO=${TARGET_CASH_RATIO-}
#
# TIME_BUDGET_*:
#   Maximum total elapsed time allowed, measured in weeks.
TIME_BUDGET_WEEKS=${TIME_BUDGET_WEEKS-}
TIME_BUDGET_RATIO=${TIME_BUDGET_RATIO-}
#
# WAREHOUSE_BUDGET_*:
#   Maximum cumulative warehouse occupancy, measured in item-weeks.
WAREHOUSE_BUDGET_ITEM_WEEKS=${WAREHOUSE_BUDGET_ITEM_WEEKS-}
WAREHOUSE_BUDGET_RATIO=${WAREHOUSE_BUDGET_RATIO-}
#
# COST_BUDGET_*:
#   Maximum cumulative spending, measured in USD. Revenue does not reduce
#   this metric because it tracks cumulative cost rather than net profit.
COST_BUDGET_USD=${COST_BUDGET_USD-}
COST_BUDGET_RATIO=${COST_BUDGET_RATIO-}

TARGET_CASH_MODE=ratio

if [[ "$BUDGET_PRESET" == "per_traj_final_turn_reachable" ]]; then
  TARGET_CASH_RATIO=${TARGET_CASH_RATIO:-$AUTO_TARGET_CASH_RATIO}
  TIME_BUDGET_RATIO=${TIME_BUDGET_RATIO:-$AUTO_TIME_BUDGET_RATIO}
  WAREHOUSE_BUDGET_RATIO=${WAREHOUSE_BUDGET_RATIO:-$AUTO_WAREHOUSE_BUDGET_RATIO}
  COST_BUDGET_RATIO=${COST_BUDGET_RATIO:-$AUTO_COST_BUDGET_RATIO}
elif [[ "$BUDGET_PRESET" == "half-reachable" ]]; then
  TARGET_CASH_MODE=half_reachable
  TIME_BUDGET_RATIO=${TIME_BUDGET_RATIO:-$AUTO_TIME_BUDGET_RATIO}
  WAREHOUSE_BUDGET_RATIO=${WAREHOUSE_BUDGET_RATIO:-$AUTO_WAREHOUSE_BUDGET_RATIO}
  COST_BUDGET_RATIO=${COST_BUDGET_RATIO:-$AUTO_COST_BUDGET_RATIO}
else
  TARGET_CASH_RATIO=${TARGET_CASH_RATIO:-1.0}
  TIME_BUDGET_RATIO=${TIME_BUDGET_RATIO:-1.0}
  WAREHOUSE_BUDGET_RATIO=${WAREHOUSE_BUDGET_RATIO:-1.0}
  COST_BUDGET_RATIO=${COST_BUDGET_RATIO:-1.0}
fi

DEFAULT_INPUT_JSON="/u/ylin30/database/origin/warehouse-origin-gpt5.2-instant-128-main/combined_gpt5.2-chat-latest_128seeds.json"
INPUT_JSON=${INPUT_JSON:-"$DEFAULT_INPUT_JSON"}

if [[ ! -f "$INPUT_JSON" ]]; then
  echo "Input json not found: $INPUT_JSON" >&2
  exit 1
fi

if [[ ! -f "$SYSTEM_PROMPT_FILE" ]]; then
  echo "System prompt file not found: $SYSTEM_PROMPT_FILE" >&2
  exit 1
fi

if [[ ! -f "$USER_PROMPT_FILE" ]]; then
  echo "User prompt file not found: $USER_PROMPT_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

CMD=(
  python scripts/budget-estimation-benchmark/run_money_estimation_env.py
  --input-json "$INPUT_JSON"
  --output-json "$OUTPUT_JSON"
  --temp-json "$TEMP_JSON"
  --provider "$PROVIDER"
  --model "$MODEL_NAME"
  --max-concurrency "$MAX_CONCURRENCY"
  --request-batch-size "$REQUEST_BATCH_SIZE"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --target-cash-mode "$TARGET_CASH_MODE"
  --target-cash-half-reachable-seed "$HALF_REACHABLE_TARGET_SEED"
  --time-budget-ratio "$TIME_BUDGET_RATIO"
  --warehouse-budget-ratio "$WAREHOUSE_BUDGET_RATIO"
  --cost-budget-ratio "$COST_BUDGET_RATIO"
  --system-prompt-file "$SYSTEM_PROMPT_FILE"
  --user-prompt-file "$USER_PROMPT_FILE"
)

if [[ -n "$REASONING_EFFORT" ]]; then
  CMD+=(--reasoning-effort "$REASONING_EFFORT")
fi

if [[ "$ANTHROPIC_THINKING_ENABLED" == "1" ]]; then
  CMD+=(--thinking-enabled)
fi

if [[ -n "$ANTHROPIC_THINKING_BUDGET_TOKENS" ]]; then
  CMD+=(--thinking-budget-tokens "$ANTHROPIC_THINKING_BUDGET_TOKENS")
fi

if [[ "$ANTHROPIC_THINKING_ADAPTIVE" == "1" ]]; then
  CMD+=(--thinking-adaptive)
fi

if [[ -n "$ANTHROPIC_THINKING_DISPLAY" ]]; then
  CMD+=(--thinking-display "$ANTHROPIC_THINKING_DISPLAY")
fi

if [[ -n "$ANTHROPIC_OUTPUT_EFFORT" ]]; then
  CMD+=(--output-effort "$ANTHROPIC_OUTPUT_EFFORT")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "$TARGET_CASH_MODE" == "ratio" ]]; then
  CMD+=(--target-cash-ratio "$TARGET_CASH_RATIO")
fi

if [[ -n "$TARGET_CASH_USD" && "$TARGET_CASH_MODE" == "ratio" ]]; then
  CMD+=(--target-cash-usd "$TARGET_CASH_USD")
fi

if [[ -n "$TIME_BUDGET_WEEKS" ]]; then
  CMD+=(--time-budget-weeks "$TIME_BUDGET_WEEKS")
fi

if [[ -n "$WAREHOUSE_BUDGET_ITEM_WEEKS" ]]; then
  CMD+=(--warehouse-budget-item-weeks "$WAREHOUSE_BUDGET_ITEM_WEEKS")
fi

if [[ -n "$COST_BUDGET_USD" ]]; then
  CMD+=(--cost-budget-usd "$COST_BUDGET_USD")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

"${CMD[@]}"
