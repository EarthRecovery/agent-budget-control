#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate ragenv2
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$HOME/agent-budget-control"}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:$PWD/verl"

PROVIDER=${PROVIDER:-openai}
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
esac

case "$PROVIDER" in
  openai)
    : "${OPENAI_API_KEY:?Please export OPENAI_API_KEY before running this evaluation.}"
    ;;
  anthropic)
    : "${ANTHROPIC_API_KEY:?Please export ANTHROPIC_API_KEY before running this evaluation.}"
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

RUN_NAME=${RUN_NAME:-sokoban-origin-gpt5.2-instant-128-main_gpt5.2-instant-token-estimation-20-test}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT_ROOT/results/evaluation-scripts/eval"}
OUTPUT_DIR=${OUTPUT_DIR:-"$RESULT_ROOT/${RUN_NAME}"}
INPUT_JSON=${INPUT_JSON:-"/u/ylin30/database/origin/sokoban-origin-gpt5.2-instant-128-main/sokoban_api_eval_estimation_eval_estimation_dialogues.json"}
OUTPUT_JSON=${OUTPUT_JSON:-"$OUTPUT_DIR/${RUN_NAME}.json"}
TEMP_JSON=${TEMP_JSON:-"$OUTPUT_DIR/${RUN_NAME}_pairs.json"}
SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-"$SCRIPT_DIR/prompts/sokoban_estimation_system.txt"}
USER_PROMPT_FILE=${USER_PROMPT_FILE:-"$SCRIPT_DIR/prompts/sokoban_estimation_user.txt"}

MAX_TURN=${MAX_TURN:-1}
MAX_CONTEXT_WINDOW_TOKENS=${MAX_CONTEXT_WINDOW_TOKENS:-2500}
MAX_SAMPLES=${MAX_SAMPLES:-20}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-8}
REQUEST_BATCH_SIZE=${REQUEST_BATCH_SIZE:-32}
MAX_TOKENS=${MAX_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-0.0}
DRY_RUN=${DRY_RUN:-0}

DEFAULT_RESULT_INPUT_JSON="$PROJECT_ROOT/results/estimation/sokoban-origin-gpt5.2-instant-128-main/sokoban_api_eval_estimation_eval_estimation_dialogues.json"
DEFAULT_BENCHMARK_INPUT_JSON="$PROJECT_ROOT/results/budget-estimation-benchmark/sokoban-origin-gpt5.2-instant-128-window=1-max-turn=6/sokoban_api_eval_estimation_eval_estimation_dialogues.json"
DEFAULT_DATABASE_INPUT_JSON="/u/ylin30/database/origin/sokoban-origin-gpt5.2-instant-128-main/sokoban_api_eval_estimation_eval_estimation_dialogues.json"
if [[ -z "${INPUT_JSON:-}" ]]; then
  if [[ -f "$DEFAULT_RESULT_INPUT_JSON" ]]; then
    INPUT_JSON="$DEFAULT_RESULT_INPUT_JSON"
  elif [[ -f "$DEFAULT_BENCHMARK_INPUT_JSON" ]]; then
    INPUT_JSON="$DEFAULT_BENCHMARK_INPUT_JSON"
  else
    INPUT_JSON="$DEFAULT_DATABASE_INPUT_JSON"
  fi
fi

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
  python scripts/budget-estimation-benchmark/run_token_estimation_env.py
  --input-json "$INPUT_JSON"
  --output-json "$OUTPUT_JSON"
  --temp-json "$TEMP_JSON"
  --provider "$PROVIDER"
  --model "$MODEL_NAME"
  --max-concurrency "$MAX_CONCURRENCY"
  --request-batch-size "$REQUEST_BATCH_SIZE"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --max-turn "$MAX_TURN"
  --max-context-window-tokens "$MAX_CONTEXT_WINDOW_TOKENS"
  --system-prompt-file "$SYSTEM_PROMPT_FILE"
  --user-prompt-file "$USER_PROMPT_FILE"
)

if [[ -n "$REASONING_EFFORT" ]]; then
  CMD+=(--reasoning-effort "$REASONING_EFFORT")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

"${CMD[@]}"
