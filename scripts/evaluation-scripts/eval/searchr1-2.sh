#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate ragenv2
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$HOME/agent-budget-control"}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:$PWD/verl"

PROVIDER=${PROVIDER:-openrouter}
MODEL_NAME=${MODEL_NAME:-OpenRouter-Gemini-3.1-Pro-Preview}
REASONING_EFFORT=${REASONING_EFFORT:-}
OPENROUTER_REASONING_ENABLED=${OPENROUTER_REASONING_ENABLED:-}
OPENROUTER_CACHE_ENABLED=${OPENROUTER_CACHE_ENABLED:-}

case "$MODEL_NAME" in
  OpenAI-5.2-Instant)
    PROVIDER=openai
    MODEL_NAME=gpt-5.2
    REASONING_EFFORT=${REASONING_EFFORT:-none}
    ;;
  OpenAI-5.2-Codex-low-thinking)
    PROVIDER=openai
    MODEL_NAME=gpt-5.2
    REASONING_EFFORT=${REASONING_EFFORT:-low}
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
  Claude-Opus-4.6-low-thinking)
    PROVIDER=anthropic
    MODEL_NAME=claude-opus-4-6
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
  OpenRouter-Gemini-2.5-Pro|google/gemini-2.5-pro)
    PROVIDER=openrouter
    MODEL_NAME=google/gemini-2.5-pro
    REASONING_EFFORT=${REASONING_EFFORT:-low}
    OPENROUTER_CACHE_ENABLED=${OPENROUTER_CACHE_ENABLED:-1}
    ;;
  OpenRouter-Gemini-3.1-Pro-Preview)
    PROVIDER=openrouter
    MODEL_NAME=google/gemini-3.1-pro-preview
    REASONING_EFFORT=${REASONING_EFFORT:-low}
    OPENROUTER_CACHE_ENABLED=${OPENROUTER_CACHE_ENABLED:-1}
    ;;
  qwen/qwen3-235b-a22b-2507)
    PROVIDER=openrouter
    MODEL_NAME=qwen/qwen3-235b-a22b-2507
    REASONING_EFFORT=${REASONING_EFFORT:-low}
    OPENROUTER_CACHE_ENABLED=${OPENROUTER_CACHE_ENABLED:-1}
    ;;
  OpenRouter-DeepSeek-V3.2|deepseek/deepseek-v3.2)
    PROVIDER=openrouter
    MODEL_NAME=deepseek/deepseek-v3.2
    OPENROUTER_REASONING_ENABLED=${OPENROUTER_REASONING_ENABLED:-0}
    OPENROUTER_CACHE_ENABLED=${OPENROUTER_CACHE_ENABLED:-1}
    ;;
  OpenRouter-MiniMax-M2.5|minimax/minimax-m2.5)
    PROVIDER=openrouter
    MODEL_NAME=minimax/minimax-m2.5
    OPENROUTER_REASONING_ENABLED=${OPENROUTER_REASONING_ENABLED:-0}
    OPENROUTER_CACHE_ENABLED=${OPENROUTER_CACHE_ENABLED:-1}
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

RUN_NAME=${RUN_NAME:-searchr1-origin-openrouter-gemini-3-1-pro-preview-128-main_openrouter-gemini-3-1-pro-preview-main}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT_ROOT/results/evaluation-scripts/eval"}
OUTPUT_DIR=${OUTPUT_DIR:-"$RESULT_ROOT/${RUN_NAME}"}
OUTPUT_JSON=${OUTPUT_JSON:-"$OUTPUT_DIR/${RUN_NAME}.json"}
TEMP_JSON=${TEMP_JSON:-"$OUTPUT_DIR/${RUN_NAME}_pairs.json"}
SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-"$SCRIPT_DIR/prompts/searchr1_estimation_system.txt"}
USER_PROMPT_FILE=${USER_PROMPT_FILE:-"$SCRIPT_DIR/prompts/searchr1_estimation_user.txt"}

MAX_TURN=${MAX_TURN:-10}
MAX_CONTEXT_WINDOW_TOKENS=${MAX_CONTEXT_WINDOW_TOKENS:-3500}
MAX_SAMPLES=${MAX_SAMPLES:-}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-8}
REQUEST_BATCH_SIZE=${REQUEST_BATCH_SIZE:-32}
MAX_TOKENS=${MAX_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-0.0}
ANTHROPIC_THINKING_ENABLED=${ANTHROPIC_THINKING_ENABLED:-0}
ANTHROPIC_THINKING_BUDGET_TOKENS=${ANTHROPIC_THINKING_BUDGET_TOKENS:-}
ANTHROPIC_THINKING_ADAPTIVE=${ANTHROPIC_THINKING_ADAPTIVE:-0}
ANTHROPIC_THINKING_DISPLAY=${ANTHROPIC_THINKING_DISPLAY:-}
ANTHROPIC_OUTPUT_EFFORT=${ANTHROPIC_OUTPUT_EFFORT:-}
DRY_RUN=${DRY_RUN:-0}

DEFAULT_RESULT_INPUT_JSON="${HOME}/database/origin/searchr1-origin-gpt5.2-instant-128-main/search_r1_api_eval_estimation_eval_estimation_dialogues.json"
DEFAULT_TEST_INPUT_JSON="$PROJECT_ROOT/results/estimation/searchr1-origin-gpt5.2-instant-15-test/search_r1_api_eval_estimation_eval_estimation_dialogues.json"
DEFAULT_DATABASE_INPUT_JSON="${HOME}/database/origin/searchr1-origin-gpt5.2-instant-128-main/search_r1_api_eval_estimation_eval_estimation_dialogues.json"
INPUT_JSON=${INPUT_JSON:-"${HOME}/database/origin/searchr1-origin-openrouter-gemini-3-1-pro-preview-128-main/search_r1_api_eval_estimation_eval_estimation_dialogues.json"}
if [[ -z "$INPUT_JSON" ]]; then
  if [[ -f "$DEFAULT_RESULT_INPUT_JSON" ]]; then
    INPUT_JSON="$DEFAULT_RESULT_INPUT_JSON"
  elif [[ -f "$DEFAULT_TEST_INPUT_JSON" ]]; then
    INPUT_JSON="$DEFAULT_TEST_INPUT_JSON"
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
  --max-turn "$MAX_TURN"
  --max-context-window-tokens "$MAX_CONTEXT_WINDOW_TOKENS"
  --system-prompt-file "$SYSTEM_PROMPT_FILE"
  --user-prompt-file "$USER_PROMPT_FILE"
)

SHOULD_PASS_TEMPERATURE=1
if [[ "$PROVIDER" == "openai" ]]; then
  case "$MODEL_NAME" in
    gpt-5*|o*)
      SHOULD_PASS_TEMPERATURE=0
      ;;
  esac
fi

if [[ "$SHOULD_PASS_TEMPERATURE" == "1" && -n "${TEMPERATURE:-}" ]]; then
  CMD+=(--temperature "$TEMPERATURE")
fi

if [[ -n "$REASONING_EFFORT" ]]; then
  CMD+=(--reasoning-effort "$REASONING_EFFORT")
fi

if [[ -n "$OPENROUTER_REASONING_ENABLED" ]]; then
  CMD+=(--reasoning-enabled "$OPENROUTER_REASONING_ENABLED")
fi

case "${OPENROUTER_CACHE_ENABLED,,}" in
  1|true|yes|on)
    CMD+=(--cache-enabled)
    ;;
esac

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

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

"${CMD[@]}"
