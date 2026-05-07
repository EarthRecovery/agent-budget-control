#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME:-ragenv2}"
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$HOME/agent-budget-control"}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:$PWD/verl"

INPUT_SOURCE=${INPUT_SOURCE:-${INPUT_PATH:-${INPUT_DIR:-${INPUT_JSON:-"${HOME}/database/mini-swebench-origin/codex"}}}}
PROVIDER=${PROVIDER:-openai}
MODEL_NAME=${MODEL_NAME:-OpenAI-5.2-Codex-low-thinking}
MODEL_TAG=${MODEL_TAG:-$MODEL_NAME}
REASONING_EFFORT=${REASONING_EFFORT:-}
OPENROUTER_REASONING_ENABLED=${OPENROUTER_REASONING_ENABLED:-}
OPENROUTER_CACHE_ENABLED=${OPENROUTER_CACHE_ENABLED:-}
INPUT_TAG=${INPUT_TAG:-}
RUN_NAME=${RUN_NAME:-miniswebanch-${MODEL_NAME}-token-estimation}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT_ROOT/results/evaluation-scripts/eval"}
SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-"$SCRIPT_DIR/prompts/swebanch_estimation_system.txt"}
USER_PROMPT_FILE=${USER_PROMPT_FILE:-"$SCRIPT_DIR/prompts/swebanch_estimation_user.txt"}
TURN_USAGE_MODE=${TURN_USAGE_MODE:-turn_excluding_history}
REBUILD_INPUT_JSON=${REBUILD_INPUT_JSON:-0}
MAX_ROLLOUTS=${MAX_ROLLOUTS:-}

MAX_TURN=${MAX_TURN:-20}
MAX_CONTEXT_WINDOW_TOKENS=${MAX_CONTEXT_WINDOW_TOKENS:-}
MAX_SAMPLES=${MAX_SAMPLES:-128}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-8}
REQUEST_BATCH_SIZE=${REQUEST_BATCH_SIZE:-16}
MAX_TOKENS=${MAX_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-0.0}
ANTHROPIC_THINKING_ENABLED=${ANTHROPIC_THINKING_ENABLED:-0}
ANTHROPIC_THINKING_BUDGET_TOKENS=${ANTHROPIC_THINKING_BUDGET_TOKENS:-}
ANTHROPIC_THINKING_ADAPTIVE=${ANTHROPIC_THINKING_ADAPTIVE:-0}
ANTHROPIC_THINKING_DISPLAY=${ANTHROPIC_THINKING_DISPLAY:-}
ANTHROPIC_OUTPUT_EFFORT=${ANTHROPIC_OUTPUT_EFFORT:-}
DRY_RUN=${DRY_RUN:-0}

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
  OpenAI-5.2-Codex-low-thinking)
    PROVIDER=openai
    MODEL_NAME=gpt-5.2
    REASONING_EFFORT=${REASONING_EFFORT:-low}
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
  Claude-Opus-4.6-low-thinking)
    PROVIDER=anthropic
    MODEL_NAME=claude-opus-4-6
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

if [[ -z "$INPUT_SOURCE" ]]; then
  echo "INPUT_SOURCE is required. Pass a mini-SWE-bench directory such as ${HOME}/database/mini-swebench-origin/codex or an already converted rollout json." >&2
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

if [[ -d "$INPUT_SOURCE" ]]; then
  if [[ -z "$INPUT_TAG" ]]; then
    if [[ "$(basename "$INPUT_SOURCE")" == "trajs" ]]; then
      INPUT_TAG=$(basename "$(dirname "$INPUT_SOURCE")")
    else
      INPUT_TAG=$(basename "$INPUT_SOURCE")
    fi
  fi
else
  if [[ ! -f "$INPUT_SOURCE" ]]; then
    echo "Input source not found: $INPUT_SOURCE" >&2
    exit 1
  fi
  if [[ -z "$INPUT_TAG" ]]; then
    INPUT_TAG=$(basename "$INPUT_SOURCE")
    INPUT_TAG=${INPUT_TAG%.json}
  fi
fi

MODEL_TAG_SAFE=${MODEL_TAG//\//-}

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="miniswebanch-${INPUT_TAG}-${MODEL_TAG_SAFE}-token-estimation"
fi

OUTPUT_DIR=${OUTPUT_DIR:-"$RESULT_ROOT/${RUN_NAME}"}
OUTPUT_JSON=${OUTPUT_JSON:-"$OUTPUT_DIR/${RUN_NAME}.json"}
TEMP_JSON=${TEMP_JSON:-"$OUTPUT_DIR/${RUN_NAME}_pairs.json"}
GENERATED_INPUT_JSON=${GENERATED_INPUT_JSON:-"$OUTPUT_DIR/${INPUT_TAG}_eval_estimation_dialogues.json"}

mkdir -p "$OUTPUT_DIR"

if [[ -d "$INPUT_SOURCE" ]]; then
  if [[ "$REBUILD_INPUT_JSON" == "1" || ! -f "$GENERATED_INPUT_JSON" ]]; then
    PREPARE_CMD=(
      python scripts/budget-estimation-benchmark/prepare_miniswebench_dialogues.py
      --input-dir "$INPUT_SOURCE"
      --output-json "$GENERATED_INPUT_JSON"
    )
    if [[ -n "$MAX_ROLLOUTS" ]]; then
      PREPARE_CMD+=(--max-rollouts "$MAX_ROLLOUTS")
    fi
    echo "==> Preparing mini-SWE-bench rollout json from ${INPUT_SOURCE}"
    "${PREPARE_CMD[@]}"
  fi
  INPUT_JSON="$GENERATED_INPUT_JSON"
else
  INPUT_JSON="$INPUT_SOURCE"
fi

if [[ ! -f "$INPUT_JSON" ]]; then
  echo "Input json not found: $INPUT_JSON" >&2
  exit 1
fi

if [[ -z "${MAX_CONTEXT_WINDOW_TOKENS:-}" ]]; then
  MAX_CONTEXT_WINDOW_TOKENS=$(
    python - "$INPUT_SOURCE" "$INPUT_JSON" <<'PY'
import json
from pathlib import Path
import statistics
import sys

input_source = Path(sys.argv[1])
input_json = Path(sys.argv[2])

def rollout_totals_from_dialogue_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = [payload]
    totals = []
    for rollout in payload:
        turns = list(rollout.get("turns") or [])
        total_tokens = 0
        for turn in turns:
            value = turn.get("api_total_tokens")
            if value is None:
                input_tokens = turn.get("api_input_tokens") or 0
                output_tokens = turn.get("api_output_tokens") or 0
                value = int(input_tokens) + int(output_tokens)
            total_tokens += int(value or 0)
        totals.append(total_tokens)
    return totals

def rollout_totals_from_traj_dir(path: Path):
    if path.name == "trajs":
        traj_dir = path
    else:
        traj_dir = path / "trajs"
    traj_paths = sorted(traj_dir.glob("*/*.traj.json"))
    totals = []
    for traj_path in traj_paths:
        with traj_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        total_tokens = 0
        for message in list(payload.get("messages") or []):
            if str(message.get("object", "") or "") == "response":
                usage = dict(message.get("usage") or {})
            elif str(message.get("role", "") or "") == "assistant":
                usage = dict((((message.get("extra") or {}).get("response") or {}).get("usage") or {}))
            else:
                continue
            value = usage.get("total_tokens")
            if value is None:
                input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
                output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
                value = int(input_tokens) + int(output_tokens)
            total_tokens += int(value or 0)
        totals.append(total_tokens)
    return totals

if input_source.is_dir():
    rollout_totals = rollout_totals_from_traj_dir(input_source)
else:
    rollout_totals = rollout_totals_from_dialogue_json(input_json)

if not rollout_totals:
    raise SystemExit("No rollouts found when computing median trajectory length.")

median_value = statistics.median(rollout_totals)
if isinstance(median_value, float) and not median_value.is_integer():
    raise SystemExit(
        f"Median trajectory length is non-integer ({median_value}); "
        "please set MAX_CONTEXT_WINDOW_TOKENS manually."
    )

print(int(median_value))
PY
  )
fi

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
  --turn-usage-mode "$TURN_USAGE_MODE"
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

echo "==> Running mini-SWE-bench token estimation with API model"
echo "==> provider=${PROVIDER}"
echo "==> model=${MODEL_NAME}"
echo "==> input_source=${INPUT_SOURCE}"
echo "==> input_json=${INPUT_JSON}"
echo "==> max_context_window_tokens=${MAX_CONTEXT_WINDOW_TOKENS}"
echo "==> output_json=${OUTPUT_JSON}"

"${CMD[@]}"
