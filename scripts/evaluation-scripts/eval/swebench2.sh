#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME:-ragenv2}"
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$HOME/agent-budget-control"}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:$PWD/verl"

INPUT_SOURCE=${INPUT_SOURCE:-${INPUT_PATH:-${INPUT_DIR:-"${HOME}/database/origin/swebench-origin-claude-sonnet4.6"}}}
PROVIDER=${PROVIDER:-anthropic}
MODEL_NAME=${MODEL_NAME:-Claude-Sonnet-4.6-low-thinking}
MODEL_TAG=${MODEL_TAG:-swebench-origin-claude-sonnet4.6_claude-sonnet-4.6-512-main}
REASONING_EFFORT=${REASONING_EFFORT:-}
OPENROUTER_REASONING_ENABLED=${OPENROUTER_REASONING_ENABLED:-}
OPENROUTER_CACHE_ENABLED=${OPENROUTER_CACHE_ENABLED:-}
INPUT_TAG=${INPUT_TAG:-}
RUN_NAME=${RUN_NAME:-swebench-origin-claude-sonnet4.6_claude-sonnet-4.6-512-main}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT_ROOT/results/evaluation-scripts/eval"}
SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-"$SCRIPT_DIR/prompts/swebanch_estimation_system.txt"}
USER_PROMPT_FILE=${USER_PROMPT_FILE:-"$SCRIPT_DIR/prompts/swebanch_estimation_user.txt"}
TURN_USAGE_MODE=${TURN_USAGE_MODE:-turn_excluding_history}
REBUILD_INPUT_JSON=${REBUILD_INPUT_JSON:-1}
MAX_ROLLOUTS=${MAX_ROLLOUTS:-}
BUDGET_FROM_ALL_ROLLOUTS=${BUDGET_FROM_ALL_ROLLOUTS:-1}

MAX_TURN=${MAX_TURN:-160}
MAX_CONTEXT_WINDOW_TOKENS=${MAX_CONTEXT_WINDOW_TOKENS:-}
MAX_SAMPLES=${MAX_SAMPLES:-512}
SAMPLE_SELECTION=${SAMPLE_SELECTION:-fair_split_random}
SAMPLE_SELECTION_SEED=${SAMPLE_SELECTION_SEED:-42}
SAMPLE_LENGTH_BINS=${SAMPLE_LENGTH_BINS:-8}
CACHE_FRIENDLY_ORDER=${CACHE_FRIENDLY_ORDER:-1}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-1}
REQUEST_BATCH_SIZE=${REQUEST_BATCH_SIZE:-1}
MAX_TOKENS=${MAX_TOKENS:-512}
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
  Claude-Sonnet-4.6-low-thinking)
    PROVIDER=anthropic
    MODEL_NAME=claude-sonnet-4-6
    ANTHROPIC_OUTPUT_EFFORT=${ANTHROPIC_OUTPUT_EFFORT:-low}
    ;;
  OpenRouter-Qwen3.6-Plus)
    PROVIDER=openrouter
    MODEL_NAME=qwen/qwen3.6-plus
    OPENROUTER_CACHE_ENABLED=${OPENROUTER_CACHE_ENABLED:-1}
    ;;
  OpenRouter-DeepSeek-V3.2|deepseek/deepseek-v3.2)
    PROVIDER=openrouter
    MODEL_NAME=deepseek/deepseek-v3.2
    OPENROUTER_REASONING_ENABLED=${OPENROUTER_REASONING_ENABLED:-0}
    OPENROUTER_CACHE_ENABLED=${OPENROUTER_CACHE_ENABLED:-1}
    ;;
  qwen/qwen3-235b-a22b-2507)
    PROVIDER=openrouter
    MODEL_NAME=qwen/qwen3-235b-a22b-2507
    REASONING_EFFORT=${REASONING_EFFORT:-low}
    OPENROUTER_CACHE_ENABLED=${OPENROUTER_CACHE_ENABLED:-1}
    ;;
  OpenRouter-MiniMax-M2.5|minimax/minimax-m2.5)
    PROVIDER=openrouter
    MODEL_NAME=minimax/minimax-m2.5
    OPENROUTER_REASONING_ENABLED=${OPENROUTER_REASONING_ENABLED:-0}
    OPENROUTER_CACHE_ENABLED=${OPENROUTER_CACHE_ENABLED:-1}
    ;;
esac

if [[ -z "$MODEL_TAG" ]]; then
  MODEL_TAG="$MODEL_NAME"
fi

if [[ "$DRY_RUN" != "1" ]]; then
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
fi

if [[ -z "$INPUT_SOURCE" ]]; then
  echo "INPUT_SOURCE is required. Pass a SWE-bench origin directory such as ${HOME}/database/origin/swebench-origin-claude-sonnet4.6 or a converted rollout json." >&2
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
    INPUT_TAG=$(basename "$INPUT_SOURCE")
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
  RUN_NAME="swebanch-${INPUT_TAG}-${MODEL_TAG_SAFE}-token-estimation"
fi

OUTPUT_DIR=${OUTPUT_DIR:-"$RESULT_ROOT/${RUN_NAME}"}
OUTPUT_JSON=${OUTPUT_JSON:-"$OUTPUT_DIR/${RUN_NAME}.json"}
TEMP_JSON=${TEMP_JSON:-"$OUTPUT_DIR/${RUN_NAME}_pairs.json"}
GENERATED_INPUT_JSON=${GENERATED_INPUT_JSON:-"$OUTPUT_DIR/${INPUT_TAG}_eval_estimation_dialogues.json"}

mkdir -p "$OUTPUT_DIR"

if [[ -d "$INPUT_SOURCE" ]]; then
  if [[ "$REBUILD_INPUT_JSON" == "1" || ! -f "$GENERATED_INPUT_JSON" ]]; then
    PREPARE_CMD=(
      python scripts/budget-estimation-benchmark/prepare_swebench_dialogues.py
      --input-dir "$INPUT_SOURCE"
      --output-json "$GENERATED_INPUT_JSON"
    )
    if [[ -n "$MAX_ROLLOUTS" ]]; then
      PREPARE_CMD+=(--max-rollouts "$MAX_ROLLOUTS")
    fi
    echo "==> Preparing SWE-bench rollout json from ${INPUT_SOURCE}"
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
  BUDGET_SOURCE="$INPUT_JSON"
  if [[ -d "$INPUT_SOURCE" && "$BUDGET_FROM_ALL_ROLLOUTS" == "1" ]]; then
    BUDGET_SOURCE="$INPUT_SOURCE"
  fi
  MAX_CONTEXT_WINDOW_TOKENS=$(
    python - "$BUDGET_SOURCE" "$TURN_USAGE_MODE" "$PROJECT_ROOT" <<'PY'
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import statistics
import sys

target = Path(sys.argv[1]).expanduser().resolve()
turn_usage_mode = str(sys.argv[2]).strip().lower()
project_root = Path(sys.argv[3]).expanduser().resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "verl"))

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from ragen.env.token_estimation.env import (
        _build_cumulative_totals_from_details,
        _convert_request_usage_to_turn_usage,
        _estimate_turn_visible_output_tokens,
        _has_turn_assistant_content,
        _infer_usage_mode_from_rollout,
        _is_context_token_truncated_turn,
        _normalize_turn_usage_details,
        _normalize_usage_detail,
        _resolve_turn_token_usage_detail,
    )


def load_prepare_module(root: Path):
    module_path = root / "scripts" / "budget-estimation-benchmark" / "prepare_swebench_dialogues.py"
    spec = importlib.util.spec_from_file_location("prepare_swebench_dialogues", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def load_rollouts_from_dir(path: Path):
    prepare_module = load_prepare_module(project_root)
    traj_files = prepare_module._collect_traj_files(path)
    preds = prepare_module._load_preds(path)
    rollouts = []
    for env_index, traj_path in enumerate(traj_files):
        rollout = prepare_module._convert_traj_to_rollout(
            traj_path,
            env_index=env_index,
            preds=preds,
        )
        if rollout is not None:
            rollouts.append(rollout)
    return rollouts


def load_rollouts_from_dialogue_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = [payload]
    return list(payload)


def compute_rollout_total_tokens(rollout):
    sorted_raw_turns = sorted(
        list(rollout.get("turns") or []),
        key=lambda turn: int(turn.get("turn_idx", 0) or 0),
    )
    if not sorted_raw_turns:
        return None

    raw_turns = []
    for turn in sorted_raw_turns:
        if _is_context_token_truncated_turn(turn):
            break
        raw_turns.append(turn)

    turns = [turn for turn in raw_turns if _has_turn_assistant_content(turn)]
    if not turns:
        return None

    request_token_usage_details = [
        _normalize_usage_detail(_resolve_turn_token_usage_detail(turn))
        for turn in turns
    ]
    visible_output_tokens = [
        _estimate_turn_visible_output_tokens(turn) for turn in turns
    ]
    local_turn_token_usage_details = _convert_request_usage_to_turn_usage(
        request_token_usage_details,
        visible_output_tokens=visible_output_tokens,
    )
    assume_cumulative = _infer_usage_mode_from_rollout(
        rollout,
        request_token_usage_details,
    )
    budget_token_usage_details, cumulative_turn_totals, _ = _normalize_turn_usage_details(
        request_token_usage_details,
        assume_cumulative=assume_cumulative,
    )

    if turn_usage_mode == "turn_excluding_history":
        actual_budget_usage_details = local_turn_token_usage_details
        actual_budget_cumulative_totals = _build_cumulative_totals_from_details(
            actual_budget_usage_details
        )
    else:
        actual_budget_usage_details = budget_token_usage_details
        actual_budget_cumulative_totals = list(cumulative_turn_totals)

    if actual_budget_cumulative_totals and all(value is not None for value in actual_budget_cumulative_totals):
        return int(actual_budget_cumulative_totals[-1])

    total_tokens = 0
    saw_any = False
    for detail in actual_budget_usage_details:
        value = detail.get("total_tokens")
        if value is None:
            continue
        total_tokens += int(value)
        saw_any = True
    return int(total_tokens) if saw_any else None

if target.is_dir():
    rollouts = load_rollouts_from_dir(target)
else:
    rollouts = load_rollouts_from_dialogue_json(target)

rollout_totals = [
    int(total_tokens)
    for total_tokens in (
        compute_rollout_total_tokens(rollout) for rollout in rollouts
    )
    if total_tokens is not None
]

if not rollout_totals:
    raise SystemExit("No rollouts found when computing the SWE-bench median token budget.")

median_value = statistics.median(rollout_totals)
if isinstance(median_value, float) and not median_value.is_integer():
    median_value = statistics.median_low(rollout_totals)

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
  --sample-selection "$SAMPLE_SELECTION"
  --sample-selection-seed "$SAMPLE_SELECTION_SEED"
  --sample-length-bins "$SAMPLE_LENGTH_BINS"
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

case "${CACHE_FRIENDLY_ORDER,,}" in
  1|true|yes|on)
    CMD+=(--cache-friendly-order)
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

echo "==> Running SWE-bench token estimation with API model"
echo "==> provider=${PROVIDER}"
echo "==> model=${MODEL_NAME}"
echo "==> input_source=${INPUT_SOURCE}"
echo "==> input_json=${INPUT_JSON}"
echo "==> max_context_window_tokens=${MAX_CONTEXT_WINDOW_TOKENS}"
echo "==> max_samples=${MAX_SAMPLES:-all}"
echo "==> sample_selection=${SAMPLE_SELECTION}"
echo "==> cache_enabled=${OPENROUTER_CACHE_ENABLED:-0}, cache_friendly_order=${CACHE_FRIENDLY_ORDER}"
echo "==> output_json=${OUTPUT_JSON}"

"${CMD[@]}"
