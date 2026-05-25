#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-warehouse_qwen3_06b_sliding_window_test}"
TASK="${TASK:-warehouse}"
if [[ -z "${DATA_FILE:-}" ]]; then
  if [[ -f "$SCRIPT_DIR/warehouse_gpt5.2instant.json" ]]; then
    DATA_FILE="$SCRIPT_DIR/warehouse_gpt5.2instant.json"
  else
    DATA_FILE="$SCRIPT_DIR/warehouse_gpt5.2.json"
  fi
fi
SOURCE_KIND="estimation_json"
LEARNER_MODEL="${LEARNER_MODEL:-Qwen/Qwen3-0.6B}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-GPT5.2 instant}"
TOKENIZER="${TOKENIZER:-$LEARNER_MODEL}"
SYSTEM_PROMPT_EXTRA="${SYSTEM_PROMPT_EXTRA:-the rollout is generated from GPT5.2 instant}"
STAGES="${STAGES:-prepare,rl}"
RL_INIT_MODEL="${RL_INIT_MODEL:-$LEARNER_MODEL}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/budget-rl/from-existing-data}"
EXP_BASE="${EXP_BASE:-$DATA_ROOT/$EXPERIMENT_NAME}"
SFT_ABLATION="${SFT_ABLATION:-no_sft}"
VERIFY_RL_SLIDING_WINDOW="${VERIFY_RL_SLIDING_WINDOW:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TARGET_UNIT="${TARGET_UNIT:-cost}"
COST_KEY="${COST_KEY:-financial_cost}"
MAX_COST="${MAX_COST:-1700000}"
COST_DECIMALS="${COST_DECIMALS:-2}"
BUDGET_PROBE_CONTEXT_WINDOW_MODE="${BUDGET_PROBE_CONTEXT_WINDOW_MODE:-limited_multi_turn}"
BUDGET_PROBE_MAX_CONTEXT_WINDOW="${BUDGET_PROBE_MAX_CONTEXT_WINDOW:-3}"
BUDGET_PROBE_MAX_PROMPT_TOKENS="${BUDGET_PROBE_MAX_PROMPT_TOKENS:-8192}"
BUDGET_PROBE_DROP_OVERLONG_PROMPTS="${BUDGET_PROBE_DROP_OVERLONG_PROMPTS:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"

NGPUS="${NGPUS:-1}"
TP_SIZE="${TP_SIZE:-1}"
SFT_NGPUS="${SFT_NGPUS:-$NGPUS}"
RL_NGPUS="${RL_NGPUS:-$NGPUS}"
RL_TP_SIZE="${RL_TP_SIZE:-$TP_SIZE}"

append_arg_once() {
  local var_name="$1"
  local key="$2"
  local arg="$3"
  local current="${!var_name:-}"

  case " $current " in
    *" $key="*|*" +$key="*)
      ;;
    *)
      printf -v "$var_name" '%s%s%s' "$current" "${current:+ }" "$arg"
      export "$var_name"
      ;;
  esac
}

if [[ "${DISABLE_FLASH_ATTN_WORKAROUND:-0}" != "1" ]]; then
  append_arg_once RL_EXTRA_ARGS \
    "actor_rollout_ref.model.override_config.attn_implementation" \
    "+actor_rollout_ref.model.override_config.attn_implementation=$ATTN_IMPLEMENTATION"
  append_arg_once RL_EXTRA_ARGS \
    "actor_rollout_ref.model.use_remove_padding" \
    "actor_rollout_ref.model.use_remove_padding=$USE_REMOVE_PADDING"
fi

export EXPERIMENT_NAME TASK DATA_FILE SOURCE_KIND LEARNER_MODEL ROLLOUT_MODEL TOKENIZER
export SYSTEM_PROMPT_EXTRA STAGES RL_INIT_MODEL DATA_ROOT EXP_BASE SFT_ABLATION MAX_TOKENS
export TARGET_UNIT COST_KEY MAX_COST COST_DECIMALS
export ATTN_IMPLEMENTATION USE_REMOVE_PADDING RL_EXTRA_ARGS
export BUDGET_PROBE_CONTEXT_WINDOW_MODE BUDGET_PROBE_MAX_CONTEXT_WINDOW
export BUDGET_PROBE_MAX_PROMPT_TOKENS BUDGET_PROBE_DROP_OVERLONG_PROMPTS
export NGPUS TP_SIZE SFT_NGPUS RL_NGPUS RL_TP_SIZE

# shellcheck source=scripts/budget-rl-data/budget_rl_existing_data_common.inc
source "$SCRIPT_DIR/budget_rl_existing_data_common.inc"
budget_rl_data_main "$@"

stages_text="${1:-$STAGES}"
if [[ "${VERIFY_RL_SLIDING_WINDOW}" = "1" && "${DRY_RUN:-0}" != "1" ]] &&
   [[ "$stages_text" = "all" || ",${stages_text// /,}," == *",prepare,"* ]]; then
  python3 - "$EXP_BASE" "$BUDGET_PROBE_MAX_CONTEXT_WINDOW" <<'PY'
import os
import re
import sys

import datasets

exp_base = sys.argv[1]
max_window = int(sys.argv[2])

candidates = [
    os.path.join(exp_base, "rl", "train.parquet"),
    os.path.join(exp_base, "rl", "test.parquet"),
    os.path.join(exp_base, "eval_test", "train.parquet"),
]

dataset = None
source_path = None
errors = []
for path in candidates:
    if not os.path.exists(path):
        continue
    try:
        candidate = datasets.load_dataset("parquet", data_files=path)["train"]
    except Exception as exc:
        errors.append(f"{path}: {type(exc).__name__}: {exc}")
        continue
    if len(candidate) == 0:
        errors.append(f"{path}: empty dataset")
        continue
    dataset = candidate
    source_path = path
    break

if dataset is None:
    print("Could not find a non-empty RL-format parquet to verify.", file=sys.stderr)
    for error in errors:
        print(f"  {error}", file=sys.stderr)
    raise SystemExit(2)

inspected = 0
eligible_for_window = 0
window_capped = 0
max_visible_assistant_turns = 0
violations = []

for row in dataset:
    extra_info = row.get("extra_info") or {}
    custom_id = extra_info.get("custom_id", "")
    match = re.search(r"_probe_after_turn_(\d+)$", custom_id)
    if not match:
        continue

    probe_after = int(match.group(1))
    prompt = row.get("prompt") or []
    visible_assistant_turns = sum(
        1 for message in prompt if message.get("role") == "assistant"
    )
    expected_max = probe_after if max_window < 0 else min(probe_after, max_window)

    inspected += 1
    max_visible_assistant_turns = max(max_visible_assistant_turns, visible_assistant_turns)

    if max_window >= 0 and probe_after > max_window:
        eligible_for_window += 1
        if visible_assistant_turns == max_window:
            window_capped += 1

    if visible_assistant_turns > expected_max:
        violations.append((custom_id, probe_after, visible_assistant_turns, expected_max))

print("RL sliding-window verification")
print(f"  source: {source_path}")
print(f"  inspected prompts: {inspected}")
print(f"  max_context_window: {max_window}")
print(f"  max visible assistant turns: {max_visible_assistant_turns}")
print(f"  prompts past window: {eligible_for_window}")
print(f"  prompts capped at window: {window_capped}")

if violations:
    print("Sliding-window violations:", file=sys.stderr)
    for custom_id, probe_after, visible, expected in violations[:10]:
        print(
            f"  {custom_id}: probe_after={probe_after}, "
            f"visible={visible}, expected<={expected}",
            file=sys.stderr,
        )
    raise SystemExit(1)

if max_window >= 0 and eligible_for_window > 0 and window_capped == 0:
    print("No prompts were capped by the sliding window.", file=sys.stderr)
    raise SystemExit(1)

print("  result: PASS")
PY
fi
