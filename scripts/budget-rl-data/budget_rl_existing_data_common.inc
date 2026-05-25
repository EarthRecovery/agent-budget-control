# Common implementation for the existing-rollout budget RL entry scripts.
#
# Expected variables from the caller:
#   EXPERIMENT_NAME, TASK, DATA_FILE, SOURCE_KIND, LEARNER_MODEL, ROLLOUT_MODEL
#
# SOURCE_KIND:
#   jsonl            DATA_FILE is already prepare_budget_probe.py JSONL input
#   estimation_json  DATA_FILE is a list of rollout dicts with turns[]
#   swe_summary      DATA_FILE is only a SWE-bench resolution summary; set
#                    SWE_INPUT_DIR or SWE_ROLLOUT_SOURCE_JSON for real training

budget_rl_contains_stage() {
  local wanted="$1"
  [[ "$STAGES_TEXT" == "all" ]] && return 0
  [[ ",${STAGES_TEXT// /,}," == *",$wanted,"* ]]
}

budget_rl_normalize_path() {
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

budget_rl_quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

budget_rl_pipeline_stages() {
  local stages=()
  for stage in prepare sft rl; do
    if budget_rl_contains_stage "$stage"; then
      stages+=("$stage")
    fi
  done
  local joined=""
  local stage
  for stage in "${stages[@]}"; do
    joined="${joined}${joined:+,}${stage}"
  done
  printf '%s\n' "$joined"
}

budget_rl_write_system_prompt_file() {
  if [[ -z "${SYSTEM_PROMPT_EXTRA:-}" ]]; then
    return 0
  fi

  SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-$EXP_BASE/system_prompt_${TASK}.txt}"
  mkdir -p "$(dirname "$SYSTEM_PROMPT_FILE")"

  case "$TASK" in
    warehouse)
      cat >"$SYSTEM_PROMPT_FILE" <<'EOF'
You are a warehouse-management planning agent. Read the current inventory, cash, demand, production, transit, and retailer state, then choose valid procurement, allocation, financing, or pass actions to keep the business operating successfully.

the rollout is generated from GPT5.2 instant
EOF
      ;;
    swebench|swe)
      cat >"$SYSTEM_PROMPT_FILE" <<'EOF'
You are a software engineering agent working on SWE-bench style GitHub issues. Inspect the repository, make targeted edits, run validation when useful, and submit a final patch when the issue is resolved.

the rollout is generated from GPT5.2 instant
EOF
      ;;
    *)
      printf '%s\n' "$SYSTEM_PROMPT_EXTRA" >"$SYSTEM_PROMPT_FILE"
      ;;
  esac

  export SYSTEM_PROMPT_FILE
  echo "SFT/RL system prompt override: $SYSTEM_PROMPT_FILE"
}

budget_rl_validate_jsonl_rollouts() {
  local path="$1"
  python3 - "$path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    first = handle.readline()
if not first:
    raise SystemExit(f"Empty rollout JSONL: {path}")
obj = json.loads(first)
if not isinstance(obj.get("messages"), list):
    raise SystemExit(f"JSONL record has no messages list: {path}")
PY
}

budget_rl_json_has_turns() {
  local path="$1"
  python3 - "$path" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "turns" in payload[0]:
    raise SystemExit(0)
if isinstance(payload, dict) and "turns" in payload:
    raise SystemExit(0)
if isinstance(payload, dict):
    for key in ("rollouts", "results"):
        value = payload.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict) and "turns" in value[0]:
            raise SystemExit(0)
raise SystemExit(1)
PY
}

budget_rl_convert_estimation_json() {
  local source_json="$1"
  local output_jsonl="$2"
  local -a cmd=(
    python3 "$PROJECT_ROOT/scripts/budget-rl/convert_estimation_dialogues.py"
    --input "$source_json"
    --output "$output_jsonl"
  )
  if [[ -n "${ROLLOUT_SOURCE_MAX_TURNS:-}" ]]; then
    cmd+=(--max-turns "$ROLLOUT_SOURCE_MAX_TURNS")
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    budget_rl_quote_cmd "${cmd[@]}"
    return 0
  fi
  "${cmd[@]}"
}

budget_rl_prepare_swe_source() {
  local source_json="${SWE_ROLLOUT_SOURCE_JSON:-${ROLLOUT_SOURCE_JSON:-}}"
  local source_dir="${SWE_INPUT_DIR:-}"

  if [[ -n "$source_dir" ]]; then
    source_dir="$(budget_rl_normalize_path "$source_dir")"
    source_json="$EXP_BASE/swebench_dialogues.json"
    local -a prepare_cmd=(
      python3 "$PROJECT_ROOT/scripts/budget-estimation-benchmark/prepare_swebench_dialogues.py"
      --input-dir "$source_dir"
      --output-json "$source_json"
    )
    if [[ -n "${MAX_ROLLOUTS:-}" ]]; then
      prepare_cmd+=(--max-rollouts "$MAX_ROLLOUTS")
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      budget_rl_quote_cmd "${prepare_cmd[@]}"
    else
      "${prepare_cmd[@]}"
    fi
  elif [[ -n "$source_json" ]]; then
    source_json="$(budget_rl_normalize_path "$source_json")"
  else
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "DRY_RUN: SWE training needs SWE_INPUT_DIR or SWE_ROLLOUT_SOURCE_JSON with real turns[]."
      return 0
    fi
    cat >&2 <<EOF
SWE_gpt5.2instant.json is a resolution summary only; it has no per-turn
messages, so it cannot be used for prepare/SFT/RL by itself.

Set one of:
  SWE_INPUT_DIR=scripts/budget-rl-data/swebench-origin-gpt5.2instant
  SWE_ROLLOUT_SOURCE_JSON=data/budget-rl/from-existing-data/swebench_dialogues.json
EOF
    exit 2
  fi

  if [[ "${DRY_RUN:-0}" != "1" && ! -f "$source_json" ]]; then
    echo "SWE source JSON not found: $source_json" >&2
    exit 2
  fi
  budget_rl_convert_estimation_json "$source_json" "$ROLLOUT_JSONL"
}

budget_rl_prepare_rollout_jsonl() {
  mkdir -p "$EXP_BASE"

  case "$SOURCE_KIND" in
    jsonl)
      ROLLOUT_JSONL="${ROLLOUT_JSONL:-$DATA_FILE}"
      ROLLOUT_JSONL="$(budget_rl_normalize_path "$ROLLOUT_JSONL")"
      if [[ "${DRY_RUN:-0}" != "1" ]]; then
        budget_rl_validate_jsonl_rollouts "$ROLLOUT_JSONL"
      fi
      ;;
    estimation_json)
      ROLLOUT_SOURCE_JSON="${ROLLOUT_SOURCE_JSON:-$DATA_FILE}"
      ROLLOUT_SOURCE_JSON="$(budget_rl_normalize_path "$ROLLOUT_SOURCE_JSON")"
      ROLLOUT_JSONL="${ROLLOUT_JSONL:-$EXP_BASE/rollouts.jsonl}"
      ROLLOUT_JSONL="$(budget_rl_normalize_path "$ROLLOUT_JSONL")"
      budget_rl_convert_estimation_json "$ROLLOUT_SOURCE_JSON" "$ROLLOUT_JSONL"
      ;;
    swe_summary)
      ROLLOUT_JSONL="${ROLLOUT_JSONL:-$EXP_BASE/rollouts.jsonl}"
      ROLLOUT_JSONL="$(budget_rl_normalize_path "$ROLLOUT_JSONL")"
      budget_rl_prepare_swe_source
      ;;
    *)
      echo "Unknown SOURCE_KIND=$SOURCE_KIND" >&2
      exit 2
      ;;
  esac

  export ROLLOUT_JSONL
}

budget_rl_summarize_original_rollout() {
  local summary_out="$EXP_BASE/compare/original_rollout_summary.json"
  mkdir -p "$(dirname "$summary_out")"

  python3 - "$DATA_FILE" "$ROLLOUT_JSONL" "$summary_out" "$TASK" "$EXPERIMENT_NAME" <<'PY'
import json
import statistics
import sys
from pathlib import Path

data_file = Path(sys.argv[1])
rollout_jsonl = Path(sys.argv[2])
summary_out = Path(sys.argv[3])
task = sys.argv[4]
experiment = sys.argv[5]


def success_from_metadata(metadata):
    for key in ("success", "rollout_success", "rollout_resolved"):
        if key in metadata:
            return bool(metadata.get(key))
    for key, value in metadata.items():
        if str(key).endswith("/success"):
            try:
                return float(value) == 1.0
            except (TypeError, ValueError):
                return bool(value)
    return False


def summarize_records(records, source):
    turns = []
    input_tokens = []
    output_tokens = []
    total_tokens = []
    success = 0
    total = 0
    for item in records:
        if not isinstance(item, dict):
            continue
        total += 1
        metadata = item.get("metadata") or item
        if success_from_metadata(metadata):
            success += 1
        if isinstance(metadata.get("num_turns"), (int, float)):
            turns.append(float(metadata["num_turns"]))
        elif isinstance(item.get("total_turns"), (int, float)):
            turns.append(float(item["total_turns"]))
        elif isinstance(item.get("turns"), list):
            turns.append(float(len(item["turns"])))
        for target, key in (
            (input_tokens, "api_input_tokens"),
            (output_tokens, "api_output_tokens"),
            (total_tokens, "api_total_tokens"),
        ):
            value = metadata.get(key)
            if isinstance(value, (int, float)):
                target.append(float(value))
    return {
        "experiment": experiment,
        "task": task,
        "source": str(source),
        "type": "rollout_records",
        "trajectories": total,
        "success": success,
        "success_rate": (success / total) if total else 0.0,
        "mean_turns": statistics.mean(turns) if turns else None,
        "mean_api_input_tokens": statistics.mean(input_tokens) if input_tokens else None,
        "mean_api_output_tokens": statistics.mean(output_tokens) if output_tokens else None,
        "mean_api_total_tokens": statistics.mean(total_tokens) if total_tokens else None,
    }


def load_json_records(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, payload
    if isinstance(payload, dict):
        for key in ("rollouts", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return payload, value
    return payload, None


summary = None
if data_file.suffix == ".jsonl":
    records = [json.loads(line) for line in data_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = summarize_records(records, data_file)
else:
    payload, records = load_json_records(data_file)
    if records is not None and records and isinstance(records[0], dict) and ("turns" in records[0] or "messages" in records[0]):
        summary = summarize_records(records, data_file)
    elif isinstance(payload, dict) and "resolved_instances" in payload:
        total = int(payload.get("total_instances") or payload.get("completed_instances") or 0)
        resolved = int(payload.get("resolved_instances") or 0)
        summary = {
            "experiment": experiment,
            "task": task,
            "source": str(data_file),
            "type": "swe_resolution_summary",
            "instances": total,
            "completed_instances": int(payload.get("completed_instances") or 0),
            "resolved_instances": resolved,
            "unresolved_instances": int(payload.get("unresolved_instances") or 0),
            "resolved_rate": (resolved / total) if total else 0.0,
        }

if summary is None and rollout_jsonl.is_file():
    records = [json.loads(line) for line in rollout_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = summarize_records(records, rollout_jsonl)

if summary is None:
    raise SystemExit(f"Cannot summarize original rollout source: {data_file}")

summary_out.parent.mkdir(parents=True, exist_ok=True)
summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("Original rollout summary")
for key, value in summary.items():
    print(f"  {key}: {value}")
print(f"  written: {summary_out}")
PY
}

budget_rl_latest_rl_checkpoint() {
  if [[ ! -d "$RL_SAVE_DIR" ]]; then
    return 0
  fi
  find "$RL_SAVE_DIR" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null \
    | sort -V \
    | tail -n 1
}

budget_rl_resolve_trained_model() {
  if [[ -n "${TRAINED_MODEL:-}" ]]; then
    printf '%s\n' "$TRAINED_MODEL"
    return 0
  fi

  local latest
  latest="$(budget_rl_latest_rl_checkpoint || true)"
  if [[ -n "$latest" ]]; then
    local local_dir target_dir
    if [[ -f "$latest/actor/fsdp_config.json" ]]; then
      local_dir="$latest/actor"
      target_dir="${RL_HF_TARGET_DIR:-$latest/actor/huggingface_merged}"
    elif [[ -f "$latest/fsdp_config.json" ]]; then
      local_dir="$latest"
      target_dir="${RL_HF_TARGET_DIR:-$latest/huggingface_merged}"
    else
      local_dir=""
      target_dir=""
    fi

    if [[ -n "$local_dir" ]]; then
      if [[ ! -f "$target_dir/config.json" ]]; then
        if [[ "${MERGE_RL_CHECKPOINT:-1}" == "1" ]]; then
          local -a merge_cmd=(
            python3 -m verl.model_merger merge
            --backend fsdp
            --local_dir "$local_dir"
            --target_dir "$target_dir"
          )
          if [[ "${USE_CPU_INITIALIZATION:-1}" == "1" ]]; then
            merge_cmd+=(--use_cpu_initialization)
          fi
          if [[ "${DRY_RUN:-0}" == "1" ]]; then
            budget_rl_quote_cmd "${merge_cmd[@]}" >&2
          else
            "${merge_cmd[@]}" >&2
          fi
        fi
      fi
      if [[ -f "$target_dir/config.json" || "${DRY_RUN:-0}" == "1" ]]; then
        printf '%s\n' "$target_dir"
        return 0
      fi
    fi
  fi

  if [[ -f "$SFT_CKPT/config.json" || "${DRY_RUN:-0}" == "1" ]]; then
    echo "No merged RL checkpoint found; using SFT checkpoint for trained-model comparison." >&2
    printf '%s\n' "$SFT_CKPT"
    return 0
  fi

  echo "No trained checkpoint found. Set TRAINED_MODEL=models/trained_hf_model." >&2
  exit 3
}

budget_rl_wait_for_vllm() {
  local url="$1"
  local seconds="${VLLM_START_TIMEOUT:-600}"
  local waited=0
  until curl -fsS "$url/v1/models" >/dev/null 2>&1; do
    sleep 5
    waited=$((waited + 5))
    if (( waited >= seconds )); then
      return 1
    fi
  done
}

budget_rl_start_vllm() {
  local model_path="$1"
  local served_name="$2"
  local port="$3"
  local log_file="$4"
  local host="${VLLM_HOST:-127.0.0.1}"
  local -a cmd=(
    python3 -m vllm.entrypoints.openai.api_server
    --host "$host"
    --port "$port"
    --model "$model_path"
    --served-model-name "$served_name"
    --tensor-parallel-size "${VLLM_TP_SIZE:-${TP_SIZE:-1}}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
    --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}"
  )
  if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local extra=(${VLLM_EXTRA_ARGS})
    cmd+=("${extra[@]}")
  fi

  echo "Starting vLLM: model=$model_path served_name=$served_name url=http://$host:$port"
  budget_rl_quote_cmd "${cmd[@]}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "$log_file")"
  CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}" \
    "${cmd[@]}" >>"$log_file" 2>&1 &
  BUDGET_RL_DATA_VLLM_PID="$!"
  if ! budget_rl_wait_for_vllm "http://$host:$port"; then
    echo "vLLM did not become healthy. Log: $log_file" >&2
    kill "$BUDGET_RL_DATA_VLLM_PID" >/dev/null 2>&1 || true
    wait "$BUDGET_RL_DATA_VLLM_PID" >/dev/null 2>&1 || true
    BUDGET_RL_DATA_VLLM_PID=""
    exit 4
  fi
}

budget_rl_stop_vllm() {
  if [[ -n "${BUDGET_RL_DATA_VLLM_PID:-}" ]]; then
    kill "$BUDGET_RL_DATA_VLLM_PID" >/dev/null 2>&1 || true
    wait "$BUDGET_RL_DATA_VLLM_PID" >/dev/null 2>&1 || true
    BUDGET_RL_DATA_VLLM_PID=""
  fi
}

budget_rl_run_eval_checkpoint() {
  local url="$1"
  local model_name="$2"
  local output="$3"
  local -a cmd=(
    python3 "$PROJECT_ROOT/scripts/budget-rl/eval_checkpoint.py"
    --vllm-url "$url"
    --model-name "$model_name"
    --test-parquet "$EXP_BASE/eval_test/train.parquet"
    --output "$output"
    --max-tokens "${EVAL_MAX_TOKENS:-512}"
    --temperature "${EVAL_TEMPERATURE:-0.0}"
    --max-concurrency "${EVAL_MAX_CONCURRENCY:-8}"
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    budget_rl_quote_cmd "${cmd[@]}"
    return 0
  fi
  "${cmd[@]}"
}

budget_rl_write_comparison_summary() {
  local original="$EXP_BASE/compare/original_rollout_summary.json"
  local untrained="$EXP_BASE/compare/untrained_eval.json"
  local trained="$EXP_BASE/compare/trained_eval.json"
  local out="$EXP_BASE/compare/comparison_summary.json"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN: would write comparison summary to $out"
    return 0
  fi
  if [[ ! -f "$untrained" || ! -f "$trained" ]]; then
    return 0
  fi

  python3 - "$original" "$untrained" "$trained" "$out" <<'PY'
import json
import sys

original_path, untrained_path, trained_path, out_path = sys.argv[1:5]

def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

original = load(original_path)
untrained = load(untrained_path).get("summary", {})
trained = load(trained_path).get("summary", {})
keys = [
    "mean_reward",
    "format_valid_rate",
    "class_accuracy",
    "pred_hit_possible",
    "pred_hit_impossible",
    "cover_rate_possible",
    "mean_relative_error",
    "median_relative_error",
]
delta = {
    key: trained.get(key, 0.0) - untrained.get(key, 0.0)
    for key in keys
    if isinstance(trained.get(key), (int, float)) and isinstance(untrained.get(key), (int, float))
}
payload = {
    "original_rollout": original,
    "untrained_model": untrained,
    "trained_model": trained,
    "trained_minus_untrained": delta,
}
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
print(f"Comparison summary -> {out_path}")
for key in keys:
    if key in delta:
        print(f"  {key}: untrained={untrained.get(key):.4f} trained={trained.get(key):.4f} delta={delta[key]:+.4f}")
PY
}

budget_rl_run_model_compare() {
  if [[ "${RUN_MODEL_COMPARE:-1}" != "1" ]]; then
    echo "RUN_MODEL_COMPARE=0: skipping untrained/trained vLLM eval."
    return 0
  fi
  if [[ ! -f "$EXP_BASE/eval_test/train.parquet" && "${DRY_RUN:-0}" != "1" ]]; then
    echo "Missing eval parquet: $EXP_BASE/eval_test/train.parquet. Run prepare first." >&2
    exit 2
  fi

  local compare_dir="$EXP_BASE/compare"
  mkdir -p "$compare_dir"
  local trained_model
  trained_model="$(budget_rl_resolve_trained_model)"
  local untrained_model="${UNTRAINED_MODEL:-$LEARNER_MODEL}"
  local mode="${MODEL_COMPARE_MODE:-serve}"

  case "$mode" in
    external)
      local untrained_url="${UNTRAINED_VLLM_URL:-${BASE_VLLM_URL:-${VLLM_URL:-}}}"
      local trained_url="${TRAINED_VLLM_URL:-${VLLM_URL:-}}"
      if [[ -z "$untrained_url" || -z "$trained_url" ]]; then
        cat >&2 <<EOF
MODEL_COMPARE_MODE=external requires:
  UNTRAINED_VLLM_URL=http://host:port
  TRAINED_VLLM_URL=http://host:port
EOF
        exit 2
      fi
      budget_rl_run_eval_checkpoint "$untrained_url" "${UNTRAINED_MODEL_NAME:-$untrained_model}" "$compare_dir/untrained_eval.json"
      budget_rl_run_eval_checkpoint "$trained_url" "${TRAINED_MODEL_NAME:-$trained_model}" "$compare_dir/trained_eval.json"
      ;;
    serve)
      local host="${VLLM_HOST:-127.0.0.1}"
      local port="${VLLM_PORT:-8000}"
      local untrained_served="${UNTRAINED_SERVED_MODEL_NAME:-untrained-${EXPERIMENT_NAME}}"
      local trained_served="${TRAINED_SERVED_MODEL_NAME:-trained-${EXPERIMENT_NAME}}"
      budget_rl_start_vllm "$untrained_model" "$untrained_served" "$port" "$compare_dir/vllm_untrained.log"
      budget_rl_run_eval_checkpoint "http://$host:$port" "$untrained_served" "$compare_dir/untrained_eval.json"
      budget_rl_stop_vllm
      budget_rl_start_vllm "$trained_model" "$trained_served" "$port" "$compare_dir/vllm_trained.log"
      budget_rl_run_eval_checkpoint "http://$host:$port" "$trained_served" "$compare_dir/trained_eval.json"
      budget_rl_stop_vllm
      ;;
    skip)
      echo "MODEL_COMPARE_MODE=skip: only original rollout summary was written."
      ;;
    *)
      echo "Unknown MODEL_COMPARE_MODE=$mode (use serve, external, or skip)" >&2
      exit 2
      ;;
  esac

  if [[ "$mode" != "skip" ]]; then
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "DRY_RUN: would run analyze_eval.py on untrained/trained eval JSON."
    else
      python3 "$PROJECT_ROOT/scripts/budget-rl/analyze_eval.py" \
        "$compare_dir/untrained_eval.json" \
        "$compare_dir/trained_eval.json"
    fi
    budget_rl_write_comparison_summary
  fi
}

budget_rl_cleanup() {
  budget_rl_stop_vllm
}

budget_rl_data_main() {
  : "${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
  : "${TASK:?TASK is required}"
  : "${DATA_FILE:?DATA_FILE is required}"
  : "${SOURCE_KIND:?SOURCE_KIND is required}"
  : "${LEARNER_MODEL:?LEARNER_MODEL is required}"
  : "${ROLLOUT_MODEL:?ROLLOUT_MODEL is required}"

  STAGES_TEXT="${1:-${STAGES:-prepare,sft,rl,compare}}"
  DATA_FILE="$(budget_rl_normalize_path "$DATA_FILE")"
  DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/budget-rl/from-existing-data}"
  EXP_BASE="${EXP_BASE:-$DATA_ROOT/$EXPERIMENT_NAME}"
  EXP_BASE="$(budget_rl_normalize_path "$EXP_BASE")"
  SFT_ABLATION="${SFT_ABLATION:-sft_interval_pct30}"
  SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-5}"
  RL_TOTAL_EPOCHS="${RL_TOTAL_EPOCHS:-5}"
  RL_KL="${RL_KL:-0.05}"
  SFT_CKPT="${SFT_CKPT:-$EXP_BASE/checkpoints/$SFT_ABLATION/huggingface_e$SFT_TOTAL_EPOCHS}"
  RL_EXPERIMENT_NAME="${RL_EXPERIMENT_NAME:-${EXPERIMENT_NAME}_rl_${SFT_ABLATION}_e${SFT_TOTAL_EPOCHS}_kl$(printf '%s' "$RL_KL" | tr -d '.')}"
  RL_SAVE_DIR="${RL_SAVE_DIR:-$EXP_BASE/checkpoints/$RL_EXPERIMENT_NAME}"

  export TASK DATA_FILE DATA_ROOT EXP_BASE
  export ROLLOUT_MODEL LEARNER_MODEL TOKENIZER
  export SFT_ABLATION SFT_TOTAL_EPOCHS RL_TOTAL_EPOCHS RL_KL
  export SFT_CKPT RL_EXPERIMENT_NAME RL_SAVE_DIR

  cd "$PROJECT_ROOT"
  mkdir -p "$EXP_BASE"
  if [[ "${BUDGET_RL_DISABLE_LOG:-0}" != "1" && "${BUDGET_RL_LOG_STARTED:-0}" != "1" ]]; then
    BUDGET_RL_LOG_FILE="${BUDGET_RL_LOG_FILE:-$EXP_BASE/run.log}"
    BUDGET_RL_LOG_FILE="$(budget_rl_normalize_path "$BUDGET_RL_LOG_FILE")"
    mkdir -p "$(dirname "$BUDGET_RL_LOG_FILE")"
    if [[ "${BUDGET_RL_APPEND_LOG:-0}" != "1" ]]; then
      : >"$BUDGET_RL_LOG_FILE"
    fi
    export BUDGET_RL_LOG_FILE
    export BUDGET_RL_LOG_STARTED=1
    exec > >(tee -a "$BUDGET_RL_LOG_FILE") 2>&1
  fi
  trap budget_rl_cleanup EXIT

  if [[ ! -f "$DATA_FILE" && "${DRY_RUN:-0}" != "1" ]]; then
    echo "Data file not found: $DATA_FILE" >&2
    exit 2
  fi

  echo "Budget RL existing-data experiment"
  echo "  experiment: $EXPERIMENT_NAME"
  echo "  task: $TASK"
  echo "  source: $DATA_FILE"
  echo "  source kind: $SOURCE_KIND"
  echo "  learner model: $LEARNER_MODEL"
  echo "  stages: $STAGES_TEXT"
  echo "  exp base: $EXP_BASE"
  echo "  log file: ${BUDGET_RL_LOG_FILE:-disabled}"
  echo "  timestamp: $(date -Is)"
  echo "  git commit: $(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
  echo "  git status short:"
  git status --short 2>/dev/null || true
  echo "  ngpus: ${NGPUS:-8}"
  echo "  tp_size: ${TP_SIZE:-4}"
  echo "  sft_ngpus: ${SFT_NGPUS:-${NGPUS:-8}}"
  echo "  rl_ngpus: ${RL_NGPUS:-${NGPUS:-8}}"
  echo "  rl_tp_size: ${RL_TP_SIZE:-${TP_SIZE:-4}}"
  echo "  rl_batch_size: ${RL_BATCH_SIZE:-64}"
  echo "  rl_rollout_n: ${RL_ROLLOUT_N:-16}"
  echo "  rl_lr: ${RL_LR:-5e-7}"
  echo "  rl_epochs: $RL_TOTAL_EPOCHS"
  echo "  rl_kl: $RL_KL"
  echo "  rl_max_prompt_length: ${RL_MAX_PROMPT_LENGTH:-8192}"
  echo "  rl_max_model_len: ${RL_MAX_MODEL_LEN:-${RL_MAX_PROMPT_LENGTH:-8192}}"
  echo "  rl_max_response_length: ${RL_MAX_RESPONSE_LENGTH:-1024}"
  echo "  eval_max_tokens: ${EVAL_MAX_TOKENS:-512}"
  echo "  eval_temperature: ${EVAL_TEMPERATURE:-0.0}"
  echo "  target_unit: ${TARGET_UNIT:-tokens}"
  echo "  max_cost: ${MAX_COST:-disabled}"
  echo "  cost_key: ${COST_KEY:-env_reward_cost}"
  echo "  budget_probe_context_window_mode: ${BUDGET_PROBE_CONTEXT_WINDOW_MODE:-full}"
  echo "  budget_probe_max_context_window: ${BUDGET_PROBE_MAX_CONTEXT_WINDOW:--1}"
  echo "  budget_probe_max_prompt_tokens: ${BUDGET_PROBE_MAX_PROMPT_TOKENS:-disabled}"
  echo "  budget_probe_drop_overlong_prompts: ${BUDGET_PROBE_DROP_OVERLONG_PROMPTS:-0}"

  if budget_rl_contains_stage prepare || budget_rl_contains_stage sft || budget_rl_contains_stage rl || budget_rl_contains_stage compare; then
    budget_rl_prepare_rollout_jsonl
  fi

  budget_rl_write_system_prompt_file

  local pipeline_stages
  pipeline_stages="$(budget_rl_pipeline_stages)"
  if [[ -n "$pipeline_stages" ]]; then
    echo
    echo "=== prepare/SFT/RL pipeline: $pipeline_stages ==="
    env \
      TASK="$TASK" \
      ROLLOUT_MODEL="$ROLLOUT_MODEL" \
      LEARNER_MODEL="$LEARNER_MODEL" \
      TOKENIZER="$TOKENIZER" \
      EXP_NAME="$EXPERIMENT_NAME" \
      EXP_BASE="$EXP_BASE" \
      ROLLOUT_JSONL="$ROLLOUT_JSONL" \
      SFT_CKPT="$SFT_CKPT" \
      RL_EXPERIMENT_NAME="$RL_EXPERIMENT_NAME" \
      RL_SAVE_DIR="$RL_SAVE_DIR" \
      RL_MAX_PROMPT_LENGTH="${RL_MAX_PROMPT_LENGTH:-8192}" \
      RL_MAX_MODEL_LEN="${RL_MAX_MODEL_LEN:-${RL_MAX_PROMPT_LENGTH:-8192}}" \
      RL_MAX_RESPONSE_LENGTH="${RL_MAX_RESPONSE_LENGTH:-1024}" \
      TARGET_UNIT="${TARGET_UNIT:-tokens}" \
      MAX_COST="${MAX_COST:-}" \
      COST_KEY="${COST_KEY:-env_reward_cost}" \
      COST_DECIMALS="${COST_DECIMALS:-2}" \
      MIN_REMAINING_TOKENS="${MIN_REMAINING_TOKENS:-1}" \
      MIN_REMAINING_VALUE="${MIN_REMAINING_VALUE:-}" \
      BUDGET_PROBE_CONTEXT_WINDOW_MODE="${BUDGET_PROBE_CONTEXT_WINDOW_MODE:-full}" \
      BUDGET_PROBE_MAX_CONTEXT_WINDOW="${BUDGET_PROBE_MAX_CONTEXT_WINDOW:--1}" \
      BUDGET_PROBE_MAX_PROMPT_TOKENS="${BUDGET_PROBE_MAX_PROMPT_TOKENS:-}" \
      BUDGET_PROBE_DROP_OVERLONG_PROMPTS="${BUDGET_PROBE_DROP_OVERLONG_PROMPTS:-0}" \
      bash "$PROJECT_ROOT/scripts/budget-rl/run_budget_rl_pipeline.sh" "$pipeline_stages"
  fi

  if budget_rl_contains_stage compare; then
    echo
    echo "=== compare original rollout, untrained model, trained model ==="
    budget_rl_summarize_original_rollout
    budget_rl_run_model_compare
  fi
}
