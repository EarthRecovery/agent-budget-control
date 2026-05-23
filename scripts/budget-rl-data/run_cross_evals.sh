#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

# shellcheck source=scripts/budget-rl-data/budget_rl_existing_data_common.inc
source "$SCRIPT_DIR/budget_rl_existing_data_common.inc"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/budget-rl-data/run_cross_evals.sh
  bash scripts/budget-rl-data/run_three_cross_evals.sh

Runs the three requested trained-model cross-rollout budget-probe evaluations.
This only evaluates on rollout-derived parquet data; it does not run task
environments or generate new rollouts.

  1. trained Qwen3-4B Sokoban -> SearchR1
  2. trained Llama3.1-8B Sokoban -> SearchR1
  3. trained Qwen2.5-7B SearchR1 -> Sokoban

Defaults:
  SearchR1 target eval data:
    data/budget-rl/from-existing-data/searchr1_qwen25_7b_existing_data/eval_test/train.parquet
  Sokoban target eval data:
    data/budget-rl/from-existing-data/sokoban_qwen3_4b_existing_data/eval_test/train.parquet

Common overrides:
  DRY_RUN=1
  CROSS_EVAL_MODE=serve            # serve or external
  VLLM_PORT=8000
  VLLM_CUDA_VISIBLE_DEVICES=0
  VLLM_TP_SIZE=4
  CROSS_EVAL_OUT_DIR=results/cross-eval-three
  CROSS_EVAL_LOG_FILE=results/cross-eval-three/run.log

Model overrides:
  TRAINED_QWEN3_4B_SOKOBAN_MODEL=models/qwen3_4b_sokoban_rl
  TRAINED_LLAMA31_8B_SOKOBAN_MODEL=models/llama31_8b_sokoban_rl
  TRAINED_QWEN25_7B_SEARCHR1_MODEL=models/qwen25_7b_searchr1_rl

Target data overrides:
  SEARCHR1_EVAL_EXP_BASE=data/budget-rl/from-existing-data/searchr1_qwen25_7b_existing_data
  SOKOBAN_EVAL_EXP_BASE=data/budget-rl/from-existing-data/sokoban_qwen3_4b_existing_data
  SOKOBAN_PREPARE_SCRIPT=scripts/budget-rl-data/run_sokoban_llama31_8b.sh

For CROSS_EVAL_MODE=external, set per-eval URLs if needed, e.g.:
  TRAINED_QWEN3_4B_SOKOBAN_TO_SEARCHR1_VLLM_URL=http://host:8000
  TRAINED_QWEN3_4B_SOKOBAN_TO_SEARCHR1_MODEL_NAME=served-model-name
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

activate_runtime() {
  if [[ "${SKIP_ENV_ACTIVATE:-0}" == "1" ]]; then
    return 0
  fi
  if [[ -n "${VENV_PATH:-}" && -f "$VENV_PATH/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
    return 0
  fi
  if [[ -n "${CONDA_ENV_NAME:-ragenv2}" ]]; then
    if command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)"
      conda activate "${CONDA_ENV_NAME:-ragenv2}"
    elif [[ -n "${CONDA_BASE:-}" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$CONDA_BASE/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV_NAME:-ragenv2}"
    fi
  fi
}

normalize_path() {
  budget_rl_normalize_path "$1"
}

quote_cmd() {
  budget_rl_quote_cmd "$@"
}

run_or_print() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    quote_cmd "$@"
  else
    "$@"
  fi
}

ensure_eval_data() {
  local name="$1"
  local exp_base="$2"
  local prepare_script="$3"
  local parquet="$exp_base/eval_test/train.parquet"

  if [[ -f "$parquet" ]]; then
    echo "$name eval data: $parquet"
    return 0
  fi

  if [[ "${ENSURE_EVAL_DATA:-1}" != "1" ]]; then
    echo "Missing $name eval data: $parquet" >&2
    exit 2
  fi

  echo "Preparing $name eval data via $prepare_script"
  run_or_print env \
    DATA_ROOT="$DATA_ROOT" \
    EXP_BASE="$exp_base" \
    RUN_MODEL_COMPARE=0 \
    MODEL_COMPARE_MODE=skip \
    bash "$prepare_script" prepare

  if [[ "${DRY_RUN:-0}" != "1" && ! -f "$parquet" ]]; then
    echo "Prepare did not produce $parquet" >&2
    exit 2
  fi
}

latest_config_dir() {
  local root="$1"
  shift
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  local config
  config="$(find "$root" -type f "$@" 2>/dev/null \
    | sort -V \
    | tail -n 1)"
  if [[ -n "$config" ]]; then
    dirname "$config"
  fi
}

resolve_trained_model() {
  local label="$1"
  local exp_base="$2"
  local override="$3"
  local ckpt_root="$exp_base/checkpoints"

  if [[ -n "$override" ]]; then
    normalize_path "$override"
    return 0
  fi

  local merged
  merged="$(
    latest_config_dir "$ckpt_root" \
      \( -path '*/global_step_*/actor/huggingface_merged/config.json' \
      -o -path '*/global_step_*/huggingface_merged/config.json' \
      -o -path '*/global_step_*/actor/huggingface/config.json' \
      -o -path '*/global_step_*/huggingface/config.json' \)
  )"
  if [[ -n "$merged" ]]; then
    printf '%s\n' "$merged"
    return 0
  fi

  local fsdp_config local_dir target_dir
  fsdp_config="$(
    find "$ckpt_root" -type f \
      \( -path '*/global_step_*/actor/fsdp_config.json' \
      -o -path '*/global_step_*/fsdp_config.json' \) 2>/dev/null \
      | sort -V \
      | tail -n 1
  )"
  if [[ -n "$fsdp_config" ]]; then
    local_dir="$(dirname "$fsdp_config")"
    target_dir="$local_dir/huggingface_merged"
    if [[ ! -f "$target_dir/config.json" ]]; then
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
        quote_cmd "${merge_cmd[@]}" >&2
      else
        "${merge_cmd[@]}" >&2
      fi
    fi
    printf '%s\n' "$target_dir"
    return 0
  fi

  if [[ "${ALLOW_SFT_FALLBACK:-0}" == "1" ]]; then
    local sft
    sft="$(
      latest_config_dir "$ckpt_root" \
        \( -path "*/huggingface_e${SFT_TOTAL_EPOCHS:-5}/config.json" \
        -o -path '*/huggingface_e*/config.json' \)
    )"
    if [[ -n "$sft" ]]; then
      echo "Using SFT fallback for $label: $sft" >&2
      printf '%s\n' "$sft"
      return 0
    fi
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%s\n' "$ckpt_root/latest_rl_huggingface_model"
    return 0
  fi

  cat >&2 <<EOF
Could not find a merged or mergeable RL checkpoint for $label under:
  $ckpt_root

Set the model path explicitly, for example:
  TRAINED_QWEN3_4B_SOKOBAN_MODEL=models/qwen3_4b_sokoban_rl
EOF
  exit 3
}

run_eval_checkpoint() {
  local url="$1"
  local model_name="$2"
  local test_parquet="$3"
  local output="$4"
  local -a cmd=(
    python3 "$PROJECT_ROOT/scripts/budget-rl/eval_checkpoint.py"
    --vllm-url "$url"
    --model-name "$model_name"
    --test-parquet "$test_parquet"
    --output "$output"
    --max-tokens "${EVAL_MAX_TOKENS:-512}"
    --temperature "${EVAL_TEMPERATURE:-0.0}"
    --max-concurrency "${EVAL_MAX_CONCURRENCY:-8}"
  )
  run_or_print "${cmd[@]}"
}

run_cross_eval() {
  local id="$1"
  local model_path="$2"
  local target_name="$3"
  local test_parquet="$4"
  local output="$CROSS_EVAL_OUT_DIR/${id}.json"
  local served_name="$id"
  local mode="${CROSS_EVAL_MODE:-serve}"
  local host="${VLLM_HOST:-127.0.0.1}"
  local port="${VLLM_PORT:-8000}"

  echo
  echo "=== $id ==="
  echo "  model: $model_path"
  echo "  target: $target_name"
  echo "  test parquet: $test_parquet"
  echo "  output: $output"

  case "$mode" in
    serve)
      {
        echo
        echo "----- vLLM log for $id starts below -----"
      } >>"$CROSS_EVAL_LOG_FILE"
      budget_rl_start_vllm "$model_path" "$served_name" "$port" "$CROSS_EVAL_LOG_FILE"
      run_eval_checkpoint "http://$host:$port" "$served_name" "$test_parquet" "$output"
      budget_rl_stop_vllm
      {
        echo "----- vLLM log for $id ended -----"
        echo
      } >>"$CROSS_EVAL_LOG_FILE"
      ;;
    external)
      local prefix url_var name_var url model_name
      prefix="$(printf '%s' "$id" | tr '[:lower:]' '[:upper:]')"
      url_var="${prefix}_VLLM_URL"
      name_var="${prefix}_MODEL_NAME"
      url="${!url_var:-${CROSS_EVAL_VLLM_URL:-}}"
      model_name="${!name_var:-$served_name}"
      if [[ -z "$url" ]]; then
        echo "CROSS_EVAL_MODE=external requires ${url_var}=http://host:port or CROSS_EVAL_VLLM_URL." >&2
        exit 2
      fi
      run_eval_checkpoint "$url" "$model_name" "$test_parquet" "$output"
      ;;
    skip)
      echo "CROSS_EVAL_MODE=skip: skipping $id"
      ;;
    *)
      echo "Unknown CROSS_EVAL_MODE=$mode (use serve, external, or skip)" >&2
      exit 2
      ;;
  esac
}

write_summary() {
  if [[ "${CROSS_EVAL_MODE:-serve}" == "skip" || "${DRY_RUN:-0}" == "1" ]]; then
    echo "Skipping cross-eval summary in dry-run/skip mode."
    return 0
  fi

  python3 - "$CROSS_EVAL_OUT_DIR/summary.json" "$CROSS_EVAL_OUT_DIR" "${CROSS_EVAL_IDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
ids = sys.argv[3:]
rows = []
for item_id in ids:
    path = out_dir / f"{item_id}.json"
    if not path.is_file():
        rows.append({"id": item_id, "missing": True, "path": str(path)})
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    rows.append({
        "id": item_id,
        "path": str(path),
        "model_name": summary.get("model_name"),
        "test_parquet": summary.get("test_parquet"),
        "n_samples": summary.get("n_samples"),
        "mean_reward": summary.get("mean_reward"),
        "format_valid_rate": summary.get("format_valid_rate"),
        "class_accuracy": summary.get("class_accuracy"),
        "pred_hit_possible": summary.get("pred_hit_possible"),
        "pred_hit_impossible": summary.get("pred_hit_impossible"),
        "cover_rate_possible": summary.get("cover_rate_possible"),
        "mean_relative_error": summary.get("mean_relative_error"),
        "median_relative_error": summary.get("median_relative_error"),
    })

out_path.write_text(json.dumps({"results": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Cross-eval summary -> {out_path}")
for row in rows:
    if row.get("missing"):
        print(f"  {row['id']}: missing")
        continue
    print(
        f"  {row['id']}: reward={row.get('mean_reward'):.4f} "
        f"acc={row.get('class_accuracy'):.4f} "
        f"cover={row.get('cover_rate_possible'):.4f}"
    )
PY

  local -a outputs=()
  local id
  for id in "${CROSS_EVAL_IDS[@]}"; do
    if [[ -f "$CROSS_EVAL_OUT_DIR/${id}.json" ]]; then
      outputs+=("$CROSS_EVAL_OUT_DIR/${id}.json")
    fi
  done
  if (( ${#outputs[@]} > 0 )); then
    python3 "$PROJECT_ROOT/scripts/budget-rl/analyze_eval.py" "${outputs[@]}"
  fi
}

cleanup() {
  budget_rl_stop_vllm
}
trap cleanup EXIT

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/budget-rl/from-existing-data}"
DATA_ROOT="$(normalize_path "$DATA_ROOT")"
SEARCHR1_EVAL_EXP_BASE="${SEARCHR1_EVAL_EXP_BASE:-$DATA_ROOT/searchr1_qwen25_7b_existing_data}"
SOKOBAN_EVAL_EXP_BASE="${SOKOBAN_EVAL_EXP_BASE:-$DATA_ROOT/sokoban_qwen3_4b_existing_data}"
SEARCHR1_EVAL_EXP_BASE="$(normalize_path "$SEARCHR1_EVAL_EXP_BASE")"
SOKOBAN_EVAL_EXP_BASE="$(normalize_path "$SOKOBAN_EVAL_EXP_BASE")"
SEARCHR1_PREPARE_SCRIPT="${SEARCHR1_PREPARE_SCRIPT:-$SCRIPT_DIR/run_searchr1_qwen25_7b.sh}"
SOKOBAN_PREPARE_SCRIPT="${SOKOBAN_PREPARE_SCRIPT:-$SCRIPT_DIR/run_sokoban_qwen3_4b.sh}"

QWEN3_SOKOBAN_EXP_BASE="${QWEN3_SOKOBAN_EXP_BASE:-$DATA_ROOT/sokoban_qwen3_4b_existing_data}"
LLAMA31_SOKOBAN_EXP_BASE="${LLAMA31_SOKOBAN_EXP_BASE:-$DATA_ROOT/sokoban_llama31_8b_existing_data}"
QWEN25_SEARCHR1_EXP_BASE="${QWEN25_SEARCHR1_EXP_BASE:-$DATA_ROOT/searchr1_qwen25_7b_existing_data}"
QWEN3_SOKOBAN_EXP_BASE="$(normalize_path "$QWEN3_SOKOBAN_EXP_BASE")"
LLAMA31_SOKOBAN_EXP_BASE="$(normalize_path "$LLAMA31_SOKOBAN_EXP_BASE")"
QWEN25_SEARCHR1_EXP_BASE="$(normalize_path "$QWEN25_SEARCHR1_EXP_BASE")"

CROSS_EVAL_OUT_DIR="${CROSS_EVAL_OUT_DIR:-$DATA_ROOT/cross-eval-three}"
CROSS_EVAL_OUT_DIR="$(normalize_path "$CROSS_EVAL_OUT_DIR")"
mkdir -p "$CROSS_EVAL_OUT_DIR/logs"
export TP_SIZE="${TP_SIZE:-4}"
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-4}"
CROSS_EVAL_LOG_FILE="${CROSS_EVAL_LOG_FILE:-$CROSS_EVAL_OUT_DIR/run.log}"
CROSS_EVAL_LOG_FILE="$(normalize_path "$CROSS_EVAL_LOG_FILE")"
mkdir -p "$(dirname "$CROSS_EVAL_LOG_FILE")"
if [[ "${CROSS_EVAL_APPEND_LOG:-0}" != "1" ]]; then
  : >"$CROSS_EVAL_LOG_FILE"
fi
exec > >(tee -a "$CROSS_EVAL_LOG_FILE") 2>&1

echo "=== cross-eval run log ==="
echo "timestamp: $(date -Is)"
echo "project_root: $PROJECT_ROOT"
echo "log_file: $CROSS_EVAL_LOG_FILE"
echo "output_dir: $CROSS_EVAL_OUT_DIR"
echo "mode: ${CROSS_EVAL_MODE:-serve}"
echo "dry_run: ${DRY_RUN:-0}"
echo "git_commit: $(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
echo "git_status_short:"
git status --short 2>/dev/null || true

activate_runtime
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"

SEARCHR1_TEST_PARQUET="$SEARCHR1_EVAL_EXP_BASE/eval_test/train.parquet"
SOKOBAN_TEST_PARQUET="$SOKOBAN_EVAL_EXP_BASE/eval_test/train.parquet"

ensure_eval_data "SearchR1" "$SEARCHR1_EVAL_EXP_BASE" "$SEARCHR1_PREPARE_SCRIPT"
ensure_eval_data "Sokoban" "$SOKOBAN_EVAL_EXP_BASE" "$SOKOBAN_PREPARE_SCRIPT"

TRAINED_QWEN3_4B_SOKOBAN_RESOLVED="$(
  resolve_trained_model \
    "trained Qwen3-4B Sokoban" \
    "$QWEN3_SOKOBAN_EXP_BASE" \
    "${TRAINED_QWEN3_4B_SOKOBAN_MODEL:-}"
)"
TRAINED_LLAMA31_8B_SOKOBAN_RESOLVED="$(
  resolve_trained_model \
    "trained Llama3.1-8B Sokoban" \
    "$LLAMA31_SOKOBAN_EXP_BASE" \
    "${TRAINED_LLAMA31_8B_SOKOBAN_MODEL:-}"
)"
TRAINED_QWEN25_7B_SEARCHR1_RESOLVED="$(
  resolve_trained_model \
    "trained Qwen2.5-7B SearchR1" \
    "$QWEN25_SEARCHR1_EXP_BASE" \
    "${TRAINED_QWEN25_7B_SEARCHR1_MODEL:-}"
)"

CROSS_EVAL_IDS=(
  trained_qwen3_4b_sokoban_to_searchr1
  trained_llama31_8b_sokoban_to_searchr1
  trained_qwen25_7b_searchr1_to_sokoban
)

cat <<EOF

=== key experiment data ===
data_root: $DATA_ROOT
searchr1_eval_exp_base: $SEARCHR1_EVAL_EXP_BASE
searchr1_test_parquet: $SEARCHR1_TEST_PARQUET
sokoban_eval_exp_base: $SOKOBAN_EVAL_EXP_BASE
sokoban_test_parquet: $SOKOBAN_TEST_PARQUET
qwen3_sokoban_exp_base: $QWEN3_SOKOBAN_EXP_BASE
llama31_sokoban_exp_base: $LLAMA31_SOKOBAN_EXP_BASE
qwen25_searchr1_exp_base: $QWEN25_SEARCHR1_EXP_BASE
trained_qwen3_4b_sokoban_model: $TRAINED_QWEN3_4B_SOKOBAN_RESOLVED
trained_llama31_8b_sokoban_model: $TRAINED_LLAMA31_8B_SOKOBAN_RESOLVED
trained_qwen25_7b_searchr1_model: $TRAINED_QWEN25_7B_SEARCHR1_RESOLVED
vllm_host: ${VLLM_HOST:-127.0.0.1}
vllm_port: ${VLLM_PORT:-8000}
vllm_cuda_visible_devices: ${VLLM_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}
vllm_tp_size: ${VLLM_TP_SIZE:-${TP_SIZE:-1}}
vllm_gpu_memory_utilization: ${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
vllm_max_model_len: ${VLLM_MAX_MODEL_LEN:-8192}
eval_max_tokens: ${EVAL_MAX_TOKENS:-512}
eval_temperature: ${EVAL_TEMPERATURE:-0.0}
eval_max_concurrency: ${EVAL_MAX_CONCURRENCY:-8}
cross_eval_ids: ${CROSS_EVAL_IDS[*]}
EOF

run_cross_eval \
  "trained_qwen3_4b_sokoban_to_searchr1" \
  "$TRAINED_QWEN3_4B_SOKOBAN_RESOLVED" \
  "searchr1" \
  "$SEARCHR1_TEST_PARQUET"

run_cross_eval \
  "trained_llama31_8b_sokoban_to_searchr1" \
  "$TRAINED_LLAMA31_8B_SOKOBAN_RESOLVED" \
  "searchr1" \
  "$SEARCHR1_TEST_PARQUET"

run_cross_eval \
  "trained_qwen25_7b_searchr1_to_sokoban" \
  "$TRAINED_QWEN25_7B_SEARCHR1_RESOLVED" \
  "sokoban" \
  "$SOKOBAN_TEST_PARQUET"

write_summary
