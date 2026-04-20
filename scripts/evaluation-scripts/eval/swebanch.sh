#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME:-ragenv2}"
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"$HOME/agent-budget-control"}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:$PWD/verl"

INPUT_JSON=${INPUT_JSON:-}
MODEL_PATH=${MODEL_PATH:-}
MODEL_TAG=${MODEL_TAG:-}
RUN_NAME=${RUN_NAME:-}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT_ROOT/results/evaluation-scripts/eval"}
SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-"$SCRIPT_DIR/prompts/swebanch_estimation_system.txt"}
USER_PROMPT_FILE=${USER_PROMPT_FILE:-"$SCRIPT_DIR/prompts/swebanch_estimation_user.txt"}

MAX_TURN=${MAX_TURN:-20}
MAX_CONTEXT_WINDOW_TOKENS=${MAX_CONTEXT_WINDOW_TOKENS:-131072}
MAX_SAMPLES=${MAX_SAMPLES:-}
REQUEST_BATCH_SIZE=${REQUEST_BATCH_SIZE:-16}
MAX_TOKENS=${MAX_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
DRY_RUN=${DRY_RUN:-0}

CUDA_VISIBLE_DEVICES_VALUE=${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-0}}
NUM_GPUS=${NUM_GPUS:-1}
TP_SIZE=${TP_SIZE:-1}
DTYPE=${DTYPE:-bfloat16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-32768}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-$((MAX_MODEL_LEN - MAX_TOKENS))}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
VLLM_USE_V1=${VLLM_USE_V1:-1}
WORKER_MULTIPROC_METHOD=${WORKER_MULTIPROC_METHOD:-spawn}
ENFORCE_EAGER=${ENFORCE_EAGER:-1}
ENABLE_SLEEP_MODE=${ENABLE_SLEEP_MODE:-0}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-1}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-0}
DISABLE_LOG_STATS=${DISABLE_LOG_STATS:-1}
DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE:-1}
DISABLE_MM_PREPROCESSOR_CACHE=${DISABLE_MM_PREPROCESSOR_CACHE:-1}
SKIP_TOKENIZER_INIT=${SKIP_TOKENIZER_INIT:-0}

if [[ -z "$INPUT_JSON" ]]; then
  echo "INPUT_JSON is required. Point it to a SWE-bench dialogue json such as results_128_v2_dialogues_incremental.json." >&2
  exit 1
fi

if [[ ! -f "$INPUT_JSON" ]]; then
  echo "Input json not found: $INPUT_JSON" >&2
  exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required. Point it to a local Hugging Face model directory for vLLM." >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "MODEL_PATH does not look like a HF model directory: $MODEL_PATH" >&2
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

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"

if [[ -z "$MODEL_TAG" ]]; then
  MODEL_TAG=$(basename "$MODEL_PATH")
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="swebanch-${MODEL_TAG}-token-estimation"
fi

OUTPUT_DIR=${OUTPUT_DIR:-"$RESULT_ROOT/${RUN_NAME}"}
OUTPUT_JSON=${OUTPUT_JSON:-"$OUTPUT_DIR/${RUN_NAME}.json"}
TEMP_JSON=${TEMP_JSON:-"$OUTPUT_DIR/${RUN_NAME}_pairs.json"}

mkdir -p "$OUTPUT_DIR"

CMD=(
  python scripts/budget-estimation-benchmark/run_token_estimation_vllm.py
  --input-json "$INPUT_JSON"
  --output-json "$OUTPUT_JSON"
  --temp-json "$TEMP_JSON"
  --model-path "$MODEL_PATH"
  --model-tag "$MODEL_TAG"
  --request-batch-size "$REQUEST_BATCH_SIZE"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --max-turn "$MAX_TURN"
  --max-context-window-tokens "$MAX_CONTEXT_WINDOW_TOKENS"
  --system-prompt-file "$SYSTEM_PROMPT_FILE"
  --user-prompt-file "$USER_PROMPT_FILE"
  --tensor-parallel-size "$TP_SIZE"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --max-input-tokens "$MAX_INPUT_TOKENS"
  --trust-remote-code "$TRUST_REMOTE_CODE"
  --vllm-use-v1 "$VLLM_USE_V1"
  --worker-multiproc-method "$WORKER_MULTIPROC_METHOD"
  --enforce-eager "$ENFORCE_EAGER"
  --enable-sleep-mode "$ENABLE_SLEEP_MODE"
  --enable-prefix-caching "$ENABLE_PREFIX_CACHING"
  --enable-chunked-prefill "$ENABLE_CHUNKED_PREFILL"
  --disable-log-stats "$DISABLE_LOG_STATS"
  --disable-custom-all-reduce "$DISABLE_CUSTOM_ALL_REDUCE"
  --disable-mm-preprocessor-cache "$DISABLE_MM_PREPROCESSOR_CACHE"
  --skip-tokenizer-init "$SKIP_TOKENIZER_INIT"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

echo "==> Running SWE-bench token estimation with local vLLM model"
echo "==> model_path=${MODEL_PATH}"
echo "==> input_json=${INPUT_JSON}"
echo "==> output_json=${OUTPUT_JSON}"
echo "==> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "==> num_gpus=${NUM_GPUS}, tp_size=${TP_SIZE}"
echo "==> max_input_tokens=${MAX_INPUT_TOKENS}"

"${CMD[@]}"
