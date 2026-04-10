#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-ragenv2}"
RUN_DIR="${RUN_DIR:-/projects/bflz/model_saving/webshop-mixed/webshop-mixed-turn-PPO-4gpu-Qwen2.5-3B-Instruct-8x16-turn9-budget2to6-mixed}"
STEP="${STEP:-100}"
ROLE="${ROLE:-actor}"
LOCAL_DIR="${LOCAL_DIR:-${RUN_DIR}/global_step_${STEP}/${ROLE}}"
TARGET_DIR="${TARGET_DIR:-${RUN_DIR}/hf_merged_step${STEP}}"
USE_CPU_INITIALIZATION="${USE_CPU_INITIALIZATION:-1}"

if [[ ! -d "${LOCAL_DIR}" ]]; then
    echo "Error: checkpoint directory not found: ${LOCAL_DIR}" >&2
    exit 1
fi

if [[ ! -f "${LOCAL_DIR}/fsdp_config.json" ]]; then
    echo "Error: missing ${LOCAL_DIR}/fsdp_config.json" >&2
    exit 1
fi

if [[ ! -d "${LOCAL_DIR}/huggingface" ]]; then
    echo "Error: missing ${LOCAL_DIR}/huggingface" >&2
    exit 1
fi

mkdir -p "${TARGET_DIR}"

if command -v conda >/dev/null 2>&1; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME}"
    set -u
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(
    python -m verl.model_merger merge
    --backend fsdp
    --local_dir "${LOCAL_DIR}"
    --target_dir "${TARGET_DIR}"
)

if [[ "${USE_CPU_INITIALIZATION}" == "1" ]]; then
    CMD+=(--use_cpu_initialization)
fi

echo "Repo root: ${REPO_ROOT}"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "Checkpoint dir: ${LOCAL_DIR}"
echo "Output dir: ${TARGET_DIR}"
printf 'Command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "Merged HF model written to: ${TARGET_DIR}"
