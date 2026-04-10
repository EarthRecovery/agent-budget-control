#!/bin/bash
#SBATCH --job-name=webshop-origin
#SBATCH --account=bflz-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=36:00:00
#SBATCH --output=/u/ylin30/agent-budget-control/logs/slurm/webshop-origin-%j.out
#SBATCH --error=/u/ylin30/agent-budget-control/logs/slurm/webshop-origin-%j.err

set -euo pipefail

REPO_ROOT="/u/ylin30/agent-budget-control"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ragenv2}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$HOME/.config/wandb/api_key}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

echo "Wrapper start: $(date -Is)"
echo "Initial PATH: ${PATH}"

cd "$REPO_ROOT"

mkdir -p "$REPO_ROOT/logs/slurm"

CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -f "/sw/external/python/anaconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="/sw/external/python/anaconda3"
fi

echo "Resolved CONDA_BASE: ${CONDA_BASE:-unset}"

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "Error: conda.sh not found under ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
    exit 1
fi

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

configure_wandb() {
    if [[ "${WANDB_MODE}" == "disabled" || "${WANDB_MODE}" == "offline" ]]; then
        export WANDB_MODE
        echo "W&B mode: ${WANDB_MODE}"
        return
    fi

    if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
        export WANDB_API_KEY
        WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_KEY_FILE}")"
    fi

    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "W&B mode requested: online, but no WANDB_API_KEY or WANDB_KEY_FILE found. Falling back to offline."
        export WANDB_MODE=offline
        return
    fi

    if wandb login --relogin "${WANDB_API_KEY}" >/dev/null 2>&1; then
        export WANDB_MODE=online
        echo "W&B mode: online"
    else
        echo "W&B login failed. Falling back to offline."
        export WANDB_MODE=offline
    fi
}

configure_wandb

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "Python: $(command -v python)"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "PREFLIGHT_ONLY: ${PREFLIGHT_ONLY}"
python -V

python - <<'PY'
import hydra
import omegaconf
import ray
import wandb
print("Python dependency check: hydra/omegaconf/ray/wandb OK")
PY

PREFLIGHT_ONLY="${PREFLIGHT_ONLY}" bash scripts/training/mixed-budget-training-webshop-origin.sh "$@"
