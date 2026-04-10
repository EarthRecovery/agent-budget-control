#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/utils/download_model_from_hf.sh <repo_id> <target_dir> [hf_api_key]

Arguments:
  repo_id      Hugging Face repo id like username/model-name or org/model-name
  target_dir   Local directory to download into
  hf_api_key   Optional Hugging Face access token. Use "-" to skip token.

Behavior:
  - Downloads the full model repo into target_dir
  - Works for public repos without a token
  - Can also download private/gated repos when a valid token is provided

Environment variables:
  CONDA_ENV_NAME   Optional conda env to activate first. Example: ragenv2
  REVISION         Optional branch/tag/commit to download
  HF_REPO_TYPE     Optional repo type. Default: model
  ALLOW_PATTERNS   Optional comma-separated allow patterns
  IGNORE_PATTERNS  Optional comma-separated ignore patterns
  DRY_RUN          Set to 1 to print the download plan only

Examples:
  bash scripts/utils/download_model_from_hf.sh \
    ylin30/webshop-mixed-step100 \
    /projects/bflz/downloaded_models/webshop-mixed-step100

  bash scripts/utils/download_model_from_hf.sh \
    meta-llama/Llama-3.2-3B-Instruct \
    /projects/bflz/downloaded_models/llama32-3b \
    hf_xxx_your_token
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 1
fi

REPO_ID="$1"
TARGET_DIR="$2"
HF_TOKEN_INPUT="${3:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
REVISION="${REVISION:-}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"
ALLOW_PATTERNS="${ALLOW_PATTERNS:-}"
IGNORE_PATTERNS="${IGNORE_PATTERNS:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${REPO_ID// }" ]]; then
    echo "Error: repo_id is empty." >&2
    exit 1
fi

if [[ -z "${TARGET_DIR// }" ]]; then
    echo "Error: target_dir is empty." >&2
    exit 1
fi

if [[ "${HF_TOKEN_INPUT}" == "-" ]]; then
    HF_TOKEN_INPUT=""
fi

if [[ -n "${CONDA_ENV_NAME}" ]] && command -v conda >/dev/null 2>&1; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME}"
    set -u
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export REPO_ID
export TARGET_DIR
export HF_TOKEN_INPUT
export REVISION
export HF_REPO_TYPE
export ALLOW_PATTERNS
export IGNORE_PATTERNS
export DRY_RUN
export HF_HUB_DISABLE_TELEMETRY=1

python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_patterns(raw: str):
    raw = raw.strip()
    if not raw:
        return None
    parts = [item.strip() for item in raw.split(",")]
    return [item for item in parts if item]


repo_id = os.environ["REPO_ID"].strip()
target_dir = Path(os.environ["TARGET_DIR"]).expanduser().resolve()
token = os.environ.get("HF_TOKEN_INPUT", "").strip() or None
revision = os.environ.get("REVISION", "").strip() or None
repo_type = os.environ.get("HF_REPO_TYPE", "model").strip() or "model"
allow_patterns = parse_patterns(os.environ.get("ALLOW_PATTERNS", ""))
ignore_patterns = parse_patterns(os.environ.get("IGNORE_PATTERNS", ""))
dry_run = os.environ.get("DRY_RUN", "0") == "1"

print(f"Repo id: {repo_id}")
print(f"Repo type: {repo_type}")
print(f"Target dir: {target_dir}")
print(f"Revision: {revision or 'default'}")
print(f"Using token: {'yes' if token else 'no'}")
if allow_patterns:
    print(f"Allow patterns: {allow_patterns}")
if ignore_patterns:
    print(f"Ignore patterns: {ignore_patterns}")

if dry_run:
    print("DRY_RUN=1, skipping remote download.")
    raise SystemExit(0)

target_dir.mkdir(parents=True, exist_ok=True)

download_path = snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    token=token,
    revision=revision,
    local_dir=str(target_dir),
    allow_patterns=allow_patterns,
    ignore_patterns=ignore_patterns,
)

print(f"Download complete: {download_path}")
PY
