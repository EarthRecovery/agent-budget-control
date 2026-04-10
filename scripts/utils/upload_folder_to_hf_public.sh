#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/utils/upload_folder_to_hf_public.sh <hf_api_key> <folder_path> [repo_name_or_repo_id]

Arguments:
  hf_api_key   Hugging Face user access token
  folder_path  Local folder to upload
  repo_name_or_repo_id
               Optional target repo name or full repo id.
               - model-name
               - username/model-name
               - org/model-name

Behavior:
  - Uploads to a public Hugging Face model repo
  - If repo_name_or_repo_id is omitted, the script resolves your username from
    the token and uses: <username>/<basename(folder_path)>
  - If only repo_name is given, the script resolves your username from the
    token and uses: <username>/<repo_name>
  - If the repo does not exist, it is created as public

Environment variables:
  CONDA_ENV_NAME   Optional conda env to activate first. Example: ragenv2
  COMMIT_MESSAGE   Optional upload commit message
  DRY_RUN          Set to 1 to print the upload plan only
  HF_USERNAME      Optional username override. Useful with DRY_RUN=1.

Examples:
  bash scripts/utils/upload_folder_to_hf_public.sh hf_xxx /path/to/hf_merged

  bash scripts/utils/upload_folder_to_hf_public.sh \
    hf_xxx \
    /path/to/hf_merged \
    webshop-mixed-step100

  bash scripts/utils/upload_folder_to_hf_public.sh \
    hf_xxx \
    /path/to/hf_merged \
    ylin30/webshop-mixed-step100
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 1
fi

HF_TOKEN="$1"
FOLDER_PATH="$2"
REPO_NAME_OR_ID="${3:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Upload model folder via script}"
DRY_RUN="${DRY_RUN:-0}"
HF_USERNAME="${HF_USERNAME:-}"

if [[ -z "${HF_TOKEN// }" ]]; then
    echo "Error: hf_api_key is empty." >&2
    exit 1
fi

if [[ ! -d "${FOLDER_PATH}" ]]; then
    echo "Error: folder does not exist: ${FOLDER_PATH}" >&2
    exit 1
fi

if [[ -n "${CONDA_ENV_NAME}" ]] && command -v conda >/dev/null 2>&1; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME}"
    set -u
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_TOKEN
export FOLDER_PATH
export REPO_NAME_OR_ID
export COMMIT_MESSAGE
export DRY_RUN
export HF_USERNAME
export HF_HUB_DISABLE_TELEMETRY=1

python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import HfApi

token = os.environ["HF_TOKEN"].strip()
folder_path = Path(os.environ["FOLDER_PATH"]).expanduser().resolve()
repo_name_or_id = os.environ.get("REPO_NAME_OR_ID", "").strip()
commit_message = os.environ.get("COMMIT_MESSAGE", "Upload model folder via script")
dry_run = os.environ.get("DRY_RUN", "0") == "1"
username_override = os.environ.get("HF_USERNAME", "").strip()

api = HfApi(token=token)

def resolve_username() -> str:
    if username_override:
        return username_override
    whoami = api.whoami(token=token)
    return whoami["name"]

username = None
repo_id = repo_name_or_id

if not repo_id:
    username = resolve_username()
    repo_id = f"{username}/{folder_path.name}"
elif "/" not in repo_id:
    username = resolve_username()
    repo_id = f"{username}/{repo_id}"

if username is None and "/" in repo_id:
    username = repo_id.split("/", 1)[0]

print(f"Resolved username: {username}")
print(f"Local folder: {folder_path}")
print(f"Target repo: {repo_id}")
print("Repo visibility: public")

if dry_run:
    print("DRY_RUN=1, skipping remote create/upload.")
    raise SystemExit(0)

api.create_repo(
    repo_id=repo_id,
    token=token,
    repo_type="model",
    private=False,
    exist_ok=True,
)

api.upload_folder(
    folder_path=str(folder_path),
    repo_id=repo_id,
    repo_type="model",
    token=token,
    commit_message=commit_message,
)

print(f"Upload complete: https://huggingface.co/{repo_id}")
PY
