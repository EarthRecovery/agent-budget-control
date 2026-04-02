#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT=${BASE_SCRIPT:-"${SCRIPT_DIR}/../4-qwen3-32b.sh"}

if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "Base script not found: $BASE_SCRIPT" >&2
  exit 1
fi

exec bash "$BASE_SCRIPT" frozen_lake_thinking
