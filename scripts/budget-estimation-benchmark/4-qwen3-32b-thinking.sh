#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT=${BASE_SCRIPT:-"${SCRIPT_DIR}/4-qwen3-32b.sh"}

if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "Base script not found: $BASE_SCRIPT" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  bash scripts/budget-estimation-benchmark/4-qwen3-32b-thinking.sh [target...]

Behavior:
  - No positional args: run all 12 jobs through `4-qwen3-32b.sh`.
  - Bare env aliases expand to both modes:
    sokoban, webshop, frozen-lake, deepcoder, searchr1, gpqa-main
  - Explicit targets like `sokoban_instant` or `sokoban_thinking` still work.
EOF
}

expand_target() {
  local raw=$1
  case "$raw" in
    -h|--help|help)
      printf '%s\n' "__HELP__"
      ;;
    all|suite)
      printf '%s\n' "__ALL__"
      ;;
    instant|thinking|all_instant|all_thinking)
      printf '%s\n' "$raw"
      ;;
    sokoban)
      printf '%s\n' "sokoban_instant" "sokoban_thinking"
      ;;
    webshop)
      printf '%s\n' "webshop_instant" "webshop_thinking"
      ;;
    frozen-lake|frozen_lake|frozenlake)
      printf '%s\n' "frozen_lake_instant" "frozen_lake_thinking"
      ;;
    deepcoder)
      printf '%s\n' "deepcoder_instant" "deepcoder_thinking"
      ;;
    searchr1|search-r1|search_r1)
      printf '%s\n' "search_r1_instant" "search_r1_thinking"
      ;;
    gpqa-main|gpqa_main|gpqa)
      printf '%s\n' "gpqa_main_instant" "gpqa_main_thinking"
      ;;
    *)
      printf '%s\n' "$raw"
      ;;
  esac
}

main() {
  local -a targets
  local raw_target
  local expanded

  if [[ $# -eq 0 ]]; then
    bash "$BASE_SCRIPT"
    return 0
  fi

  targets=()
  for raw_target in "$@"; do
    while IFS= read -r expanded; do
      if [[ "$expanded" == "__HELP__" ]]; then
        usage
        return 0
      fi
      if [[ "$expanded" == "__ALL__" ]]; then
        bash "$BASE_SCRIPT"
        return 0
      fi
      [[ -n "$expanded" ]] && targets+=("$expanded")
    done < <(expand_target "$raw_target")
  done

  bash "$BASE_SCRIPT" "${targets[@]}"
}

main "$@"
