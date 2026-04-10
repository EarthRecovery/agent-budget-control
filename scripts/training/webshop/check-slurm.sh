#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

usage() {
    cat <<'EOF'
Usage: bash scripts/training/check-slurm.sh <mixed|origin> [training-script-args...]

Runs local preflight checks for the webshop Slurm wrappers:
  1. bash -n on the selected .slurm wrapper
  2. bash -n on the underlying training script
  3. wrapper execution with PREFLIGHT_ONLY=1

Examples:
  bash scripts/training/check-slurm.sh mixed
  WANDB_MODE=offline bash scripts/training/check-slurm.sh origin --steps 10
EOF
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

MODE="$1"
shift

case "$MODE" in
    mixed)
        WRAPPER="$SCRIPT_DIR/submit-webshop-mixed.slurm"
        TRAIN_SCRIPT="$SCRIPT_DIR/mixed-budget-training-webshop-mixed.sh"
        ;;
    origin)
        WRAPPER="$SCRIPT_DIR/submit-webshop-origin.slurm"
        TRAIN_SCRIPT="$SCRIPT_DIR/mixed-budget-training-webshop-origin.sh"
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Error: unknown mode '$MODE'. Use 'mixed' or 'origin'." >&2
        usage
        exit 1
        ;;
esac

cd "$REPO_ROOT"

echo "Checking wrapper syntax: $WRAPPER"
bash -n "$WRAPPER"

echo "Checking training script syntax: $TRAIN_SCRIPT"
bash -n "$TRAIN_SCRIPT"

echo "Running local preflight via wrapper"
PREFLIGHT_ONLY=1 bash "$WRAPPER" "$@"

echo "Preflight completed successfully."
