#!/usr/bin/env bash
set -euo pipefail
#
# Launch script for the dense-only retrieval server.
#
# Usage:
#     bash launch_server.sh [data_dir] [port]
#

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

resolve_path() {
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

# Default values
SEARCHR1_DATA_ROOT=${SEARCHR1_DATA_ROOT:-/projects/bflz/searchr1_data}
SEARCHR1_INDEX_DIR=${SEARCHR1_INDEX_DIR:-${SEARCHR1_DATA_ROOT}/search_data/prebuilt_indices}
DATA_DIR=$(resolve_path "${1:-$SEARCHR1_INDEX_DIR}")
PORT=${2:-8000}
HOST=${HOST:-127.0.0.1}
DEVICE=${DEVICE:-cpu}
GPU_MEMORY_LIMIT_MB=${GPU_MEMORY_LIMIT_MB:-1024}

echo "Starting dense-only retrieval server..."
echo "Project root: $PROJECT_ROOT"
echo "Data directory: $DATA_DIR"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Device: $DEVICE"
echo "GPU memory limit (MB): $GPU_MEMORY_LIMIT_MB"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' not found!"
    echo "Please download the search index first:"
    echo "  python scripts/download_search_index.py"
    exit 1
fi

# Check for required files
required_files=("corpus.json" "e5_Flat.index")
for file in "${required_files[@]}"; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo "Error: $file not found in $DATA_DIR"
        echo "Please run:"
        echo "  python scripts/download_search_index.py"
        echo "If part_aa and part_ab were downloaded, concatenate them with:"
        echo "  cat \"$DATA_DIR/part_aa\" \"$DATA_DIR/part_ab\" > \"$DATA_DIR/e5_Flat.index\""
        exit 1
    fi
done

# Start server
echo "Launching dense-only server..."
cd "$PROJECT_ROOT"
exec python -u scripts/retrieval/server.py \
    --data_dir "$DATA_DIR" \
    --port "$PORT" \
    --host "$HOST" \
    --device "$DEVICE" \
    --gpu_memory_limit_mb "$GPU_MEMORY_LIMIT_MB"
