#!/usr/bin/env bash
# Download the example recording used by notebooks/baseline.ipynb.
#
#     ./scripts/fetch_example_data.sh [dest_dir]
#
# Destination defaults to ./data, which is git-ignored. Safe to re-run: a
# complete file is skipped and a partial one is resumed.
#
# Source: NIN V1_V4_1024_electrode_resting_state_data, hosted on GIN
# (gin.g-node.org). The revision below is pinned so the file does not change
# underneath a set of previously computed parameters.
set -euo pipefail

URL="https://gin.g-node.org/NIN/V1_V4_1024_electrode_resting_state_data/raw/bcf0f801f14e409ee12133aa305293cd32b1707f/data/L_SNR_250717/raw/NSP1_aligned.ns6"
EXPECTED_BYTES=740453187

DEST_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/data}"
DEST="$DEST_DIR/NSP1_aligned.ns6"

mkdir -p "$DEST_DIR"

if [ -f "$DEST" ]; then
    actual=$(wc -c <"$DEST" | tr -d ' ')
    if [ "$actual" -eq "$EXPECTED_BYTES" ]; then
        echo "Already present, nothing to do: $DEST"
        exit 0
    fi
    echo "Found an incomplete file ($actual of $EXPECTED_BYTES bytes), resuming"
fi

echo "Downloading NSP1_aligned.ns6 (~706 MiB) to $DEST_DIR"
curl -fL -C - -o "$DEST" "$URL"

actual=$(wc -c <"$DEST" | tr -d ' ')
if [ "$actual" -ne "$EXPECTED_BYTES" ]; then
    echo "Size mismatch: got $actual bytes, expected $EXPECTED_BYTES" >&2
    exit 1
fi

echo "Done: $DEST"
