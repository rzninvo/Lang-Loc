#!/bin/bash

# ---------------------------------------------------------------------------------
# setup_multiple_scenes.sh
#
# Downloads, extracts, and generates keyframes for multiple ScanNet scenes.
# Loops through scene0000_00 → scene0200_00 (first 200 scenes).
# Skips scenes already present in data/scans/.
#
# Usage:
#   ./scripts/setup_multiple_scenes.sh <config_path> [num_scenes]
#
# Example:
#   ./scripts/setup_multiple_scenes.sh config/default.yaml 20
#
# ---------------------------------------------------------------------------------

set -e  # Exit immediately if a command fails

# -------- ARGUMENT CHECK --------
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <config_path> [num_scenes]"
    echo "Example: $0 config/default.yaml 100"
    exit 1
fi

CONFIG_PATH=$1
NUM_SCENES=${2:-20}  # Default to 20 scenes if not provided

# -------- LOOP OVER SCENES --------
for i in $(seq -w 0 $((NUM_SCENES - 1))); do
    SCENE_ID=$(printf "scene%04d_00" $((10#$i)))
    SCENE_PATH="data/scans/$SCENE_ID"

    # Skip if already downloaded
    if [ -d "$SCENE_PATH" ]; then
        echo "[INFO] Skipping $SCENE_ID — already exists in $SCENE_PATH"
        continue
    fi

    echo "============================================"
    echo "[INFO] Processing $SCENE_ID..."
    echo "============================================"

    bash scripts/setup_sample_data.sh "$SCENE_ID" "$CONFIG_PATH"
done

echo "[INFO] All requested scenes processed."
