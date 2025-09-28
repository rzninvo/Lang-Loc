#!/bin/bash

# ---------------------------------------------------------------------------------
# setup_multiple_scenes.sh
#
# Downloads, extracts, and generates keyframes for multiple ScanNet or 3RScan scenes.
#
# Usage:
#   ./scripts/setup_multiple_scenes.sh --dataset {scannet|3RScan} <config_path> [num_scenes]
#
# Examples:
#   ./scripts/setup_multiple_scenes.sh --dataset scannet config/default.yaml 20
#   ./scripts/setup_multiple_scenes.sh --dataset 3RScan config/default.yaml 10
# ---------------------------------------------------------------------------------

set -e  # Exit immediately if a command fails

# -------- ARGUMENT CHECK --------
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 --dataset {scannet|3RScan} <config_path> [num_scenes]"
    exit 1
fi

if [ "$1" != "--dataset" ]; then
    echo "[ERROR] First argument must be --dataset"
    exit 1
fi

DATASET=$2
CONFIG_PATH=$3
NUM_SCENES=${4:-20}  # Default to 20 scenes if not provided

# -------- LOAD CONFIG VALUES --------
BASE_DIR=$(python3 - <<PY "$CONFIG_PATH"
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg["paths"]["base_data_dir"])
PY
)

DATASET_PATH=$(python3 - <<PY "$CONFIG_PATH"
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg["paths"]["dataset_path"])
PY
)

RSCAN_FILE=$(python3 - <<PY "$CONFIG_PATH"
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg["paths"].get("3rscan_release_scans_file", ""))
PY
)

# -------- LOOP OVER DATASETS --------
if [ "$DATASET" == "scannet" ]; then
    for i in $(seq -w 0 $((NUM_SCENES - 1))); do
        SCENE_ID=$(printf "scene%04d_00" $((10#$i)))
        SCENE_PATH="${DATASET_PATH}/${SCENE_ID}"

        # Skip if already downloaded
        if [ -d "$SCENE_PATH" ]; then
            echo "[INFO] Skipping $SCENE_ID — already exists in $SCENE_PATH"
            continue
        fi

        echo "============================================"
        echo "[INFO] Processing $SCENE_ID..."
        echo "============================================"

        bash scripts/setup_sample_data.sh --dataset scannet "$SCENE_ID" "$CONFIG_PATH"
    done

elif [ "$DATASET" == "3RScan" ]; then
    if [ ! -f "$RSCAN_FILE" ]; then
        echo "[ERROR] 3RScan release scans file not found: $RSCAN_FILE"
        exit 1
    fi

    SCAN_IDS=$(head -n "$NUM_SCENES" "$RSCAN_FILE")

    for SCAN_ID in $SCAN_IDS; do
        SCENE_PATH="${BASE_DIR}/3RScan/${SCAN_ID}"

        # Skip if already downloaded
        if [ -d "$SCENE_PATH" ]; then
            echo "[INFO] Skipping $SCAN_ID — already exists in $SCENE_PATH"
            continue
        fi

        echo "============================================"
        echo "[INFO] Processing $SCAN_ID..."
        echo "============================================"

        bash scripts/setup_sample_data.sh --dataset 3RScan "$SCAN_ID" "$CONFIG_PATH"
    done

else
    echo "[ERROR] Unknown dataset: $DATASET (must be 'scannet' or '3RScan')"
    exit 1
fi

echo "[INFO] All requested $DATASET scenes processed."