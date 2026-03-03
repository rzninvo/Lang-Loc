#!/bin/bash

# ---------------------------------------------------------------------------------
# setup_sample_data.sh
#
# Prepares a ScanNet or 3RScan scene for annotation or model training by:
#   1. Downloading the necessary files for the given scene ID (via download_subset.sh).
#   2. Extracting RGB, depth, poses, and intrinsics (.sens for ScanNet, sequence.zip for 3RScan).
#   3. Running the keyframe selection pipeline (scannetpp_best_views.py) to select
#      a subset of high-quality, diverse frames.
#   4. Optionally cleaning up intermediate raw files to save disk space.
#
# Usage:
#     ./scripts/setup_sample_data.sh --dataset scannet <scene_id> <config_path>
#     ./scripts/setup_sample_data.sh --dataset 3RScan <scan_uuid> <config_path>
#
# Example:
#     ./scripts/setup_sample_data.sh --dataset scannet scene0000_00 configs/dataset/default.yaml
#     ./scripts/setup_sample_data.sh --dataset 3RScan 7272e161-a01b-20f6-8b5a-0b97efeb6545 configs/dataset/default.yaml
# ---------------------------------------------------------------------------------

set -e  # Exit immediately if a command fails

# -------- ARGUMENT CHECK --------
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 --dataset {scannet|3RScan} <scene_id> <config_path>"
    exit 1
fi

if [ "$1" != "--dataset" ]; then
    echo "[ERROR] First argument must be --dataset"
    exit 1
fi

DATASET=$2
SCAN_ID=$3
CONFIG_PATH=$4

# -------- RUN DOWNLOAD & EXTRACTION --------
echo "[INFO] Step 1/3: Downloading and extracting data for $SCAN_ID (dataset=$DATASET)..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
"$SCRIPT_DIR/download_subset.sh" --dataset "$DATASET" "$SCAN_ID"

# -------- RUN KEYFRAME GENERATION --------
echo "[INFO] Step 2/3: Running keyframe generation for $SCAN_ID (dataset=$DATASET)..."

if [ "$DATASET" == "scannet" ]; then
    python3 -m langloc.dataset.frame_selection.scannetpp_best_views \
        "$SCAN_ID" \
        --config "$CONFIG_PATH" \
        --auto_clean
elif [ "$DATASET" == "3RScan" ]; then
    python3 -m langloc.dataset.frame_selection.3rscan_best_views \
        "$SCAN_ID" \
        --config "$CONFIG_PATH" \
        --auto_clean \
        --debug
else
    echo "[ERROR] Unknown dataset: $DATASET (must be 'scannet' or '3RScan')"
    exit 1
fi

# -------- RUN DESCRIPTION GENERATION --------
echo "[INFO] Step 3/3: Generating automatic keyframe descriptions for $SCAN_ID..."

python3 -m langloc.dataset.annotation.generate_descriptions \
    "$SCAN_ID" \
    --dataset "$DATASET" \
    --config "$CONFIG_PATH"

# -------- PRINT WHERE WE SAVED THINGS --------
DATASET_PATH=$(python3 - <<PY "$CONFIG_PATH"
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg["paths"]["scannet_dataset_path"])
PY
)

if [ "$DATASET" == "3RScan" ]; then
    DATASET_PATH="${DATASET_PATH}/3RScan"
fi

OUTPUT_FOLDER=$(python3 - <<PY "$CONFIG_PATH" "$DATASET"
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
dataset = sys.argv[2]
config_key = "scannetpp" if dataset == "scannet" else "3rscan"
print(cfg[config_key]["output_folder"])
PY
)

echo "[INFO] Setup complete for $SCAN_ID (dataset=$DATASET)."
echo "[INFO] Keyframes saved in: ${DATASET_PATH}/${SCAN_ID}/${OUTPUT_FOLDER}"
