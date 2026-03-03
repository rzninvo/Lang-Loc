#!/bin/bash

# ---------------------------------------------------------------------------------
# setup_sample_data.sh
#
# Prepares a ScanNet or 3RScan scene for annotation or model training by:
#   1. Downloading the necessary files for the given scene ID (via download_subset.sh).
#   2. Extracting RGB, depth, poses, and intrinsics (.sens for ScanNet, sequence.zip for 3RScan).
#   3. Running the keyframe selection pipeline to select a subset of high-quality,
#      diverse frames.
#   4. Optionally cleaning up intermediate raw files to save disk space.
#
# Usage:
#     ./scripts/dataset/setup_sample_data.sh --dataset scannet <scene_id>
#     ./scripts/dataset/setup_sample_data.sh --dataset 3RScan <scan_uuid>
#
# Example:
#     ./scripts/dataset/setup_sample_data.sh --dataset scannet scene0000_00
#     ./scripts/dataset/setup_sample_data.sh --dataset 3RScan 7272e161-a01b-20f6-8b5a-0b97efeb6545
# ---------------------------------------------------------------------------------

set -e  # Exit immediately if a command fails

# -------- ARGUMENT CHECK --------
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 --dataset {scannet|3RScan} <scene_id>"
    exit 1
fi

if [ "$1" != "--dataset" ]; then
    echo "[ERROR] First argument must be --dataset"
    exit 1
fi

DATASET=$2
SCAN_ID=$3

# -------- RUN DOWNLOAD & EXTRACTION --------
echo "[INFO] Step 1/3: Downloading and extracting data for $SCAN_ID (dataset=$DATASET)..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
"$SCRIPT_DIR/download_subset.sh" --dataset "$DATASET" "$SCAN_ID"

# -------- RUN KEYFRAME GENERATION --------
echo "[INFO] Step 2/3: Running keyframe generation for $SCAN_ID (dataset=$DATASET)..."

if [ "$DATASET" == "scannet" ]; then
    python3 -m langloc.dataset.frame_selection.scannetpp_best_views \
        scan_id="$SCAN_ID" \
        dataset.auto_clean=true
elif [ "$DATASET" == "3RScan" ]; then
    python3 -m langloc.dataset.frame_selection.3rscan_best_views \
        scan_id="$SCAN_ID" \
        dataset.auto_clean=true \
        dataset.debug=true
else
    echo "[ERROR] Unknown dataset: $DATASET (must be 'scannet' or '3RScan')"
    exit 1
fi

# -------- RUN DESCRIPTION GENERATION --------
echo "[INFO] Step 3/3: Generating automatic keyframe descriptions for $SCAN_ID..."

python3 -m langloc.dataset.annotation.generate_descriptions \
    scan_id="$SCAN_ID" \
    dataset.target="$DATASET"

# -------- PRINT WHERE WE SAVED THINGS --------
DATASET_PATH=$(python3 -c "
from langloc.utils.config_loader import load_config
cfg = load_config()
dataset = '$DATASET'
if dataset == 'scannet':
    print(cfg['paths']['scannet_root'])
else:
    print(cfg['paths']['rscan_root'])
")

OUTPUT_FOLDER=$(python3 -c "
from langloc.utils.config_loader import load_config
cfg = load_config()
dataset = '$DATASET'
key = 'scannetpp' if dataset == 'scannet' else '3rscan'
print(cfg['dataset'][key]['output_folder'])
")

echo "[INFO] Setup complete for $SCAN_ID (dataset=$DATASET)."
echo "[INFO] Keyframes saved in: ${DATASET_PATH}/${SCAN_ID}/${OUTPUT_FOLDER}"
