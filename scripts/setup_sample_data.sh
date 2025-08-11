
#!/bin/bash

# ---------------------------------------------------------------------------------
# setup_sample_data.sh
#
# This script prepares a ScanNet scene for use in annotation or model training by:
#   1. Downloading the necessary ScanNet files for the given scene ID
#      (via download_subset.sh).
#   2. Extracting RGB, depth, poses, and intrinsics from the .sens file.
#   3. Running the keyframe selection pipeline (keyframe.py) to select
#      a subset of high-quality, diverse frames based on semantic richness,
#      3D coverage, and image sharpness.
#   4. Automatically cleaning up intermediate raw files (.sens, color, depth,
#      pose, and label folders) to save disk space.
#
#Usage:
#     ./scripts/setup_sample_data.sh <scene_id> <config_path>
#
# Example:
#     ./scripts/setup_sample_data.sh scene0000_00 config/default.yaml
#
# Requirements:
#   - The ScanNet download scripts must be available in src/utils/
#   - config_path must point to a valid YAML config
#   - download_subset.sh must be in the same directory as this script (or update path)
# ---------------------------------------------------------------------------------

set -e  # Exit immediately if a command fails

# -------- ARGUMENT CHECK --------
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <scene_id> <config_path>"
    echo "Example: $0 scene0000_00 config/default.yaml"
    exit 1
fi

SCAN_ID=$1
CONFIG_PATH=$2

# -------- RUN DOWNLOAD & EXTRACTION --------
echo "[INFO] Step 1/2: Downloading and extracting data for $SCAN_ID..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
"$SCRIPT_DIR/download_subset.sh" "$SCAN_ID"

# -------- RUN KEYFRAME GENERATION --------
echo "[INFO] Step 2/2: Running keyframe generation for $SCAN_ID..."
python3 -m src.image_generation.scannetpp_best_views "$SCAN_ID" --config "$CONFIG_PATH" --auto_clean

# -------- PRINT WHERE WE SAVED THINGS --------
DATASET_PATH=$(python3 - <<PY "$2"
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg["paths"]["dataset_path"])
PY
)

OUTPUT_FOLDER=$(python3 - <<PY "$2"
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg["render"]["output_folder"])
PY
)

echo "[INFO] Setup complete for $SCAN_ID."
echo "[INFO] Keyframes saved in: ${DATASET_PATH}/${SCAN_ID}/${OUTPUT_FOLDER}"
