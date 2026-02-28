#!/bin/bash

# ---------------------------------------------------------------------------------
# setup_multiple_scenes.sh
#
# Downloads, extracts, and generates keyframes for multiple ScanNet or 3RScan scenes.
#
# Usage:
#   ./scripts/setup_multiple_scenes.sh --dataset {scannet|3RScan} <config_path> [num_scenes] [--source {default|scanscribe}]
#
# Examples:
#   ./scripts/setup_multiple_scenes.sh --dataset scannet config/default.yaml 20
#   ./scripts/setup_multiple_scenes.sh --dataset 3RScan config/default.yaml 10
#   ./scripts/setup_multiple_scenes.sh --dataset 3RScan config/default.yaml --source scanscribe  # Downloads all scenes from scanscribe
#   ./scripts/setup_multiple_scenes.sh --dataset 3RScan config/default.yaml  # Downloads all scenes from default source
# ---------------------------------------------------------------------------------

set -e  # Exit immediately if a command fails

# -------- ARGUMENT CHECK --------
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 --dataset {scannet|3RScan} <config_path> [num_scenes] [--source {default|scanscribe}]"
    exit 1
fi

if [ "$1" != "--dataset" ]; then
    echo "[ERROR] First argument must be --dataset"
    exit 1
fi

DATASET=$2
CONFIG_PATH=$3
NUM_SCENES="all"  # Default to "all" scenes
SOURCE="default"  # Default source

# Parse remaining arguments (can be in any order)
shift 3  # Remove first 3 arguments (--dataset, DATASET, CONFIG_PATH)

while [ "$#" -gt 0 ]; do
    case "$1" in
        --source)
            SOURCE=${2:-default}
            shift 2
            ;;
        *)
            # If it's not a flag, assume it's num_scenes
            if [[ "$1" =~ ^[0-9]+$ ]] || [ "$1" == "all" ]; then
                NUM_SCENES=$1
                shift
            else
                echo "[WARN] Unknown argument: $1"
                shift
            fi
            ;;
    esac
done

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
print(cfg["paths"]["scannet_dataset_path"])
PY
)

RSCAN_FILE=$(python3 - <<PY "$CONFIG_PATH"
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg["paths"].get("3rscan_release_scans_file", ""))
PY
)

RSCAN_PARTIAL_FILE=$(python3 - <<PY "$CONFIG_PATH"
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg["paths"].get("3rscan_partial_scans_file", ""))
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
    # Pre-download 3RScan metadata needed for scene setup
    mkdir -p data/3RScan

    if [[ ! -f "data/3RScan/3RScan.json" ]]; then
        wget "http://campar.in.tum.de/public_datasets/3RScan/3RScan.json" -P data/3RScan
    fi
    if [[ ! -f "data/3RScan/objects.json" ]]; then
        wget "http://campar.in.tum.de/public_datasets/3DSSG/3DSSG/objects.json" -P data/3RScan
    fi
    if [[ ! -f "data/3RScan/relationships.json" ]]; then
        wget "http://campar.in.tum.de/public_datasets/3DSSG/3DSSG/relationships.json" -P data/3RScan
    fi
    if [[ ! -f "data/3RScan/relationships.txt" ]]; then
        wget "http://campar.in.tum.de/public_datasets/3DSSG/3DSSG/relationships.txt" -P data/3RScan
    fi

    # Determine source of scene IDs
    if [ "$SOURCE" == "scanscribe" ]; then
        SCANSCRIBE_FILE="config/scanscribe_cleaned.json"
        if [ ! -f "$SCANSCRIBE_FILE" ]; then
            echo "[ERROR] ScanScribe file not found: $SCANSCRIBE_FILE"
            exit 1
        fi

        echo "[INFO] Using ScanScribe cleaned dataset: $SCANSCRIBE_FILE"

        # Extract scene IDs (JSON keys) from scanscribe_cleaned.json
        SCAN_IDS=$(python3 - <<PY "$SCANSCRIBE_FILE" "$NUM_SCENES"
import sys, json
with open(sys.argv[1]) as f:
    data = json.load(f)
num_scenes = sys.argv[2]
if num_scenes == "all":
    scene_ids = list(data.keys())
else:
    scene_ids = list(data.keys())[:int(num_scenes)]
for scene_id in scene_ids:
    print(scene_id)
PY
)
        TOTAL_SCENES=$(echo "$SCAN_IDS" | wc -l)
        echo "[INFO] Found $TOTAL_SCENES scenes in ScanScribe dataset"
    else
        if [ ! -f "$RSCAN_FILE" ]; then
            echo "[ERROR] 3RScan release scans file not found: $RSCAN_FILE"
            exit 1
        fi

        echo "[INFO] Using default 3RScan release scans: $RSCAN_FILE"

        if [ "$NUM_SCENES" == "all" ]; then
            SCAN_IDS=$(cat "$RSCAN_FILE")
            TOTAL_SCENES=$(echo "$SCAN_IDS" | wc -l)
            echo "[INFO] Found $TOTAL_SCENES scenes in release scans file"
        else
            SCAN_IDS=$(head -n "$NUM_SCENES" "$RSCAN_FILE")
        fi
    fi

    # Filter known partial/incomplete scans.
    if [ -n "$RSCAN_PARTIAL_FILE" ] && [ -f "$RSCAN_PARTIAL_FILE" ]; then
        BEFORE_FILTER=$(echo "$SCAN_IDS" | sed '/^\s*$/d' | wc -l)
        CLEAN_PARTIAL_IDS=$(mktemp)
        sed -e 's/#.*$//' -e '/^[[:space:]]*$/d' "$RSCAN_PARTIAL_FILE" > "$CLEAN_PARTIAL_IDS"
        SCAN_IDS=$(echo "$SCAN_IDS" | grep -vxF -f "$CLEAN_PARTIAL_IDS" || true)
        rm -f "$CLEAN_PARTIAL_IDS"
        AFTER_FILTER=$(echo "$SCAN_IDS" | sed '/^\s*$/d' | wc -l)
        REMOVED=$((BEFORE_FILTER - AFTER_FILTER))
        echo "[INFO] Filtered partial 3RScan IDs using $RSCAN_PARTIAL_FILE: removed $REMOVED, kept $AFTER_FILTER."
    elif [ -n "$RSCAN_PARTIAL_FILE" ]; then
        echo "[WARN] Partial scans file not found: $RSCAN_PARTIAL_FILE (skipping partial-scan filtering)."
    fi

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
