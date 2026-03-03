#!/bin/bash

# ---------------------------------------------------------------------------------
# Download subset of ScanNet or 3RScan files for a given scan ID.
#
# Usage:
#     ./scripts/download_subset.sh --dataset scannet <scan_id>
#     ./scripts/download_subset.sh --dataset 3RScan <scan_id>
#
# Example:
#     ./scripts/download_subset.sh --dataset scannet scene0000_00
#     ./scripts/download_subset.sh --dataset 3RScan 7272e161-a01b-20f6-8b5a-0b97efeb6545
# ---------------------------------------------------------------------------------

set -e

# -------- ARGUMENT PARSING --------
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 --dataset {scannet|3RScan} <scan_id>"
    exit 1
fi

if [ "$1" != "--dataset" ]; then
    echo "[ERROR] First argument must be --dataset"
    exit 1
fi

DATASET=$2
SCAN_ID=$3

# -------- LOAD CONFIG FROM PYTHON --------
CONFIG_JSON=$(python3 -m langloc.utils.config_loader)
OUT_DIR=$(echo "$CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['base_dir'])")
LABEL_MAP_FILE=$(echo "$CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin).get('label_map', ''))")
FILES_TO_DOWNLOAD=($(echo "$CONFIG_JSON" | python3 -c "import sys, json; print(' '.join(json.load(sys.stdin).get('file_types', [])))"))

# -------- DISPATCH PER DATASET --------
if [ "$DATASET" == "scannet" ]; then
    echo "[INFO] Downloading ScanNet scan: $SCAN_ID"

    # Label map
    if [ -n "$LABEL_MAP_FILE" ]; then
        if [ -f "$LABEL_MAP_FILE" ]; then
            echo "[INFO] Label map already exists at $LABEL_MAP_FILE. Skipping download."
        else
            echo "[INFO] Downloading label map..."
            python3 tools/download_scannet.py -o "$OUT_DIR" --label_map
        fi
    fi

    # Download required files
    for file_type in "${FILES_TO_DOWNLOAD[@]}"; do
        echo "[INFO] Downloading: $file_type"
        python3 tools/download_scannet.py -o "$OUT_DIR" --id "$SCAN_ID" --type "$file_type"
    done

    # Extract from .sens
    SENS_FILE="$OUT_DIR/scans/$SCAN_ID/${SCAN_ID}.sens"
    SCAN_OUTPUT_DIR="$OUT_DIR/scans/$SCAN_ID"
    if [ -f "$SENS_FILE" ]; then
        echo "[INFO] Extracting RGB, depth, poses, and intrinsics from: $SENS_FILE"
        python3 tools/sensor_reader.py \
            --filename "$SENS_FILE" \
            --output_path "$SCAN_OUTPUT_DIR" \
            --export_depth_images \
            --export_color_images \
            --export_poses \
            --export_intrinsics
        echo "[INFO] Extraction complete: $SCAN_OUTPUT_DIR"
    else
        echo "[ERROR] .sens file not found at: $SENS_FILE"
        exit 1
    fi

elif [ "$DATASET" == "3RScan" ]; then
    if [ ! -d "$OUT_DIR/3RScan" ]; then
        mkdir -p "$OUT_DIR/3RScan"
    fi
    OUT_DIR="$OUT_DIR/3RScan"
    echo "[INFO] Downloading 3RScan scan: $SCAN_ID"
    python3 tools/download_3rscan.py --id "$SCAN_ID" -o "$OUT_DIR"
    echo "[INFO] Download complete for 3RScan scan: $SCAN_ID"

    SCAN_DIR="$OUT_DIR/$SCAN_ID"
    ZIP_PATH="$SCAN_DIR/sequence.zip"

    if [ -f "$ZIP_PATH" ]; then
        echo "[INFO] Extracting sequence.zip to: $SCAN_DIR"
        # Extract into a temp dir to be robust to folder names inside the zip
        TMP_EXTRACT="$SCAN_DIR/.extract_tmp"
        mkdir -p "$TMP_EXTRACT"
        unzip -q -o "$ZIP_PATH" -d "$TMP_EXTRACT"

        # If the zip has a 'sequence/' folder, flatten it; otherwise move everything up
        if [ -d "$TMP_EXTRACT/sequence" ]; then
            shopt -s nullglob
            mv "$TMP_EXTRACT/sequence/"* "$SCAN_DIR/"
            shopt -u nullglob
        else
            shopt -s dotglob nullglob
            mv "$TMP_EXTRACT/"* "$SCAN_DIR/"
            shopt -u dotglob nullglob
        fi
        rm -rf "$TMP_EXTRACT"
        
        # Optional: keep the zip or remove it (uncomment to delete)
        rm -f "$ZIP_PATH"

        echo "[INFO] Extraction complete. Verifying key files..."
        if [ ! -f "$SCAN_DIR/_info.txt" ]; then
            echo "[WARN] _info.txt not found at scene root. Please check the zip content."
        fi

        # Quick count for sanity
        NUM_JPG=$(ls -1 "$SCAN_DIR"/frame-*.color.jpg 2>/dev/null | wc -l)
        NUM_PGM=$(ls -1 "$SCAN_DIR"/frame-*.depth.pgm 2>/dev/null | wc -l)
        NUM_POSE=$(ls -1 "$SCAN_DIR"/frame-*.pose.txt 2>/dev/null | wc -l)
        echo "[INFO] Found: ${NUM_JPG} color JPGs, ${NUM_PGM} depth PGMs, ${NUM_POSE} poses"
    else
        echo "[ERROR] sequence.zip not found at: $ZIP_PATH"
        exit 1
    fi

else
    echo "[ERROR] Unknown dataset: $DATASET (must be 'scannet' or '3RScan')"
    exit 1
fi
