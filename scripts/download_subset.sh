#!/bin/bash

# ---------------------------------------------------------------------------------
# This script downloads selected ScanNet files for a given scan ID using the Python
# utility `download-scannet.py`. It reads configuration from `config/render_config.yaml`.
#
# Usage:
#     ./scripts/download_subset.sh <scan_id>
#
# Example:
#     ./scripts/download_subset.sh scene0000_00
# ---------------------------------------------------------------------------------

set -e

# -------- ARGUMENT CHECK --------
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <scan_id> (e.g., scene0000_00)"
    exit 1
fi

SCAN_ID=$1

# -------- LOAD CONFIG FROM PYTHON --------
CONFIG_JSON=$(python3 src/utils/config_loader.py)
OUT_DIR=$(echo "$CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['base_dir'])")
LABEL_MAP_FILE=$(echo "$CONFIG_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['label_map'])")
FILES_TO_DOWNLOAD=($(echo "$CONFIG_JSON" | python3 -c "import sys, json; print(' '.join(json.load(sys.stdin)['file_types']))"))

# -------- DOWNLOAD LABEL MAP FILE --------
if [ -f "$LABEL_MAP_FILE" ]; then
    echo "[INFO] Label map already exists at $LABEL_MAP_FILE. Skipping download."
else
    echo "[INFO] Downloading label map..."
    python3 src/utils/download_scannet.py -o "$OUT_DIR" --label_map
fi

# -------- DOWNLOAD SCAN FILES --------
echo "[INFO] Downloading files for scan: $SCAN_ID"
for file_type in "${FILES_TO_DOWNLOAD[@]}"; do
    echo "[INFO] Downloading: $file_type"
    python3 src/utils/download_scannet.py -o "$OUT_DIR" --id "$SCAN_ID" --type "$file_type"
done

echo "[INFO] Download complete for scan: $SCAN_ID"
