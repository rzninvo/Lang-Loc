#!/bin/bash

# Usage: ./download_scan.sh scene0000_00
# This script downloads selected ScanNet files for a given scan ID.

set -e

# -------- CONFIG --------
OUT_DIR="data"
LABEL_MAP_FILE="${OUT_DIR}/scannetv2-labels.combined.tsv"
FILES_TO_DOWNLOAD=(
  "_vh_clean_2.ply"
  "_vh_clean_2.labels.ply"
  "_vh_clean_2.0.010000.segs.json"
  ".aggregation.json"
  ".txt"
)

# -------- ARGUMENT CHECK --------
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <scan_id> (e.g., scene0000_00)"
    exit 1
fi

SCAN_ID=$1

# -------- DOWNLOAD LABEL MAP FILE --------
if [ -f "$LABEL_MAP_FILE" ]; then
    echo "[INFO] Label map already exists at $LABEL_MAP_FILE. Skipping download."
else
    echo "[INFO] Downloading label map..."
    python3 download-scannet.py -o "$OUT_DIR" --label_map
fi

# -------- DOWNLOAD SCAN FILES --------
echo "[INFO] Downloading files for scan: $SCAN_ID"
for file_type in "${FILES_TO_DOWNLOAD[@]}"; do
    echo "[INFO] Downloading: $file_type"
    python3 download-scannet.py -o "$OUT_DIR" --id "$SCAN_ID" --type "$file_type"
done

echo "[INFO] Download complete for scan: $SCAN_ID"
