#!/bin/bash
# Run Qwen VLM baseline evaluator and store baseline-prefixed outputs in eval/.

PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"


# Scene root (scene folders with meshes)
SCENE_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed"

# Query root (scene folders with topdown.png, topdown_camera.npz, output/descriptions)
# Defaults to SCENE_ROOT when empty.
QUERY_ROOT=""

# Dataset layout for loaders: 3rscan | scannet
DATASET="3rscan"

# # ScanNet example:
DATASET="scannet"
SCENE_ROOT="/media/klrshak/Backup/Datasets/scannet_scenes_100/scans"
QUERY_ROOT="/media/klrshak/Backup/Datasets/scannet_scenes_100/scans"

# Optional: restrict to subset scene IDs (space separated). Leave empty for all.
# SCENE_IDS=(41385867-a238-2435-8152-dc84ef14eae1)
SCENE_IDS=()

# RUNNING ON A SUBSET OF SCENES: (comment out if running on all scenes)
# SCENE_IDS_FILE="${SCENE_IDS_FILE:-$REPO_ROOT/playground/testing/subset_100_scene_ids.txt}"
# if [ -f "$SCENE_IDS_FILE" ]; then
#   mapfile -t SCENE_IDS < <(grep -vE '^[[:space:]]*(#|$)' "$SCENE_IDS_FILE")
#   echo "[INFO] Loaded ${#SCENE_IDS[@]} scene IDs from $SCENE_IDS_FILE"
# else
#   echo "[WARN] SCENE_IDS_FILE not found: $SCENE_IDS_FILE (running on all scenes)"
# fi

# FOV defaults per dataset
if [ "$DATASET" = "scannet" ]; then
  H_FOV_DEG=58.30   # ScanNet
  V_FOV_DEG=45.33   # ScanNet
else
  H_FOV_DEG=39.31   # 3RScan
  V_FOV_DEG=64.76   # 3RScan
fi

EXTRA_ARGS=(
  --seed 42
  --frame_policy max_visible  # Options: "first", "index", "random", "max_visible", "max_pixels", "all"
  --h_fov_deg "$H_FOV_DEG"
  --v_fov_deg "$V_FOV_DEG"
  --save_metrics "./eval/baseline_eval_metrics_qwen_${DATASET}.json"
  --log_file "./eval/baseline_eval_metrics_qwen_${DATASET}.log"
  # --resume
  # --max_frames_per_scene 10
  # --visualize
)

cd "$REPO_ROOT" || exit 1

CMD=(
  python3 -m langloc.baselines.vlm_baseline
  --root "$SCENE_ROOT"
  --dataset "$DATASET"
)

if [ -n "$QUERY_ROOT" ]; then
  CMD+=(--query_root "$QUERY_ROOT")
fi

if [ ${#SCENE_IDS[@]} -gt 0 ]; then
  CMD+=(--scene_ids "${SCENE_IDS[@]}")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
