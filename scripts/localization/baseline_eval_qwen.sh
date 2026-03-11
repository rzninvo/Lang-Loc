#!/bin/bash
# Run Qwen VLM baseline evaluator.
#
# Requires pre-rendered topdown images (run topdown_3rscan.py first).
#
# Usage:
#   bash scripts/localization/baseline_eval_qwen.sh
#
# Configure by editing the variables below or setting environment variables:
#   SCENE_ROOT=/path/to/scenes DATASET=3rscan bash scripts/localization/baseline_eval_qwen.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# --- Configure these for your setup ---

# Dataset layout: 3rscan | scannet
DATASET="${DATASET:-3rscan}"

# Scene root (scene folders with meshes + topdown images)
if [ "$DATASET" = "scannet" ]; then
  SCENE_ROOT="${SCENE_ROOT:-./data/scans}"
else
  SCENE_ROOT="${SCENE_ROOT:-./data/3RScan}"
fi

# Query root (defaults to SCENE_ROOT if empty)
QUERY_ROOT="${QUERY_ROOT:-}"

# Optional: restrict to subset scene IDs (space separated)
SCENE_IDS=()

# --- End configuration ---

# FOV values — configured in configs/eval/view_iou.yaml (same for both datasets).
H_FOV_DEG="${H_FOV_DEG:-39.31}"
V_FOV_DEG="${V_FOV_DEG:-64.76}"

cd "$REPO_ROOT" || exit 1

CMD=(
  python3 -m langloc.baselines.vlm_baseline
  --root "$SCENE_ROOT"
  --dataset "$DATASET"
  --seed 42
  --frame_policy max_visible
  --h_fov_deg "$H_FOV_DEG"
  --v_fov_deg "$V_FOV_DEG"
  --save_metrics "${EVAL_OUTPUT_DIR:-./eval}/baseline_eval_metrics_qwen_${DATASET}.json"
  --log_file "${EVAL_OUTPUT_DIR:-./eval}/baseline_eval_metrics_qwen_${DATASET}.log"
)

if [ -n "$QUERY_ROOT" ]; then
  CMD+=(--query_root "$QUERY_ROOT")
fi

if [ ${#SCENE_IDS[@]} -gt 0 ]; then
  CMD+=(--scene_ids "${SCENE_IDS[@]}")
fi

CMD+=("$@")

"${CMD[@]}"
