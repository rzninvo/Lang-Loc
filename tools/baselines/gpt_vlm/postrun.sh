#!/usr/bin/env bash
# Post-run pipeline for the GPT-5.5 vision rebuttal experiment.
#
# After tools/baselines/gpt_vlm/run_descriptions.py has filled
#   eval/gpt55_vlm/<dataset>/<scene>/output/descriptions/*.json
# this script:
#   1. parses each frame's free-text description into a parsed_graph,
#   2. runs the LangLoc localization pipeline (caption_source=parsed,
#      frame_policy=all) over those parsed graphs,
#   3. rebuilds tools/annotation_website/data/scenes_<dataset>.json so
#      the website ranks scenes by GPT-5.5 per-scene median position
#      error (lower = easier first).
#
# Idempotent. Safe to re-run. Does NOT restart the site.

set -euo pipefail
cd "$(dirname "$0")/../../.."

PY="${LANGLOC_PYTHON:-/home/rohamzn/miniconda3/envs/langloc/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python)"
[[ -f .env ]] && set -a && source .env && set +a

DATASETS=(scannet 3rscan)

declare -A LOC_OVERLAY=(
  [scannet]="localization=scannet"
  [3rscan]="localization=3rscan"
)

declare -A METRICS_PATH=(
  [scannet]=eval/gpt55_vlm_full_scannet_metrics.json
  [3rscan]=eval/gpt55_vlm_full_3rscan_metrics.json
)

declare -A POOL_KEYFRAMES=(
  [scannet]=tools/annotation_website/data/scenes_keyframes_scannet.json
  [3rscan]=tools/annotation_website/data/scenes_keyframes_3rscan.json
)

declare -A POOL_OUT=(
  [scannet]=tools/annotation_website/data/scenes_scannet.json
  [3rscan]=tools/annotation_website/data/scenes_3rscan.json
)

for ds in "${DATASETS[@]}"; do
  echo "================================================================"
  echo "[postrun] dataset=$ds"
  echo "================================================================"

  out_root="eval/gpt55_vlm/$ds"

  echo "[postrun] parsing descriptions in $out_root"
  "$PY" -m langloc.dataset.annotation.parse_descriptions \
      --data_root "$out_root" \
      --workers 8 \
      --seed 42

  scene_ids=$("$PY" - <<EOF
import os, sys
ds = "$ds"
root = "eval/gpt55_vlm/" + ds
ids = sorted(d for d in os.listdir(root)
             if os.path.isdir(os.path.join(root, d, "output", "descriptions"))
             and any(f.endswith("_parsed.json") for f in os.listdir(os.path.join(root, d, "output", "descriptions"))))
print(",".join(ids))
EOF
  )
  echo "[postrun] $ds: $(echo "$scene_ids" | tr ',' '\n' | wc -l) parsed scenes"

  echo "[postrun] running localization for $ds"
  "$PY" -m langloc.localization.cli \
      "${LOC_OVERLAY[$ds]}" \
      paths.query_root="$out_root" \
      localization.query_root="$out_root" \
      localization.caption_source=parsed \
      localization.frame_policy=all \
      localization.seed=42 \
      "+localization.scene_ids=[$scene_ids]" \
      localization.save_metrics="${METRICS_PATH[$ds]}" \
      localization.show_3d=false \
      localization.show_heatmap=false \
      localization.show_arrows=false \
      2>&1 | tail -20

  echo "[postrun] rebuilding ${POOL_OUT[$ds]} from ${METRICS_PATH[$ds]}"
  "$PY" tools/annotation_website/scripts/compute_difficulty.py \
      --dataset "$ds" \
      --keyframes-json "${POOL_KEYFRAMES[$ds]}" \
      --metrics-json "${METRICS_PATH[$ds]}" \
      --out "${POOL_OUT[$ds]}"

done

echo
echo "[postrun] all done."
echo "  Metrics:"
echo "    eval/gpt55_vlm_full_scannet_metrics.json"
echo "    eval/gpt55_vlm_full_3rscan_metrics.json"
echo "  Re-ranked website pools:"
echo "    tools/annotation_website/data/scenes_scannet.json"
echo "    tools/annotation_website/data/scenes_3rscan.json"
echo
echo "  Restart the site (keeps existing leases/descriptions intact):"
echo "    cd tools/annotation_website && ./launch.sh"
