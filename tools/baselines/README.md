# Describer baselines

Each sub-directory implements one alternative way to generate the
natural-language description that the LangLoc fine-localization
pipeline consumes. The point of the comparison is to isolate the
describer's contribution — keyframes, parser, grounder and localizer
are held constant.

| Describer | Where it lives | Used in |
|---|---|---|
| **GT-aware GPT-4o-mini** (paper's describer) — generates description from `visible_objects` + `spatial_relations` extracted from the 3DSSG ground-truth scene graph. | [`langloc/dataset/annotation/`](../../langloc/dataset/annotation/) (not in this directory — it's the original paper pipeline). | Paper Table 4 main results. |
| **GPT-5.5 vision** (April 2026) — one image → 2–4 sentence first-person description. No GT metadata. | [`gpt_vlm/`](gpt_vlm/) | Rebuttal §6 (closed-loop-bias defense). |
| **Humans** typing first-person descriptions via the public annotation website. | [`human/`](human/) (extracts from `tools/annotation_website/`). | Rebuttal §6 (WxoL "report results on real human-written descriptions"). |

All three describers feed the SAME parser
(`langloc.dataset.annotation.parse_descriptions`) and the SAME
localizer (`langloc.localization.cli`), so paired differences
cleanly attribute to the describer alone.

## Output schema

Every describer writes one JSON per keyframe at:

    <out_root>/<scene_id>/output/descriptions/<frame_id>.json

with the same fields used by the original paper pipeline:

```json
{
  "scene_id":          "scene0016_00",
  "image_index":       "000004",
  "scene_pose":        [[...], [...], [...], [...]],
  "visible_objects":   { ... copied from data/scans/<scene>/output/descriptions/<frame>.json ... },
  "spatial_relations": { ... same source ... },
  "description":       "<the new free-text description>",
  "_describer":        "gpt-5.5"   # or "human:<uuid-prefix>"
}
```

`visible_objects` and `spatial_relations` are intentionally copied
from the original GT-derived files. This makes the **grounder**
(matches parsed-graph nodes back to 3-D centroids via word2vec at
γ=0.7) operate on identical inputs in every arm — only the
description text changes. See the per-sub-directory README for
how to invoke each describer.

## Reproduce the rebuttal three-way comparison

All three arms on the same 10 ScanNet anchor scenes, paper-protocol
`frame_policy=max_visible`, `caption_source=parsed`, `seed=42`:

```bash
# (a) GT-parsed baseline — already in eval/new_data/eval_metrics_table4_scannet_parsed_NEW.json

# (b) GPT-5.5 vision arm
python tools/baselines/gpt_vlm/run_descriptions.py \
    --dataset scannet --data-root data/scans \
    --manifest manifests/scannet_table4_first_100.txt \
    --out-root eval/gpt55_vlm/scannet --concurrency 8 --seed 42
python -m langloc.dataset.annotation.parse_descriptions \
    --data_root eval/gpt55_vlm/scannet --workers 8 --seed 42
python -m langloc.localization.cli localization=scannet \
    paths.query_root=eval/gpt55_vlm/scannet \
    localization.query_root=eval/gpt55_vlm/scannet \
    localization.caption_source=parsed \
    localization.frame_policy=max_visible localization.seed=42 \
    localization.save_metrics=eval/gpt55_vlm_scannet_max_visible.json

# (c) Human arm
python tools/baselines/human/extract_descriptions.py \
    --db tools/annotation_website/data/annotations.db \
    --data-root data/scans --out-root eval/human_vlm/scannet \
    --pick longest --skip-flagged
# then parser + localizer same as (b) but with paths.query_root=eval/human_vlm/scannet
```

Set `OPENAI_API_KEY` in `.env` (see `.env.example`). Total cost on
ScanNet 100 + 3RScan 97: ~$31 of GPT-5.5 vision + ~$0.30 of
gpt-4o-mini parsing.
