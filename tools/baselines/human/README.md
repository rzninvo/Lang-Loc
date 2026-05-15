# Human describer

Extracts human-written descriptions from the LangLoc annotation
website's SQLite DB and writes them into the same per-frame JSON
schema the paper pipeline expects. Then the standard parser +
localizer run on the human tree exactly like they do on the
GT-derived tree.

## Where the descriptions come from

`tools/annotation_website/` is a small FastAPI site where humans
write first-person descriptions of ScanNet / 3RScan keyframes (RGB
only — no GT scene-graph, no `visible_objects` list, no map). Each
submission is saved to `tools/annotation_website/data/annotations.db`
in the `descriptions` table. See
`tools/annotation_website/README.md` for how to run the site and
collect new annotations.

## Quick start

```bash
# 1. Extract one description per (scene, frame) into pipeline JSONs.
#    --pick longest = prefer the most-effortful description when
#    multiple humans described the same frame.
#    --skip-flagged = drop server-side-flagged entries (very-short
#    time on task; near-duplicate of self). Falls back to flagged if
#    a frame has no other description.
python tools/baselines/human/extract_descriptions.py \
    --db tools/annotation_website/data/annotations.db \
    --data-root data/scans \
    --out-root eval/human_vlm/scannet \
    --pick longest \
    --skip-flagged

# 2. Parse with gpt-4o-mini (paper protocol).
python -m langloc.dataset.annotation.parse_descriptions \
    --data_root eval/human_vlm/scannet --workers 8 --seed 42

# 3. Localize.
SCENE_IDS=$(ls eval/human_vlm/scannet | tr '\n' ',' | sed 's/,$//')
python -m langloc.localization.cli localization=scannet \
    paths.query_root=eval/human_vlm/scannet \
    localization.query_root=eval/human_vlm/scannet \
    localization.caption_source=parsed \
    localization.frame_policy=max_visible \
    localization.seed=42 \
    "+localization.scene_ids=[$SCENE_IDS]" \
    localization.save_metrics=eval/human_scannet_metrics.json
```

## Scripts

- **`extract_descriptions.py`** — pulls from the DB, writes
  pipeline-ready per-frame JSONs. Pick policies: `earliest`,
  `longest`, `all` (one JSON per (frame, annotator)). Anchor option:
  `--anchors scannet_run2_top10` restricts to the 10 anchor scenes.
- **`build_abu_tree.py`** — produces a parallel copy of someone
  else's scene tree with descriptions replaced by humans. Useful for
  sharing the human-described data with collaborators who run their
  own pipeline.

## Output

Per-frame JSON includes the same `visible_objects` /
`spatial_relations` fields the original GT-derived file had (copied
verbatim) so the parser / grounder operate on identical 3-D
metadata. The new fields are:

- `description` — the human's text
- `_describer` — `human:<first 8 chars of annotator UUID>`
- `_word_count`, `_submitted_at` — provenance
- `_flagged`, `_flag_reason` — if the kept entry was server-flagged

## What to disclose if you publish these numbers

Per the rebuttal-results report:

1. The grounder still uses GT `visible_objects` centroids — this is
   a **describer-only** ablation, not a full closed-loop break.
2. The 10 anchor scenes (`manifests/scannet_run2_10.txt`) are the
   easy tertile of the run-2 selection — absolute numbers don't
   transfer to the full 100-scene table. Quote paired Δ instead.
3. Annotator quality varies: one annotator's flagged entries averaged
   9 s/frame in the rebuttal run. `--skip-flagged` is the default for
   a reason.

See `docs/reports/2026-05-09/01_gpt55_rebuttal_results.md` (gitignored
working notes) for the full caveat list.
