#!/usr/bin/env python3
"""Extract human annotations from the LangLoc website DB into the
per-frame JSON schema the localization pipeline expects.

For every (scene_id, frame_id) in the anchor pool that has at least
one human description, write::

    {out_root}/{scene_id}/output/descriptions/{frame_id}.json

with the schema used by run_descriptions.py / parse_descriptions.py:
``scene_id``, ``image_index``, ``scene_pose``, ``visible_objects``,
``spatial_relations``, ``description``, ``_describer``. The
``visible_objects`` and ``spatial_relations`` are copied verbatim
from the original ``all_descriptions.json``-derived per-frame JSON
(same convention as the GPT-5.5 vision runner) so the parser /
grounder has 3-D centroids to recover.

Usage:

    python tools/baselines/human/extract_descriptions.py \\
        --db tools/annotation_website/data/annotations.db \\
        --data-root data/scans \\
        --out-root eval/human_vlm/scannet \\
        --pick earliest

Pick policies:
  earliest  — first submitted description per frame (default)
  longest   — description with most words
  all       — write one JSON per (frame, annotator), suffixed with
              ``__<annotator-prefix>.json`` so the parser sees them
              all (useful for inter-annotator-error variance).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True,
                   help="root containing <scene>/output/descriptions/<frame>.json "
                        "(GT-derived per-frame metadata)")
    p.add_argument("--out-root", type=Path, required=True,
                   help="parallel tree to write the human descriptions to")
    p.add_argument("--anchors", choices=["scannet_run2_top10", "all_descriptions"],
                   default="scannet_run2_top10",
                   help="which subset of human descriptions to extract")
    p.add_argument("--pick", choices=["earliest", "longest", "all"],
                   default="earliest")
    p.add_argument("--skip-flagged", action="store_true",
                   help="exclude descriptions with flagged=1 (server-side "
                        "validation flagged them: e.g. 'very short time on "
                        "task', 'high overlap with previous'). If a frame's "
                        "only descriptions are flagged, the frame is dropped.")
    return p.parse_args()


def _read_frame_meta(scene_dir: Path, frame_id: str) -> dict | None:
    p = scene_dir / "output" / "descriptions" / f"{frame_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _query_descriptions(db: Path, anchors: str, skip_flagged: bool) -> Iterable[tuple]:
    clauses = []
    if anchors == "scannet_run2_top10":
        clauses.append(
            "s.dataset='scannet' AND s.difficulty_tertile=0 "
            "AND s.difficulty_rank<=10"
        )
    if skip_flagged:
        clauses.append("d.flagged = 0")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with sqlite3.connect(db) as conn:
        rows = conn.execute(f"""
            SELECT d.scene_id, d.frame_id, d.annotator_id, d.text,
                   d.word_count, d.submitted_at, d.flagged, d.flag_reason
            FROM descriptions d
            JOIN scenes s ON s.id = d.scene_id
            {where}
            ORDER BY d.submitted_at ASC
        """).fetchall()
    return rows


def _group_per_frame(rows: list[tuple]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for sid, fid, ann, text, wc, ts, flagged, flag_reason in rows:
        groups.setdefault((sid, fid), []).append({
            "annotator_id": ann,
            "text": text,
            "word_count": wc,
            "submitted_at": ts,
            "flagged": int(flagged or 0),
            "flag_reason": flag_reason,
        })
    return groups


def main() -> int:
    args = parse_args()
    rows = list(_query_descriptions(args.db, args.anchors, args.skip_flagged))
    if not rows:
        print(f"[ERROR] no descriptions matched anchors={args.anchors}", file=sys.stderr)
        return 1
    groups = _group_per_frame(rows)
    print(
        f"[extract_human] {len(groups)} (scene, frame) pairs from "
        f"{len(rows)} descriptions; pick={args.pick}"
    )

    n_written = 0
    n_missing_meta = 0
    for (sid, fid), entries in groups.items():
        scene_dir = args.data_root / sid
        meta = _read_frame_meta(scene_dir, fid)
        if meta is None:
            print(f"[WARN] missing GT meta for {sid}/{fid}; skipping (cannot ground)",
                  file=sys.stderr)
            n_missing_meta += 1
            continue
        visible_objects = meta.get("visible_objects") or {}
        spatial_relations = meta.get("spatial_relations") or {}
        scene_pose = meta.get("scene_pose") or meta.get("pose")

        out_scene = args.out_root / sid / "output" / "descriptions"
        out_scene.mkdir(parents=True, exist_ok=True)

        if args.pick == "earliest":
            picks = [entries[0]]
            suffixes = [""]
        elif args.pick == "longest":
            picks = [max(entries, key=lambda e: e["word_count"])]
            suffixes = [""]
        else:
            picks = entries
            suffixes = [f"__{e['annotator_id'][:8]}" for e in entries]

        for entry, sfx in zip(picks, suffixes):
            obj = {
                "scene_id": sid,
                "image_index": fid,
                "scene_pose": scene_pose,
                "visible_objects": visible_objects,
                "spatial_relations": spatial_relations,
                "description": entry["text"],
                "_describer": f"human:{entry['annotator_id'][:8]}",
                "_word_count": entry["word_count"],
                "_submitted_at": entry["submitted_at"],
            }
            out_path = out_scene / f"{fid}{sfx}.json"
            out_path.write_text(json.dumps(obj, indent=2))
            n_written += 1

    print(
        f"[extract_human] wrote {n_written} JSON files under {args.out_root}; "
        f"missing GT meta on {n_missing_meta} frames"
    )
    return 0 if n_missing_meta == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
