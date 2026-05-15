#!/usr/bin/env python3
"""Promote a fixed list of "anchor" scenes to ranks 1..N at the top of
the ScanNet annotation pool, keeping the rest in their existing order.

Use case: manifests/scannet_run2_10.txt holds the 10 ScanNet scenes
the colleague vetted as easy / well-localised in the Qwen-A3 run; we
want annotators to see those first regardless of which difficulty
metric currently drives the rest of the pool. The remaining 90 scenes
keep their relative ordering from the input pool.

Inputs:
  --pool        path to scenes_<dataset>.json to rewrite (in place by default).
  --anchors     text file, one scene_id per line. Order in the file =
                rank order at the top.
  --rank-source where the per-scene rank for the non-anchor scenes
                comes from:
                  json (default) — read difficulty_rank from --pool;
                  db             — query annotations.db (use this when
                                   --pool was just rewritten by
                                   compute_difficulty and you want the
                                   *previous* ordering instead).

Tertile thresholds are recomputed from the surviving
difficulty_median_pos_err values (anchors get tertile 0 by definition;
their original median is preserved if known, otherwise None).

Idempotent: re-running with the same anchors is a no-op.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True,
                   help="scenes_<dataset>.json file to rewrite")
    p.add_argument("--anchors", type=Path, required=True,
                   help="text file, one scene_id per line")
    p.add_argument("--rank-source", choices=["json", "db"], default="json",
                   help="where to read the non-anchor scene order from")
    p.add_argument("--db", type=Path,
                   default=Path("tools/annotation_website/data/annotations.db"),
                   help="SQLite DB path when --rank-source=db")
    p.add_argument("--out", type=Path, default=None,
                   help="output path (default: rewrite --pool in place)")
    return p.parse_args()


def _tertile(value: Optional[float], p33: float, p66: float) -> int:
    if value is None:
        return 1
    if value <= p33:
        return 0
    if value <= p66:
        return 1
    return 2


def _previous_ranks_from_db(db: Path, dataset: str) -> Dict[str, int]:
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT id, difficulty_rank FROM scenes WHERE dataset = ?",
            (dataset,),
        ).fetchall()
    return {sid: rank for sid, rank in rows}


def main() -> None:
    args = parse_args()

    pool = json.loads(args.pool.read_text())
    dataset = pool.get("dataset")
    if dataset is None:
        raise SystemExit(f"--pool {args.pool} is missing 'dataset' key")

    by_id = {s["scene_id"]: s for s in pool["scenes"]}

    anchor_ids: List[str] = []
    for ln in args.anchors.read_text().splitlines():
        parts = ln.strip().split()
        if not parts:
            continue
        first = parts[0]
        if first.startswith("scene") or (len(first) == 36 and first.count("-") == 4):
            anchor_ids.append(first)
    if not anchor_ids:
        raise SystemExit(f"--anchors {args.anchors} produced no scene IDs after filtering")

    missing = [s for s in anchor_ids if s not in by_id]
    if missing:
        raise SystemExit(
            f"anchors not found in pool {args.pool}: {missing}. "
            f"Pool has {len(by_id)} scenes."
        )

    if args.rank_source == "db":
        prev_ranks = _previous_ranks_from_db(args.db, dataset)
    else:
        prev_ranks = {s["scene_id"]: s.get("difficulty_rank", 999) for s in pool["scenes"]}

    non_anchors = [sid for sid in by_id if sid not in set(anchor_ids)]
    non_anchors.sort(key=lambda sid: prev_ranks.get(sid, 9999))

    new_order: List[str] = list(anchor_ids) + non_anchors
    if len(new_order) != len(by_id):
        raise SystemExit(f"order length mismatch: got {len(new_order)}, expected {len(by_id)}")

    medians = [
        by_id[sid].get("difficulty_median_pos_err")
        for sid in new_order
        if isinstance(by_id[sid].get("difficulty_median_pos_err"), (int, float))
    ]
    if medians:
        sorted_vals = sorted(medians)
        n = len(sorted_vals)
        p33 = sorted_vals[max(0, int(round(n * 0.33)) - 1)]
        p66 = sorted_vals[max(0, int(round(n * 0.66)) - 1)]
    else:
        p33 = p66 = float("inf")

    out_scenes = []
    for new_rank, sid in enumerate(new_order, start=1):
        s = dict(by_id[sid])
        s["difficulty_rank"] = new_rank
        if sid in set(anchor_ids):
            s["difficulty_tertile"] = 0
        else:
            s["difficulty_tertile"] = _tertile(s.get("difficulty_median_pos_err"), p33, p66)
        out_scenes.append(s)

    pool["scenes"] = out_scenes
    pool["tertile_thresholds"] = {"p33": p33, "p66": p66}
    pool["anchor_scene_ids"] = list(anchor_ids)

    out_path = args.out or args.pool
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pool, indent=2))

    print(
        f"[{dataset}] rewrote {out_path} with {len(anchor_ids)} anchors at ranks 1..{len(anchor_ids)} "
        f"and {len(non_anchors)} other scenes at ranks {len(anchor_ids)+1}..{len(out_scenes)}; "
        f"non-anchor order source: {args.rank_source}"
    )


if __name__ == "__main__":
    main()
