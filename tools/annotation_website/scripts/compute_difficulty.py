#!/usr/bin/env python3
"""Compute a difficulty rank (1 = easiest) and tertile (0/1/2) for each
scene in the annotation pool of one dataset.

Difficulty proxy: median LangLoc-no-dialog position error from the
existing GPT-described pipeline run. Lower error = easier scene =
lower rank. The fresh-scene assignment phase later picks scenes in
ascending rank order, so annotators see the easy scenes first.

Usage:

    python scripts/compute_difficulty.py \\
        --dataset scannet \\
        --keyframes-json data/scenes_keyframes_scannet.json \\
        --metrics-json ../../eval/new_data/eval_metrics_table4_scannet_parsed_NEW.json \\
        --out data/scenes_scannet.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["3rscan", "scannet"])
    p.add_argument("--keyframes-json", type=Path, required=True,
                   help="output of prepare_keyframes.py")
    p.add_argument("--metrics-json", type=Path, default=None,
                   help="per-frame metrics from the no-dialog GPT-pipeline run "
                        "(distance_error column). Mutually exclusive with --metric-csv.")
    p.add_argument("--metric-csv", type=Path, default=None,
                   help="CSV with one row per scene; aggregate metric used as the "
                        "ranking key. Use for with-dialog A3 MAP errors etc.")
    p.add_argument("--metric-csv-col", default="a3_map_pos_err",
                   help="CSV column to use as the per-scene error (default: a3_map_pos_err).")
    p.add_argument("--metric-csv-id-col", default="scene_id",
                   help="CSV column holding the scene id (default: scene_id).")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def _per_scene_median(metrics: list, scene_ids: list[str]) -> Dict[str, float]:
    by_scene: Dict[str, List[float]] = {sid: [] for sid in scene_ids}
    for entry in metrics:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("scene_id") or entry.get("scene")
        de = entry.get("distance_error")
        if sid in by_scene and isinstance(de, (int, float)) and de == de:
            by_scene[sid].append(float(de))
    out: Dict[str, float] = {}
    for sid, errs in by_scene.items():
        if errs:
            out[sid] = float(statistics.median(errs))
    return out


def _per_scene_from_csv(
    csv_path: Path, scene_ids: list[str], id_col: str, val_col: str
) -> Dict[str, float]:
    """Read one row per scene from a CSV (e.g. the colleague's
    qwen_results_all_3.csv) and return scene_id -> error."""
    import csv as _csv
    out: Dict[str, float] = {}
    target_ids = set(scene_ids)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = _csv.DictReader(fh)
        if id_col not in reader.fieldnames:
            raise SystemExit(f"--metric-csv-id-col={id_col!r} not in {csv_path.name}; got {reader.fieldnames}")
        if val_col not in reader.fieldnames:
            raise SystemExit(f"--metric-csv-col={val_col!r} not in {csv_path.name}; got {reader.fieldnames}")
        for row in reader:
            sid = row[id_col]
            if sid not in target_ids:
                continue
            v = row[val_col]
            if v in ("", "nan", "NaN", None):
                continue
            try:
                out[sid] = float(v)
            except ValueError:
                continue
    return out


def _tertile(value: float, p33: float, p66: float) -> int:
    if value <= p33:
        return 0
    if value <= p66:
        return 1
    return 2


def main() -> None:
    args = parse_args()

    pool = json.loads(args.keyframes_json.read_text())
    scenes = pool["scenes"]
    scene_ids = [s["scene_id"] for s in scenes]

    if args.metric_csv is not None:
        medians = _per_scene_from_csv(
            args.metric_csv, scene_ids, args.metric_csv_id_col, args.metric_csv_col
        )
        rank_source = f"{args.metric_csv.name}::{args.metric_csv_col}"
    elif args.metrics_json is not None:
        metrics_data = json.loads(args.metrics_json.read_text())
        metrics_list = metrics_data["metrics"] if isinstance(metrics_data, dict) and "metrics" in metrics_data else metrics_data
        medians = _per_scene_median(metrics_list, scene_ids)
        rank_source = f"{args.metrics_json.name}::distance_error (per-frame median)"
    else:
        raise SystemExit("must pass --metrics-json (per-frame) or --metric-csv (per-scene)")
    if len(medians) < len(scene_ids):
        missing = set(scene_ids) - set(medians)
        print(
            f"[WARN] no metrics for {len(missing)} scenes; "
            f"expected={len(scene_ids)}, got={len(medians)}, "
            f"fallback=middle tertile, last rank",
            file=sys.stderr,
        )

    if medians:
        sorted_vals = sorted(medians.values())
        n = len(sorted_vals)
        p33 = sorted_vals[max(0, int(round(n * 0.33)) - 1)]
        p66 = sorted_vals[max(0, int(round(n * 0.66)) - 1)]
    else:
        p33 = p66 = float("inf")

    # Order scenes by their median error (ascending). Scenes without a
    # metric go to the back so we never assign them before scenes whose
    # difficulty we know.
    big = float("inf")
    sorted_scenes = sorted(scenes, key=lambda s: medians.get(s["scene_id"], big))

    out_scenes = []
    for rank, s in enumerate(sorted_scenes, start=1):
        sid = s["scene_id"]
        med = medians.get(sid)
        tertile = _tertile(med, p33, p66) if med is not None else 1
        out_scenes.append({
            "scene_id": sid,
            "dataset": args.dataset,
            "display_index": s["display_index"],
            "difficulty_rank": rank,
            "difficulty_tertile": tertile,
            "difficulty_median_pos_err": med,
            "frames": s["frames"],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "scenes": out_scenes,
                "tertile_thresholds": {"p33": p33, "p66": p66},
            },
            indent=2,
        )
    )
    n_zero = sum(1 for v in medians.values() if v == 0.0)
    print(
        f"[{args.dataset}] wrote {len(out_scenes)} scenes to {args.out}; "
        f"rank source: {rank_source}; "
        f"easiest = {medians.get(out_scenes[0]['scene_id'], 'n/a')}, "
        f"hardest = {medians.get(out_scenes[-1]['scene_id'], 'n/a')}; "
        f"zero-error scenes: {n_zero} / {len(medians)}; "
        f"tertile thresholds p33={p33:.3f} m, p66={p66:.3f} m"
    )


if __name__ == "__main__":
    main()
