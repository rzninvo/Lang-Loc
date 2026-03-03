#!/usr/bin/env python3
"""Aggregate per-scene JSON scene graphs into a single .pt file.

Collects all ``*_scene_graph.json`` files under a dataset root and merges
them into a single ``torch.save``-compatible dictionary keyed by scene ID.
The resulting .pt file can be loaded by the localization pipeline via
``load_scene_graphs()`` in ``langloc/localization/frame_io.py``.

Usage:
    python scripts/dataset/build_scene_graph_pt.py \\
        --root data/3RScan --output data/processed_data/generated/scene_graphs.pt

    python scripts/dataset/build_scene_graph_pt.py \\
        --root data/scans --output data/processed_data/generated/scannet_scene_graphs.pt
"""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-scene JSON scene graphs into a single .pt file."
    )
    parser.add_argument(
        "--root", type=Path, required=True,
        help="Root directory containing scene folders (e.g. data/3RScan).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output .pt file path.",
    )
    parser.add_argument(
        "--glob", type=str, default="**/output/*_scene_graph.json",
        help="Glob pattern to find scene graph JSON files (default: **/output/*_scene_graph.json).",
    )
    args = parser.parse_args()

    json_files = sorted(args.root.glob(args.glob))
    if not json_files:
        print(f"[ERROR] No scene graph JSON files found under {args.root} with pattern '{args.glob}'")
        return

    print(f"[INFO] Found {len(json_files)} scene graph files.")

    all_scenes = {}
    for jf in json_files:
        scene_id = jf.stem.replace("_scene_graph", "")
        with open(jf) as f:
            graph = json.load(f)

        all_scenes[scene_id] = graph
        n_obj = len(graph.get("objects", {}))
        n_edge = len(graph.get("edge_lists", {}).get("relation", []))
        print(f"  {scene_id}: {n_obj} objects, {n_edge} edges")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_scenes, args.output)
    print(f"[INFO] Saved {len(all_scenes)} scene graphs → {args.output}")


if __name__ == "__main__":
    main()
