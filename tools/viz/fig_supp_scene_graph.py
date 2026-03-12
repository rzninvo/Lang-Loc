#!/usr/bin/env python3
"""Generate supp_scene_graph.png: 3D scene graph with pruned radius.

Opens an interactive Open3D viewer for camera selection (press C to capture),
then renders the scene graph with object centroids, relation edges, and labels.

Usage::

    python -m tools.viz.fig_supp_scene_graph \
        --dataset scannet --root ./data/scans \
        --scan-id scene0000_01 \
        --graphs-3dssg ./data/3DSSG/graphs.pt \
        --prune-radius 1.5 \
        --output docs/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch


def _prune_graph_by_radius(
    graph_data: dict,
    center: np.ndarray,
    radius: float,
) -> List[int]:
    """Return object IDs within `radius` metres of `center`."""
    objects = graph_data.get("objects", {})
    keep = []
    for oid_str, obj in objects.items():
        oid = int(oid_str) if isinstance(oid_str, str) else oid_str
        obb = obj.get("obb")
        if obb is None:
            continue
        centroid = np.array(obb["centroid"], dtype=np.float64)
        if np.linalg.norm(centroid - center) <= radius:
            keep.append(oid)
    return keep


def parse_args():
    ap = argparse.ArgumentParser(description="Scene graph figure with interactive camera.")
    ap.add_argument("--dataset", choices=["3rscan", "scannet"], required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--scan-id", "--scan_id", dest="scan_id", required=True)
    ap.add_argument("--graphs-3dssg", "--graphs_3dssg", dest="graphs_3dssg", required=True)
    ap.add_argument("--prune-radius", "--prune_radius", dest="prune_radius",
                    type=float, default=0.0,
                    help="Prune objects beyond this radius from scene center (0 = no pruning).")
    ap.add_argument("--prune-center", "--prune_center", dest="prune_center",
                    type=float, nargs=3, default=None,
                    help="3D center for pruning (default: mesh centroid).")
    ap.add_argument("--desat", type=float, default=0.5,
                    help="Desaturation of non-matched mesh regions (0-1).")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--output", type=Path, default=Path("docs/figures"))
    ap.add_argument("--camera-json", "--camera_json", dest="camera_json", default=None,
                    help="Saved camera JSON {eye, center, up}. If omitted, interactive viewer opens.")
    return ap.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from tools.viz.visualize_teaser import (
        load_scene_any, detect_up_axis, render_scene_graph,
    )
    from PIL import Image

    scan_dir = args.root / args.scan_id
    print(f"[1/3] Loading scene: {args.scan_id}")
    mesh, tri2obj, obj2faces = load_scene_any(scan_dir, dataset=args.dataset)
    up_axis = detect_up_axis(mesh)

    print(f"[2/3] Loading 3DSSG graphs...")
    sg_all = torch.load(args.graphs_3dssg, map_location="cpu", weights_only=False)

    # Find graph for this scan
    g = None
    for key in [args.scan_id, f"3RScan/{args.scan_id}", args.scan_id.replace("3RScan/", "")]:
        if key in sg_all:
            g = sg_all[key]
            break
    if g is None:
        raise KeyError(f"Scan '{args.scan_id}' not found in graphs file")

    # Determine which objects to show
    if args.prune_radius > 0:
        center = np.array(args.prune_center, dtype=np.float64) if args.prune_center else \
            np.asarray(mesh.get_axis_aligned_bounding_box().get_center(), dtype=np.float64)
        obj_ids = _prune_graph_by_radius(g, center, args.prune_radius)
        print(f"  Pruned to {len(obj_ids)} objects within {args.prune_radius}m of center")
    else:
        obj_ids = [int(k) if isinstance(k, str) else k for k in g.get("objects", {}).keys()]
        print(f"  Showing all {len(obj_ids)} objects")

    # Labels PLY (for ScanNet semantic colours)
    labels_ply = None
    if args.dataset == "scannet":
        lp = scan_dir / f"{args.scan_id}_vh_clean_2.labels.ply"
        if lp.exists():
            labels_ply = str(lp)

    interactive = args.camera_json is None

    print(f"[3/3] Rendering scene graph...")
    if interactive:
        print("  Opening interactive viewer -- navigate to desired viewpoint, press C to capture.")

    img = render_scene_graph(
        mesh, sg_all, args.scan_id,
        matched_obj_ids=obj_ids,
        obj2faces=obj2faces,
        up_axis=up_axis,
        width=args.width, height=args.height,
        camera_json=args.camera_json,
        interactive=interactive,
        desat=args.desat,
        labels_ply=labels_ply,
    )

    out_path = args.output / "supp_scene_graph.png"
    Image.fromarray(img).save(out_path, dpi=(300, 300))
    print(f"  Saved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
