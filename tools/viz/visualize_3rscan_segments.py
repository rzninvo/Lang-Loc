#!/usr/bin/env python3
"""CLI viewer for 3RScan meshes with instance segmentation overlays.

Core functions live in langloc.utils.mesh_segmentation; this script is
just the command-line entry point.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Dict

import numpy as np
import open3d as o3d

from langloc.utils.mesh_segmentation import build_segmented_mesh

try:
    from langloc.utils.config_loader import load_config
except ImportError:
    load_config = None  # type: ignore[assignment]


def create_segment_visualizer(
    mesh: o3d.geometry.TriangleMesh,
    obj_stats: Sequence[Dict[str, object]],
    *,
    highlight_ids: Optional[Iterable[int]] = None,
    show_bboxes: bool = True,
    window_name: str = "3RScan Segmentation",
) -> o3d.visualization.O3DVisualizer:
    """Create an O3DVisualizer with coloured segments, bboxes, and 3D labels."""
    from open3d.visualization import gui, rendering

    highlight_ids = set(highlight_ids or [])

    material = rendering.MaterialRecord()
    material.shader = "defaultLit"

    line_material = rendering.MaterialRecord()
    line_material.shader = "unlitLine"
    line_material.line_width = 1.0

    vis = o3d.visualization.O3DVisualizer(window_name, 1280, 720)
    vis.show_settings = False
    vis.add_geometry("mesh", mesh, material)

    for stats in obj_stats:
        oid = int(stats["object_id"])
        label = stats.get("label") or f"id_{oid}"
        colour = stats.get("color", (0.8, 0.8, 0.8))
        centroid = np.asarray(stats["centroid"])

        if show_bboxes and "bbox" in stats:
            bbox: o3d.geometry.AxisAlignedBoundingBox = stats["bbox"]
            vis.add_geometry(f"bbox_{oid}", bbox, line_material)

        vis.add_3d_label(centroid, f"{oid}: {label}")

        if highlight_ids and oid in highlight_ids:
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
            marker.translate(centroid)
            marker.paint_uniform_color(np.asarray(colour))
            if not marker.has_vertex_normals():
                marker.compute_vertex_normals()
            vis.add_geometry(f"marker_{oid}", marker, material)

    vis.reset_camera_to_default()
    gui.Application.instance.add_window(vis)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Visualize 3RScan instance labels in 3D.")
    parser.add_argument("--scene", required=True, help="3RScan scene folder name.")
    parser.add_argument("--root", type=Path, help="Path to the 3RScan dataset root.")
    parser.add_argument("--config", default="configs/dataset/default.yaml",
                        help="Project config path (fallback if --root missing).")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for colors.")
    parser.add_argument("--only-ids", type=int, nargs="*",
                        help="Optional list of object IDs to annotate.")
    parser.add_argument("--no-bboxes", action="store_true", help="Disable bounding boxes.")
    args = parser.parse_args()

    scene_root = args.root
    if scene_root is None:
        if load_config is None:
            raise RuntimeError("Either --root must be provided or load_config must be available.")
        cfg = load_config(args.config)
        scene_root = Path(cfg["paths"]["3rscan_dataset_path"]).expanduser()

    scene_path = scene_root / args.scene
    if not scene_path.exists():
        raise FileNotFoundError(scene_path)

    mesh_vis, obj_stats = build_segmented_mesh(scene_path, seed=args.seed, only_ids=args.only_ids)

    from open3d.visualization import gui

    gui.Application.instance.initialize()
    create_segment_visualizer(
        mesh_vis, obj_stats,
        highlight_ids=args.only_ids,
        show_bboxes=not args.no_bboxes,
    )
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
