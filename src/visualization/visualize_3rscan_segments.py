#!/usr/bin/env python3
"""
Utility helpers to visualise 3RScan meshes with instance segmentation overlays.

The module exposes reusable functions that can be imported by other scripts
(e.g., visualize_eval_loc.py) while still supporting the original CLI viewer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
import plyfile
from sklearn.neighbors import NearestNeighbors

try:  # Optional dependency; only needed for the CLI entry point.
    from src.utils.config_loader import load_config  # type: ignore
except ImportError:  # pragma: no cover
    load_config = None  # type: ignore[assignment]

__all__ = [
    "build_segmented_mesh",
    "create_segment_visualizer",
]


def _load_segmentation(scene_path: Path, base_vertices: np.ndarray) -> Tuple[np.ndarray, Dict[int, int], Dict[int, str]]:
    """
    Assign instance IDs and semantic labels to each vertex of the textured mesh.

    Because the segmentation JSON is aligned with a simplified mesh, we rely on
    the annotated point cloud (`labels.instances.annotated.v2.ply`) and transfer
    its per-point `objectId` to the mesh via nearest-neighbour search.

    Args:
        scene_path (Path): 3RScan scene directory.
        base_vertices (np.ndarray): (N, 3) array of the original OBJ vertices.

    Returns:
        tuple[np.ndarray, dict[int, int], dict[int, str]]:
            - vert_obj: per-vertex instance IDs aligned with `base_vertices`.
            - seg_to_obj: identity mapping kept for compatibility with other code.
            - obj_to_label: objectId → human-readable label (lowercased).

    Raises:
        FileNotFoundError: If required annotation files are missing.
    """
    semseg_json = scene_path / "semseg.v2.json"
    ply_path = scene_path / "labels.instances.annotated.v2.ply"
    if not semseg_json.exists() or not ply_path.exists():
        raise FileNotFoundError(f"Missing semantic annotations or annotated point cloud next to {scene_path}")

    groups = json.loads(semseg_json.read_text())["segGroups"]
    obj_to_label: Dict[int, str] = {int(g["objectId"]): g.get("label", "").strip() for g in groups}

    ply = plyfile.PlyData.read(ply_path)
    pts = np.vstack([ply["vertex"][axis] for axis in ("x", "y", "z")]).T.astype(np.float32)
    obj_ids = np.asarray(ply["vertex"]["objectId"], dtype=np.int32)

    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(pts)
    _, idx = nn.kneighbors(base_vertices.astype(np.float32), return_distance=True)
    vert_obj = obj_ids[idx[:, 0]]

    seg_to_obj = {int(oid): int(oid) for oid in np.unique(vert_obj) if oid >= 0}
    return vert_obj.astype(np.int32), seg_to_obj, obj_to_label


def build_segmented_mesh(scene_path: Path,
                         seed: int = 7,
                         only_ids: Optional[Sequence[int]] = None) -> Tuple[o3d.geometry.TriangleMesh, List[Dict[str, object]]]:
    """
    Construct an Open3D mesh with per-vertex colors encoding instance IDs.

    The function duplicates OBJ vertices per triangle so that each face can be
    rendered with an unambiguous color. It also computes per-object statistics
    (centroid, bbox, palette) used by the viewer for overlays and labels.

    Args:
        scene_path (Path): 3RScan scene directory.
        rng (np.random.Generator): Random number generator for color palette.

    Returns:
        tuple[o3d.geometry.TriangleMesh, list[dict[str, object]]]:
            - Colored mesh ready to be displayed.
            - List of per-object metadata dictionaries.
    """
    mesh_path = scene_path / "mesh.refined.v2.obj"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    expanded_verts = verts[faces.reshape(-1)]
    expanded_faces = np.arange(len(expanded_verts), dtype=np.int32).reshape(-1, 3)

    vert_seg_raw, seg_to_obj, obj_to_label = _load_segmentation(scene_path, verts)
    vert_obj = vert_seg_raw[faces.reshape(-1)]

    unique_obj_ids = sorted({oid for oid in vert_obj if oid >= 0})
    rng = np.random.default_rng(seed)
    palette = {oid: rng.uniform(0.15, 0.95, size=3) for oid in unique_obj_ids}

    colors = np.zeros((expanded_verts.shape[0], 3), dtype=np.float64)
    for oid in unique_obj_ids:
        colors[vert_obj == oid] = palette[oid]
    colors[vert_obj < 0] = np.array([0.6, 0.6, 0.6])

    mesh_vis = o3d.geometry.TriangleMesh()
    mesh_vis.vertices = o3d.utility.Vector3dVector(expanded_verts)
    mesh_vis.triangles = o3d.utility.Vector3iVector(expanded_faces)
    mesh_vis.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh_vis.compute_vertex_normals()

    obj_stats: List[Dict[str, object]] = []
    for oid in unique_obj_ids:
        vert_idx = np.nonzero(vert_obj == oid)[0]
        if vert_idx.size == 0:
            continue
        if only_ids and oid not in only_ids:
            continue
        obj_vertices = expanded_verts[vert_idx]
        centroid = obj_vertices.mean(axis=0)
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            obj_vertices.min(axis=0), obj_vertices.max(axis=0)
        )
        bbox.color = palette[oid]
        obj_stats.append(
            {
                "object_id": oid,
                "label": obj_to_label.get(oid, ""),
                "centroid": centroid,
                "bbox": bbox,
                "color": palette[oid],
                "vertex_indices": vert_idx,
            }
        )
    return mesh_vis, obj_stats


def create_segment_visualizer(mesh: o3d.geometry.TriangleMesh,
                              obj_stats: Sequence[Dict[str, object]],
                              *,
                              highlight_ids: Optional[Iterable[int]] = None,
                              show_bboxes: bool = True,
                              window_name: str = "3RScan Segmentation") -> o3d.visualization.O3DVisualizer:
    """
    Create an Open3D O3DVisualizer instance populated with coloured segments,
    per-object bounding boxes, and 3D labels.
    """
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


def parse_args():
    """Parse command-line arguments for the visualization script."""
    parser = argparse.ArgumentParser(description="Visualize 3RScan instance labels in 3D.")
    parser.add_argument("--scene", required=True, help="3RScan scene folder name.")
    parser.add_argument("--root", type=Path, help="Path to the 3RScan dataset root.")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Project config path (fallback if --root missing).")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for colors.")
    parser.add_argument(
        "--only-ids", type=int, nargs="*", help="Optional list of object IDs to annotate."
    )
    parser.add_argument("--no-bboxes", action="store_true", help="Disable bounding boxes.")
    return parser.parse_args()


def main():
    """Program entry: load configuration, build mesh, and start the viewer."""
    args = parse_args()
    scene_root = args.root
    if scene_root is None:
        if load_config is None:
            raise RuntimeError("Either --root must be provided or load_config must be available.")
        cfg = load_config(args.config)
        scene_root = Path(cfg["paths"]["3rscan_dataset_path"]).expanduser()

    dataset_root = scene_root
    scene_path = dataset_root / args.scene
    if not scene_path.exists():
        raise FileNotFoundError(scene_path)

    mesh_vis, obj_stats = build_segmented_mesh(scene_path, seed=args.seed, only_ids=args.only_ids)

    from open3d.visualization import gui

    gui.Application.instance.initialize()
    vis = create_segment_visualizer(
        mesh_vis, obj_stats,
        highlight_ids=args.only_ids,
        show_bboxes=not args.no_bboxes,
    )
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
