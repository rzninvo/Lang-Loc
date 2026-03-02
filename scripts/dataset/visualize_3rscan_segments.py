#!/usr/bin/env python3
"""
Interactive viewer for 3RScan meshes with instance segmentation overlays.

Usage:
    python scripts/visualize_3rscan_segments.py --scene <scene_id>
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import open3d as o3d
import plyfile
from sklearn.neighbors import NearestNeighbors

from src.utils.config_loader import load_config


def _load_segmentation(scene_path: Path, base_vertices: np.ndarray):
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


def _build_colored_mesh(scene_path: Path, rng: np.random.Generator):
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
            }
        )
    return mesh_vis, obj_stats


def _setup_visualizer(mesh, obj_stats, args):
    """
    Launch the Open3D GUI window and populate it with geometry and labels.

    Args:
        mesh (o3d.geometry.TriangleMesh): Colored mesh to display.
        obj_stats (list[dict]): Metadata produced by `_build_colored_mesh`.
        args (argparse.Namespace): Parsed CLI arguments controlling filters.
    """
    from open3d.visualization import gui, rendering

    gui.Application.instance.initialize()
    vis = o3d.visualization.O3DVisualizer("3RScan Segmentation", 1280, 720)
    vis.show_settings = False

    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    vis.add_geometry("mesh", mesh, material)

    if not args.no_bboxes:
        line_material = rendering.MaterialRecord()
        line_material.shader = "unlitLine"
        line_material.line_width = 1.0
        for stats in obj_stats:
            oid = stats["object_id"]
            if args.only_ids and oid not in args.only_ids:
                continue
            vis.add_geometry(f"bbox_{oid}", stats["bbox"], line_material)

    for stats in obj_stats:
        oid = stats["object_id"]
        if args.only_ids and oid not in args.only_ids:
            continue
        label = stats["label"] or f"id_{oid}"
        vis.add_3d_label(stats["centroid"], f"{oid}: {label}")

    vis.reset_camera_to_default()
    gui.Application.instance.add_window(vis)
    gui.Application.instance.run()


def parse_args():
    """Parse command-line arguments for the visualization script."""
    parser = argparse.ArgumentParser(description="Visualize 3RScan instance labels in 3D.")
    parser.add_argument("--scene", required=True, help="3RScan scene folder name.")
    parser.add_argument("--config", default="config/default.yaml", help="Project config path.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for colors.")
    parser.add_argument(
        "--only-ids", type=int, nargs="*", help="Optional list of object IDs to annotate."
    )
    parser.add_argument("--no-bboxes", action="store_true", help="Disable bounding boxes.")
    return parser.parse_args()


def main():
    """Program entry: load configuration, build mesh, and start the viewer."""
    args = parse_args()
    cfg = load_config(args.config)
    dataset_root = Path(cfg["paths"]["3rscan_dataset_path"]).expanduser()
    scene_path = dataset_root / args.scene
    if not scene_path.exists():
        raise FileNotFoundError(scene_path)

    rng = np.random.default_rng(args.seed)
    mesh_vis, obj_stats = _build_colored_mesh(scene_path, rng)
    _setup_visualizer(mesh_vis, obj_stats, args)


if __name__ == "__main__":
    main()
