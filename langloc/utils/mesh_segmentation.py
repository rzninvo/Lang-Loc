"""Reusable helpers for 3RScan mesh instance-segmentation overlays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
import plyfile
from sklearn.neighbors import NearestNeighbors


def _load_segmentation(
    scene_path: Path, base_vertices: np.ndarray
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, str]]:
    """Assign instance IDs and semantic labels to each mesh vertex via NN transfer."""
    semseg_json = scene_path / "semseg.v2.json"
    ply_path = scene_path / "labels.instances.annotated.v2.ply"
    if not semseg_json.exists() or not ply_path.exists():
        raise FileNotFoundError(
            f"Missing semantic annotations or annotated point cloud next to {scene_path}"
        )

    groups = json.loads(semseg_json.read_text())["segGroups"]
    obj_to_label: Dict[int, str] = {
        int(g["objectId"]): g.get("label", "").strip() for g in groups
    }

    ply = plyfile.PlyData.read(ply_path)
    pts = np.vstack([ply["vertex"][axis] for axis in ("x", "y", "z")]).T.astype(np.float32)
    obj_ids = np.asarray(ply["vertex"]["objectId"], dtype=np.int32)

    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(pts)
    _, idx = nn.kneighbors(base_vertices.astype(np.float32), return_distance=True)
    vert_obj = obj_ids[idx[:, 0]]

    seg_to_obj = {int(oid): int(oid) for oid in np.unique(vert_obj) if oid >= 0}
    return vert_obj.astype(np.int32), seg_to_obj, obj_to_label


def build_segmented_mesh(
    scene_path: Path,
    seed: int = 7,
    only_ids: Optional[Sequence[int]] = None,
) -> Tuple[o3d.geometry.TriangleMesh, List[Dict[str, object]]]:
    """Construct an Open3D mesh with per-vertex colors encoding instance IDs."""
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
