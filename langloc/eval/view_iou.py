#!/usr/bin/env python3
"""3D View IoU evaluation for predicted camera poses.

Computes angular error and frustum-based triangle-overlap IoU between
a predicted world-space camera pose and a ground-truth frame pose.

Can be used as a library::

    from langloc.eval.view_iou import build_iou_context, compute_view_iou

Or as a CLI::

    python -m langloc.eval.view_iou --dataset-root /data --scene-id scene0000_00 \\
        --pred-position 1.0 2.0 3.0 --pred-direction 0 0 1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
try:
    import open3d as o3d
except ImportError:
    o3d = None


PREFERRED_MESH_FILES = (
    "mesh.refined.v2.obj",
    "labels.instances.annotated.v2.ply",
    "mesh.refined.ply",
    "mesh.refined.obj",
    "mesh.obj",
)


def normalize(v: np.ndarray, eps: float = 1e-9) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def discover_mesh(scene_dir: Path) -> Path:
    for name in PREFERRED_MESH_FILES:
        mesh_path = scene_dir / name
        if mesh_path.exists():
            return mesh_path
    raise FileNotFoundError(f"No known mesh file found in {scene_dir}")


def load_gt_from_frame(scene_dir: Path, frame_index: int) -> Tuple[str, Path, np.ndarray, Optional[np.ndarray]]:
    desc_dir = scene_dir / "output" / "descriptions"
    frame_paths = sorted(desc_dir.glob("frame-*.json"))
    if not frame_paths:
        raise FileNotFoundError(f"No frame JSONs found in {desc_dir}")
    if frame_index < 0 or frame_index >= len(frame_paths):
        raise IndexError(f"--frame-index {frame_index} out of range [0, {len(frame_paths) - 1}]")

    frame_path = frame_paths[frame_index]
    frame = json.loads(frame_path.read_text())
    pose = np.asarray(frame.get("scene_pose"), dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"scene_pose must be 4x4 in {frame_path}, got {pose.shape}")

    gt_pos = pose[:3, 3].astype(np.float64)
    gt_dir = normalize(pose[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64))
    frame_id = str(frame.get("image_index", frame_path.name))
    return frame_id, frame_path, gt_pos, gt_dir


def build_iou_context(scene_dir: Path) -> Tuple[o3d.t.geometry.RaycastingScene, int, np.ndarray, np.ndarray, np.ndarray]:
    if o3d is None:
        raise RuntimeError("open3d is required for IoU computation but is not installed.")

    mesh_path = discover_mesh(scene_dir)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    if mesh.is_empty():
        raise ValueError(f"Mesh is empty: {mesh_path}")

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    if len(verts) == 0 or len(tris) == 0:
        raise ValueError(f"Mesh has no usable vertices/triangles: {mesh_path}")

    tri_points = verts[tris]
    edge_a = tri_points[:, 1] - tri_points[:, 0]
    edge_b = tri_points[:, 2] - tri_points[:, 0]
    tri_cross = np.cross(edge_a, edge_b)
    tri_areas = 0.5 * np.linalg.norm(tri_cross, axis=1)
    tri_centroids = tri_points.mean(axis=1)

    raycasting_scene = o3d.t.geometry.RaycastingScene()
    mesh_id = int(raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh)))
    return raycasting_scene, mesh_id, tri_points, tri_centroids, tri_areas


def _camera_axes_from_forward(forward: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    fwd = normalize(np.asarray(forward, dtype=np.float64))
    if fwd is None:
        return None

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(fwd, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = normalize(np.cross(fwd, up))
    if right is None:
        return None

    up_ortho = normalize(np.cross(right, fwd))
    if up_ortho is None:
        return None

    return fwd, right, up_ortho


def _visible_triangles_from_view(
    cam: np.ndarray,
    forward: Optional[np.ndarray],
    hfov: float,
    vfov: float,
    raycasting_scene: o3d.t.geometry.RaycastingScene,
    mesh_id: int,
    tri_points: np.ndarray,
    tri_centroids: np.ndarray,
    near: float,
    far: Optional[float],
) -> set[int]:
    if o3d is None:
        raise RuntimeError("open3d is required for IoU computation but is not installed.")

    if forward is None:
        return set()

    axes = _camera_axes_from_forward(forward)
    if axes is None:
        return set()
    fwd_axis, right_axis, up_axis = axes

    cam = np.asarray(cam, dtype=np.float64)
    rel = tri_points - cam[None, None, :]
    fwd = rel @ fwd_axis
    right = rel @ right_axis
    up = rel @ up_axis

    near = max(float(near), 1e-4)
    in_front = np.all(fwd > near, axis=1)
    if far is not None:
        in_front &= np.all(fwd < float(far), axis=1)

    tan_h = math.tan(hfov * 0.5)
    tan_v = math.tan(vfov * 0.5)
    inside_h = np.all(np.abs(right) <= fwd * tan_h, axis=1)
    inside_v = np.all(np.abs(up) <= fwd * tan_v, axis=1)
    frustum_mask = in_front & inside_h & inside_v
    if not np.any(frustum_mask):
        return set()

    selected_idx = np.nonzero(frustum_mask)[0]
    vectors = tri_centroids[selected_idx] - cam[None, :]
    distance = np.linalg.norm(vectors, axis=1)
    valid = distance > 1e-6
    if not np.any(valid):
        return set()

    selected_idx = selected_idx[valid]
    directions = vectors[valid] / distance[valid][:, None]
    rays = np.concatenate(
        [np.repeat(cam[None, :], len(selected_idx), axis=0), directions],
        axis=1,
    ).astype(np.float32)

    cast = raycasting_scene.cast_rays(o3d.core.Tensor(rays))
    primitive_ids = np.asarray(cast["primitive_ids"].numpy())
    geometry_ids = np.asarray(cast["geometry_ids"].numpy())
    hit_mask = (primitive_ids == selected_idx) & (geometry_ids == mesh_id)
    return {int(idx) for idx in selected_idx[hit_mask]}


def compute_view_iou(
    gt_cam: np.ndarray,
    gt_dir: Optional[np.ndarray],
    pred_cam: np.ndarray,
    pred_dir: Optional[np.ndarray],
    hfov_rad: float,
    vfov_rad: float,
    raycasting_scene: o3d.t.geometry.RaycastingScene,
    mesh_id: int,
    tri_points: np.ndarray,
    tri_centroids: np.ndarray,
    tri_areas: np.ndarray,
    near: float,
    far: Optional[float],
) -> Optional[float]:
    if gt_dir is None or pred_dir is None:
        return None

    gt_visible = _visible_triangles_from_view(
        gt_cam, gt_dir, hfov_rad, vfov_rad, raycasting_scene, mesh_id, tri_points, tri_centroids, near, far
    )
    pred_visible = _visible_triangles_from_view(
        pred_cam, pred_dir, hfov_rad, vfov_rad, raycasting_scene, mesh_id, tri_points, tri_centroids, near, far
    )
    if not gt_visible and not pred_visible:
        return None

    union = gt_visible | pred_visible
    if not union:
        return None

    intersection = gt_visible & pred_visible
    inter_area = float(tri_areas[list(intersection)].sum()) if intersection else 0.0
    union_area = float(tri_areas[list(union)].sum())
    if union_area <= 1e-9:
        return None
    return inter_area / union_area


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute angular error and view IoU for a predicted world-space pose."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory containing <scene_id>/ mesh and frame JSONs.",
    )
    parser.add_argument("--scene-id", required=True, help="Scene ID folder under --dataset-root.")
    parser.add_argument(
        "--pred-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        required=True,
        help="Predicted camera position in world coordinates.",
    )
    parser.add_argument(
        "--pred-direction",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        required=True,
        help="Predicted viewing direction vector in world coordinates.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Index into sorted frame-*.json files (default: 0).",
    )
    parser.add_argument("--h-fov-deg", type=float, default=39.31, help="Horizontal FOV for IoU.")
    parser.add_argument("--v-fov-deg", type=float, default=64.76, help="Vertical FOV for IoU.")
    parser.add_argument("--near", type=float, default=0.05, help="Near plane (meters) for IoU.")
    parser.add_argument("--far", type=float, default=None, help="Optional far plane (meters) for IoU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = args.dataset_root / args.scene_id
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    pred_pos = np.asarray(args.pred_position, dtype=np.float64)
    pred_dir = normalize(np.asarray(args.pred_direction, dtype=np.float64))
    if pred_dir is None:
        raise ValueError("--pred-direction must be non-zero.")

    frame_id, frame_path, gt_pos, gt_dir = load_gt_from_frame(scene_dir, args.frame_index)
    if gt_dir is None:
        raise ValueError(f"GT forward direction is invalid in {frame_path}")

    dot = float(np.clip(np.dot(pred_dir, gt_dir), -1.0, 1.0))
    angular_error_deg = float(math.degrees(math.acos(dot)))

    raycasting_scene, mesh_id, tri_points, tri_centroids, tri_areas = build_iou_context(scene_dir)
    iou = compute_view_iou(
        gt_cam=gt_pos,
        gt_dir=gt_dir,
        pred_cam=pred_pos,
        pred_dir=pred_dir,
        hfov_rad=math.radians(float(args.h_fov_deg)),
        vfov_rad=math.radians(float(args.v_fov_deg)),
        raycasting_scene=raycasting_scene,
        mesh_id=mesh_id,
        tri_points=tri_points,
        tri_centroids=tri_centroids,
        tri_areas=tri_areas,
        near=float(args.near),
        far=args.far,
    )

    print(f"Scene: {args.scene_id}")
    print(f"Frame: {frame_id} ({frame_path})")
    print(f"GT position: {gt_pos.tolist()}")
    print(f"GT direction: {gt_dir.tolist()}")
    print(f"Pred position: {pred_pos.tolist()}")
    print(f"Pred direction (normalized): {pred_dir.tolist()}")
    print(f"Angular error (deg): {angular_error_deg:.6f}")
    if iou is None:
        print("IoU: n/a")
        print("IoU error: n/a")
    else:
        print(f"IoU: {float(iou):.6f}")
        print(f"IoU error: {float(1.0 - iou):.6f}")


if __name__ == "__main__":
    main()
