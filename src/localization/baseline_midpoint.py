"""Midpoint baseline for localization evaluation.

Places the camera at the midpoint of the scene floor (or mesh bounding box)
with a random viewing direction, then computes the same evaluation metrics
as the main localization evaluator.

Usage::

    python -m src.localization.baseline_midpoint
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

import hydra
from omegaconf import DictConfig, OmegaConf

from src.localization.grid import load_scene, extract_floor_bbox
from src.localization.frame_io import (
    camera_center_from_pose,
    ensure_query_root,
    format_args_section,
    load_frame_jsons,
    load_scene_graphs,
    select_frame,
)
from src.localization.metrics import (
    SceneMetrics,
    build_metrics_table_standard,
    compute_metrics_standard,
    compute_view_iou_error,
)


def random_forward(rng: np.random.Generator, max_pitch_deg: float) -> np.ndarray:
    """Sample a random unit direction from yaw + bounded pitch.

    Args:
        rng: Numpy random generator.
        max_pitch_deg: Maximum pitch angle in degrees (sampled uniformly in [-deg, +deg]).

    Returns:
        Unit direction vector, shape ``(3,)``.
    """
    yaw = float(rng.uniform(-math.pi, math.pi))
    max_pitch = math.radians(float(max_pitch_deg))
    pitch = float(rng.uniform(-max_pitch, max_pitch))
    cp = math.cos(pitch)
    direction = np.array([
        cp * math.cos(yaw),
        cp * math.sin(yaw),
        math.sin(pitch),
    ], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return direction / norm


def evaluate_scene(scene_id: str,
                   cfg: DictConfig,
                   rng: np.random.Generator) -> Optional[SceneMetrics]:
    """Evaluate the midpoint baseline on a single scene.

    Args:
        scene_id: 3RScan scene identifier.
        cfg: Hydra configuration.
        rng: Numpy random generator.

    Returns:
        SceneMetrics or None if the scene cannot be evaluated.
    """
    loc = cfg.localization
    scene_dir = Path(loc.root) / scene_id
    if not scene_dir.exists():
        print(f"[WARN] Scene directory missing for {scene_id} — skipped.")
        return None

    query_root = ensure_query_root(
        Path(loc.query_root) if loc.query_root else None,
        Path(loc.root),
    )
    desc_dir = query_root / scene_id / "output" / "descriptions"
    if not desc_dir.exists():
        desc_dir = scene_dir / "output" / "descriptions"
    frames = load_frame_jsons(desc_dir)
    if not frames:
        print(f"[WARN] No frame JSONs under {desc_dir} — skipped.")
        return None

    selection = select_frame(frames, loc.frame_policy, loc.frame_index, rng)
    if selection is None:
        print(f"[WARN] Frame selection failed for {scene_id} — skipped.")
        return None
    frame = selection.frame

    gt_pose = frame.get("scene_pose")
    if gt_pose is None:
        print(f"[WARN] scene_pose missing in {selection.path} — skipped.")
        return None

    pose_mat = np.asarray(gt_pose, dtype=np.float64)
    gt_cam = camera_center_from_pose(pose_mat)
    rot_cam_world = pose_mat[:3, :3]
    gt_forward = rot_cam_world @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    gt_forward_norm = float(np.linalg.norm(gt_forward))
    gt_dir = gt_forward / gt_forward_norm if gt_forward_norm > 1e-6 else None

    mesh, _tri2obj, obj2faces = load_scene(scene_dir)
    rc = o3d.t.geometry.RaycastingScene()
    mesh_id = rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    tri_pts = verts[tris]
    tri_vecs = tri_pts[:, 1] - tri_pts[:, 0]
    tri_vecs_b = tri_pts[:, 2] - tri_pts[:, 0]
    tri_cross = np.cross(tri_vecs, tri_vecs_b)
    tri_areas = 0.5 * np.linalg.norm(tri_cross, axis=1)
    tri_centroids = tri_pts.mean(axis=1)

    floor_bbox = extract_floor_bbox(scene_dir, verts, tris, obj2faces)
    if floor_bbox is not None:
        x_mid = 0.5 * (floor_bbox["x_min"] + floor_bbox["x_max"])
        y_mid = 0.5 * (floor_bbox["y_min"] + floor_bbox["y_max"])
        z_eye = floor_bbox["z_max"] + loc.eye_height
        bounds_msg = (f"floor bbox center=({x_mid:.2f}, {y_mid:.2f}) "
                      f"z_eye={z_eye:.2f} m")
    else:
        x_mid = 0.5 * float(verts[:, 0].min() + verts[:, 0].max())
        y_mid = 0.5 * float(verts[:, 1].min() + verts[:, 1].max())
        z_eye = float(verts[:, 2].min()) + loc.eye_height
        bounds_msg = (f"mesh bbox center=({x_mid:.2f}, {y_mid:.2f}) "
                      f"z_eye={z_eye:.2f} m")

    pred_cam = np.array([x_mid, y_mid, z_eye], dtype=np.float64)
    pred_dir = random_forward(rng, cfg.baseline.random_pitch_deg)

    cams = pred_cam.reshape(1, 3)
    probs = np.array([1.0], dtype=np.float64)
    _pred_idx, metrics = compute_metrics_standard(
        cams, probs, gt_cam,
        hit_radii=list(loc.hit_radii),
        mass_percentiles=list(loc.mass_percentiles),
        topk_k=loc.top_k_min_dist,
    )
    metrics.scene_id = scene_id
    metrics.frame_id = str(frame.get("image_index", selection.path.name))
    metrics.matched_objects = 0
    metrics.distance_error = float(np.linalg.norm(pred_cam - gt_cam))

    if gt_dir is not None:
        dot = float(np.clip(np.dot(gt_dir, pred_dir), -1.0, 1.0))
        metrics.angular_error_deg = float(math.degrees(math.acos(dot)))
    else:
        metrics.angular_error_deg = None

    hfov = math.radians(loc.h_fov_deg)
    vfov = math.radians(loc.v_fov_deg)
    iou_val, iou_err, _gt_set, _pred_set = compute_view_iou_error(
        gt_cam, gt_dir, pred_cam, pred_dir,
        hfov=hfov, vfov=vfov,
        rc=rc, geom_id=int(mesh_id),
        tri_pts=tri_pts, tri_centroids=tri_centroids, tri_areas=tri_areas,
        near=0.05, far=None,
    )
    metrics.iou_error = iou_err

    print(f"    baseline pose: {bounds_msg}")
    print(f"    predicted camera (midpoint): {pred_cam.tolist()} | "
          f"err={metrics.distance_error:.3f} m")
    print(f"    predicted direction (random): {pred_dir.tolist()}")
    if iou_val is not None and iou_err is not None:
        print(f"    view IoU: {iou_val:.3f} | IoU error: {iou_err:.3f}\n")
    else:
        print("    view IoU: n/a (missing direction or empty visibility)\n")

    return metrics


def run_baseline(cfg: DictConfig) -> None:
    """Run the midpoint baseline evaluation over all candidate scenes.

    Args:
        cfg: Merged Hydra configuration.
    """
    loc = cfg.localization
    params_text = format_args_section(OmegaConf.to_container(cfg, resolve=True))
    rng = np.random.default_rng(seed=cfg.seed)

    scenes = load_scene_graphs(Path(loc.graphs))
    candidate_ids = list(scenes.keys())

    if loc.visualize_scene:
        if loc.scene_ids:
            print("[WARN] localization.visualize_scene overrides localization.scene_ids.")
        if loc.visualize_scene not in scenes:
            print(f"[ERROR] Requested scene '{loc.visualize_scene}' not found.")
            return
        candidate_ids = [loc.visualize_scene]
    elif loc.scene_ids:
        scene_set = set(loc.scene_ids)
        candidate_ids = [sid for sid in candidate_ids if sid in scene_set]
    else:
        query_root = ensure_query_root(
            Path(loc.query_root) if loc.query_root else None,
            Path(loc.root),
        )
        candidate_ids = [
            sid for sid in candidate_ids
            if (query_root / sid / "output" / "descriptions").exists()
            or (Path(loc.root) / sid / "output" / "descriptions").exists()
        ]

    candidate_ids.sort()
    if loc.max_scenes is not None:
        candidate_ids = candidate_ids[:loc.max_scenes]

    print(f"Evaluating midpoint baseline on {len(candidate_ids)} scene(s)...\n")

    metrics_list: List[SceneMetrics] = []
    for idx, sid in enumerate(candidate_ids, start=1):
        print(f"[{idx:03d}/{len(candidate_ids):03d}] {sid}")
        scene_metrics = evaluate_scene(sid, cfg, rng)
        if scene_metrics is None:
            continue
        metrics_list.append(scene_metrics)
        print(f"    frame: {scene_metrics.frame_id}")
        print(f"    matches: {scene_metrics.matched_objects} | grid pts: {scene_metrics.grid_points}")
        hit_line = " | ".join(
            f"hit@{r:.2f}m: {scene_metrics.hit_masses.get(r, 0.0):.3f}"
            for r in sorted(scene_metrics.hit_masses)
        )
        print(f"    {hit_line}")
        mass_line = " | ".join(
            f"R{p:.0f}%: {scene_metrics.mass_radii.get(p, float('nan')):.3f} m"
            for p in sorted(scene_metrics.mass_radii)
        )
        if mass_line:
            print(f"    mass-radius: {mass_line}")
        ang_err = ("n/a" if scene_metrics.angular_error_deg is None
                   else f"{scene_metrics.angular_error_deg:.2f}")
        print(f"    topK{loc.top_k_min_dist} min dist: {scene_metrics.topk_min_dist:.3f} m | "
              f"dist_err: {scene_metrics.distance_error:.3f} m | ang_err: {ang_err}\n")
        if scene_metrics.iou_error is not None:
            print(f"    view IoU error: {scene_metrics.iou_error:.3f}")

    if not metrics_list:
        print("No scenes produced metrics. Nothing to report.")
        return

    hit_radii = list(loc.hit_radii)
    mass_percentiles = list(loc.mass_percentiles)
    table_text = build_metrics_table_standard(metrics_list, hit_radii,
                                              mass_percentiles, loc.top_k_min_dist)
    if table_text:
        print("Scene-level summary table -------------------------------")
        print(table_text)
        print("---------------------------------------------------------\n")

    # Aggregate statistics
    def agg(values: List[float]) -> Tuple[float, float]:
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(np.median(arr))

    hit_stats: Dict[float, Tuple[float, float]] = {}
    for r in sorted(set(hit_radii)):
        vals = [m.hit_masses.get(r, 0.0) for m in metrics_list]
        hit_stats[r] = agg(vals)

    mass_radius_stats: Dict[float, Tuple[float, float]] = {}
    for p in sorted(set(mass_percentiles)):
        vals = [m.mass_radii.get(p, float("nan")) for m in metrics_list]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            mass_radius_stats[p] = agg(vals)

    mean_topk, med_topk = agg([m.topk_min_dist for m in metrics_list])
    mean_err, med_err = agg([m.distance_error for m in metrics_list])
    ang_values = [m.angular_error_deg for m in metrics_list if m.angular_error_deg is not None]
    mean_ang: Optional[float] = None
    med_ang: Optional[float] = None
    if ang_values:
        mean_ang, med_ang = agg([float(v) for v in ang_values])
    iou_err_values = [m.iou_error for m in metrics_list if m.iou_error is not None]
    mean_iou_err: Optional[float] = None
    med_iou_err: Optional[float] = None
    if iou_err_values:
        mean_iou_err, med_iou_err = agg([float(v) for v in iou_err_values])

    agg_lines = [
        "Aggregate metrics ---------------------------------------",
        f"  TopK{loc.top_k_min_dist} min dist (m): mean={mean_topk:.3f} | median={med_topk:.3f}",
        f"  Distance error (m)      : mean={mean_err:.3f} | median={med_err:.3f}",
        "---------------------------------------------------------\n",
    ]
    for r in sorted(hit_stats):
        mean_hit, med_hit = hit_stats[r]
        agg_lines.insert(-1, f"  Hit@{r:.2f}m              : mean={mean_hit:.3f} | median={med_hit:.3f}")
    for p in sorted(mass_radius_stats):
        mean_r, med_r = mass_radius_stats[p]
        agg_lines.insert(-1, f"  Mass-radius R{p:.0f}% (m): mean={mean_r:.3f} | median={med_r:.3f}")
    if mean_ang is not None and med_ang is not None:
        agg_lines.insert(-1, f"  Angular error (deg)   : mean={mean_ang:.2f} | median={med_ang:.2f}")
    if mean_iou_err is not None and med_iou_err is not None:
        agg_lines.insert(-1, f"  View IoU error        : mean={mean_iou_err:.3f} | median={med_iou_err:.3f}")
    print("\n".join(agg_lines))

    # Write log file
    save_metrics = Path(cfg.baseline.save_metrics)
    log_file = Path(cfg.baseline.log_file)

    log_sections: List[str] = [params_text]
    if table_text:
        log_sections.extend(["Scene-level summary table", table_text])
    log_sections.append("\n".join(agg_lines))
    log_payload = "\n\n".join(log_sections).rstrip() + "\n"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(log_payload)
    print(f"Metrics summary logged to {log_file}")

    # Write per-scene JSON
    save_metrics.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "scene_id": m.scene_id,
            "frame_id": m.frame_id,
            "hit_masses": {str(k): v for k, v in m.hit_masses.items()},
            "mass_radii": {str(k): v for k, v in m.mass_radii.items()},
            "topk_min_dist": m.topk_min_dist,
            "distance_error": m.distance_error,
            "angular_error_deg": m.angular_error_deg,
            "grid_points": m.grid_points,
            "matched_objects": m.matched_objects,
            "iou_error": m.iou_error,
        }
        for m in metrics_list
    ]
    save_metrics.write_text(json.dumps(payload, indent=2))
    print(f"Per-scene metrics saved to {save_metrics}")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entry point for midpoint baseline evaluation."""
    run_baseline(cfg)


if __name__ == "__main__":
    main()
