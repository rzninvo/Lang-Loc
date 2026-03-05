"""Unified localization evaluation pipeline with three modes.

Consolidates the previously separate evaluation scripts into a single
``evaluate_scene`` function dispatched by :class:`EvalMode`:

- **standard**: Full metrics (multi-radius Hit@r, mass-radius, Top-K
  min distance, angular error, view IoU) with softmax probability
  sharpening and single-level arrow field.
- **coarse_to_fine**: Simple metrics (GT probability, NLL, single Hit@r)
  with raw-count probabilities and iterative coarse-to-fine arrow search.
- **candidates**: No metrics computation; exports grid and FOV pose
  candidates as JSON records.
"""
from __future__ import annotations

import json
import math
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from langloc.localization.grid import (
    load_scene,
    sample_grid,
    first_hit_is_object,
    compute_visible_dirs,
)
from langloc.localization.matching import topk_matched_objects
from langloc.localization.frame_io import (
    FrameSelection,
    load_frame_jsons,
    select_frame,
    frame_to_scenegraph,
    camera_center_from_pose,
    load_scene_graphs,
    ensure_query_root,
    format_args_section,
)
from langloc.localization.prediction import (
    select_prediction_point,
    top_n_fov_poses,
    build_grid_candidates,
    build_pose_candidates,
)
from langloc.localization.metrics import (
    SceneMetrics,
    compute_metrics_standard,
    compute_metrics_simple,
    compute_view_iou_error,
    build_metrics_table_standard,
    build_metrics_table_simple,
)
from langloc.localization.coarse_search import (
    arrow_field_from_visibility,
    coarse_to_fine_arrow_search,
)
from langloc.localization.visualization import (
    colour_objects,
    colormap,
    dir_to_yaw_pitch,
    best_fov_window,
    average_direction,
    add_heatmap_markers,
    add_arrow_markers,
    create_camera_frustum,
)


class EvalMode(str, Enum):
    """Localization evaluation mode selector."""
    STANDARD = "standard"
    COARSE_TO_FINE = "coarse_to_fine"
    CANDIDATES = "candidates"


# ---------------------------------------------------------------------------
#  Configuration adapter
# ---------------------------------------------------------------------------

def _cfg_get(cfg, key, default=None):
    """Get a value from an argparse Namespace or dict-like config."""
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "__getitem__"):
        try:
            return cfg[key]
        except (KeyError, TypeError):
            pass
    return default


# ---------------------------------------------------------------------------
#  Unified evaluate_scene
# ---------------------------------------------------------------------------

def evaluate_scene(scene_id: str,
                   scene_graph,
                   mode: EvalMode,
                   cfg,
                   rng: np.random.Generator,
                   *,
                   graph_cfg=None) -> Optional[Union[SceneMetrics, Dict]]:
    """Run localization evaluation for a single scene.

    Steps shared across all modes:
      1. Load frame JSONs and select one frame.
      2. Build a caption scene graph from visible objects.
      3. Extract the ground-truth camera pose.
      4. Match caption nodes to scene objects via cosine similarity.
      5. Load the mesh, sample a dense camera grid, compute object centroids.
      6. Cast visibility rays from each grid camera to matched objects.
      7. Derive probability distribution from visibility counts.

    Mode-specific behaviour after step 7:
      - **standard**: Softmax sharpening, single-level arrows, standard
        metrics (Hit@r, mass-radius, angular error, view IoU), optional
        visualisation.
      - **coarse_to_fine**: Raw-count probabilities, multi-level arrow
        refinement, simple metrics (GT prob, NLL, Hit@r), optional
        visualisation.
      - **candidates**: Raw-count probabilities, single-level arrows,
        no metrics; returns a JSON-serialisable dict.

    Args:
        scene_id: 3RScan scene identifier.
        scene_graph: Pre-loaded :class:`SceneGraph` for this scene.
        mode: Evaluation mode.
        cfg: Configuration namespace (``argparse.Namespace``) or dict-like
            object with evaluation parameters.
        rng: NumPy random generator.
        graph_cfg: Optional graph configuration with ``embedding_type``
            and ``use_attributes``.

    Returns:
        - **standard / coarse_to_fine**: A :class:`SceneMetrics` instance,
          or ``None`` if the scene was skipped.
        - **candidates**: A JSON-serialisable dict, or ``None``.
    """
    mesh_root = Path(_cfg_get(cfg, "root"))
    scene_dir = mesh_root / scene_id
    if not scene_dir.exists():
        print(f"[WARN] Scene directory missing for {scene_id} — skipped.")
        return None

    query_root = ensure_query_root(_cfg_get(cfg, "query_root"), mesh_root)
    desc_dir = query_root / scene_id / "output" / "descriptions"
    if not desc_dir.exists():
        desc_dir = scene_dir / "output" / "descriptions"
    frames = load_frame_jsons(desc_dir)
    if not frames:
        if mode == EvalMode.CANDIDATES:
            return None
        print(f"[WARN] No frame JSONs under {desc_dir} — skipped.")
        return None

    frame_policy = _cfg_get(cfg, "frame_policy", "max_visible")
    frame_index = _cfg_get(cfg, "frame_index", 0)
    selection = select_frame(frames, frame_policy, frame_index, rng)
    if selection is None:
        print(f"[WARN] Frame selection failed for {scene_id} — skipped.")
        return None

    frame = selection.frame
    try:
        graph_kw = {}
        if graph_cfg is not None:
            graph_kw = dict(
                embedding_type=graph_cfg.embedding_type,
                use_attributes=graph_cfg.use_attributes,
            )
        caption_graph, _ = frame_to_scenegraph(frame, **graph_kw)
    except Exception as exc:
        if mode == EvalMode.CANDIDATES:
            return None
        print(f"[WARN] Failed to build caption graph for {scene_id}: {exc}")
        return None

    gt_pose = frame.get("scene_pose")
    if gt_pose is None:
        if mode != EvalMode.CANDIDATES:
            print(f"[WARN] scene_pose missing in {selection.path} — skipped.")
        return None

    pose_mat = np.asarray(gt_pose, dtype=np.float64)
    gt_cam = camera_center_from_pose(pose_mat)
    rot_cam_world = pose_mat[:3, :3]
    forward_cv = rot_cam_world @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norm_forward = np.linalg.norm(forward_cv)
    gt_dir = forward_cv / norm_forward if norm_forward > 1e-6 else None

    top_k = _cfg_get(cfg, "top_k", 25)
    obj_ids = topk_matched_objects(caption_graph, scene_graph, k=top_k)
    if not obj_ids:
        if mode != EvalMode.CANDIDATES:
            print(f"[WARN] {scene_id}: no cosine matches — skipped.")
        return None

    mesh, tri2obj, obj2faces = load_scene(scene_dir)
    rc = o3d.t.geometry.RaycastingScene()
    mesh_id = rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    verts = np.asarray(mesh.vertices)
    grid_step = _cfg_get(cfg, "grid_step", 0.25)
    eye_height = _cfg_get(cfg, "eye_height", 1.6)
    cams = sample_grid(verts, step=grid_step, z_eye=eye_height, mesh=mesh)

    xs, ys = verts[:, 0], verts[:, 1]
    gx = np.arange(xs.min(), xs.max() + 1e-4, grid_step)
    gy = np.arange(ys.min(), ys.max() + 1e-4, grid_step)
    Nx, Ny = len(gx), len(gy)

    tris_arr = np.asarray(mesh.triangles)
    centroids: Dict[int, np.ndarray] = {}
    for oid in obj_ids:
        faces = obj2faces.get(int(oid))
        if faces is None or not len(faces):
            continue
        centroids[int(oid)] = verts[np.unique(tris_arr[faces].ravel())].mean(axis=0)

    if not centroids:
        if mode != EvalMode.CANDIDATES:
            print(f"[WARN] {scene_id}: matched objects missing geometry — skipped.")
        return None

    visible_dirs = compute_visible_dirs(cams, centroids, rc, tri2obj)
    counts = np.array([len(v) for v in visible_dirs], dtype=np.int32)
    total = counts.sum()
    if total == 0:
        if mode != EvalMode.CANDIDATES:
            print(f"[WARN] {scene_id}: matched objects invisible from grid — skipped.")
        return None

    # --- Mode-specific probability computation ---
    if mode == EvalMode.STANDARD:
        tau = float(_cfg_get(cfg, "score_tau", 0.5))
        scores_f = counts.astype(np.float32) / max(tau, 1e-6)
        scores_f -= scores_f.max()
        exp_scores = np.exp(scores_f)
        probs = exp_scores / exp_scores.sum()
    else:
        probs = counts / float(total)

    h_fov_deg = _cfg_get(cfg, "h_fov_deg", 100.0)
    v_fov_deg = _cfg_get(cfg, "v_fov_deg", 60.0)
    hfov_rad = math.radians(h_fov_deg)
    vfov_rad = math.radians(v_fov_deg)
    arrow_stride = _cfg_get(cfg, "arrow_stride", 2)
    prediction_strategy = _cfg_get(cfg, "prediction_strategy", "weighted")
    cluster_bandwidth = _cfg_get(cfg, "cluster_bandwidth", 0.75)
    max_cluster_points = _cfg_get(cfg, "max_cluster_points", 512)

    # --- Mode-specific metrics ---
    if mode == EvalMode.STANDARD:
        hit_radii = _cfg_get(cfg, "hit_radii", [0.75, 1.0, 1.5, 2.0, 2.5])
        mass_percentiles = _cfg_get(cfg, "mass_percentiles", [50.0, 90.0])
        topk_min_dist_k = _cfg_get(cfg, "top_k_min_dist", 10)
        pred_idx, metrics = compute_metrics_standard(cams, probs, gt_cam,
                                                     hit_radii=hit_radii,
                                                     mass_percentiles=mass_percentiles,
                                                     topk_k=topk_min_dist_k)
    elif mode == EvalMode.COARSE_TO_FINE:
        prob_eps = _cfg_get(cfg, "prob_eps", 1e-6)
        hit_radius = _cfg_get(cfg, "hit_radius", 0.5)
        pred_idx, metrics = compute_metrics_simple(cams, probs, gt_cam,
                                                   eps=prob_eps,
                                                   hit_radius=hit_radius)
    else:
        pred_idx = int(np.argmax(probs))
        metrics = None

    if metrics is not None:
        metrics.scene_id = scene_id
        metrics.frame_id = str(frame.get("image_index", selection.path.name))
        metrics.matched_objects = len(obj_ids)

    pred_cam_prob = cams[pred_idx]

    # --- Mode-specific arrow computation ---
    arrow_positions: List[np.ndarray] = []
    arrow_dirs: List[np.ndarray] = []
    arrow_weights: List[float] = []
    refined_points = np.empty((0, 3), dtype=np.float64)
    refined_weights = np.empty((0,), dtype=np.float64)
    arrow_all_points = np.empty((0, 3), dtype=np.float64)
    arrow_all_weights = np.empty((0,), dtype=np.float64)
    arrow_all_dirs = np.empty((0, 3), dtype=np.float64)
    arrow_step_used = grid_step
    arrow_source = "arrow_field"

    if mode == EvalMode.COARSE_TO_FINE:
        coarse_grid_step = _cfg_get(cfg, "coarse_grid_step", 2.0)
        coarse_refine_levels = _cfg_get(cfg, "coarse_refine_levels", 3)
        coarse_refine_factor = _cfg_get(cfg, "coarse_refine_factor", 2.0)
        coarse_keep_ratio = _cfg_get(cfg, "coarse_keep_ratio", 0.7)
        coarse_top_k = _cfg_get(cfg, "coarse_top_k", 16)
        coarse_disable_nms = _cfg_get(cfg, "coarse_disable_nms", False)
        try:
            (arrow_positions, arrow_dirs, arrow_weights,
             arrow_step_used, refined_points, refined_weights,
             arrow_all_points, arrow_all_weights, arrow_all_dirs) = coarse_to_fine_arrow_search(
                verts=verts,
                centroids=centroids,
                rc=rc,
                tri2obj=tri2obj,
                z_eye=eye_height,
                hfov=hfov_rad,
                vfov=vfov_rad,
                base_step=coarse_grid_step,
                levels=coarse_refine_levels,
                refine_factor=coarse_refine_factor,
                keep_ratio=coarse_keep_ratio,
                top_k=coarse_top_k,
                apply_nms=not coarse_disable_nms,
            )
            arrow_source = "arrow_field_coarse"
        except Exception as exc:
            print(f"    [warn] coarse-to-fine arrow search failed ({exc}) — "
                  f"falling back to base grid.")
            arrow_positions, arrow_dirs, arrow_weights = arrow_field_from_visibility(
                cams, visible_dirs, Nx, Ny, hfov_rad, vfov_rad,
                stride=max(1, int(arrow_stride)))
            arrow_step_used = grid_step
    else:
        arrow_positions, arrow_dirs, arrow_weights = arrow_field_from_visibility(
            cams, visible_dirs, Nx, Ny, hfov_rad, vfov_rad,
            stride=max(1, int(arrow_stride)))

    # --- Candidates mode: build and return JSON record ---
    if mode == EvalMode.CANDIDATES:
        top_candidates = _cfg_get(cfg, "top_candidates", 10)
        top_pose_candidates = _cfg_get(cfg, "top_pose_candidates", 10)
        grid_point_candidates = build_grid_candidates(cams, counts, probs, top_candidates)
        fov_pose_candidates = build_pose_candidates(arrow_positions, arrow_dirs,
                                                     arrow_weights, top_pose_candidates)

        candidate_dirs_np: Optional[np.ndarray] = None
        candidate_source = "grid_visibility"
        if arrow_weights:
            candidate_source = "arrow_fov"
            candidate_positions = np.asarray(arrow_positions, dtype=np.float64)
            candidate_weights = np.asarray(arrow_weights, dtype=np.float64)
            candidate_dirs_np = np.asarray(arrow_dirs, dtype=np.float64)
        else:
            candidate_positions = cams
            candidate_weights = counts.astype(np.float64)

        pred_dir: Optional[np.ndarray] = None
        pred_cam, selection_idx, selection_wts = select_prediction_point(
            candidate_positions, candidate_weights,
            strategy=prediction_strategy, rng=rng,
            bandwidth=cluster_bandwidth, max_points=max_cluster_points)

        if candidate_dirs_np is not None and selection_idx:
            dir_vectors = candidate_dirs_np[selection_idx]
            weight_vec = selection_wts
            if weight_vec.shape[0] != len(selection_idx):
                weight_vec = np.ones(len(selection_idx), dtype=np.float64)
            weight_vec = np.clip(weight_vec, 0.0, None)
            if not np.any(weight_vec > 0):
                weight_vec = np.ones_like(weight_vec)
            weight_vec /= weight_vec.sum()
            mean_dir = np.sum(weight_vec[:, None] * dir_vectors, axis=0)
            norm_dir = float(np.linalg.norm(mean_dir))
            if norm_dir > 1e-6:
                pred_dir = mean_dir / norm_dir

        frame_id = str(frame.get("image_index", selection.path.name))
        return {
            "scene_id": scene_id,
            "frame_id": frame_id,
            "frame_path": str(selection.path),
            "description": frame.get("description"),
            "timestamp": frame.get("timestamp"),
            "visible_objects_count": len(frame.get("visible_objects", {}) or {}),
            "matched_object_ids": [int(o) for o in obj_ids],
            "matched_object_count": len(obj_ids),
            "grid": {
                "step": grid_step,
                "eye_height": eye_height,
                "points": len(cams),
            },
            "grid_max_visible_count": int(counts.max()),
            "grid_max_visible_point_count": int(np.sum(counts == int(counts.max()))),
            "grid_point_candidates": grid_point_candidates,
            "fov_pose_candidates": fov_pose_candidates,
            "gt_pose": {
                "position": gt_cam.tolist(),
                "direction": gt_dir.tolist() if gt_dir is not None else None,
                "scene_pose": pose_mat.tolist(),
            },
            "predicted_pose": {
                "position": pred_cam.tolist(),
                "direction": pred_dir.tolist() if pred_dir is not None else None,
                "source": f"{candidate_source}:{prediction_strategy}",
            },
        }

    # --- Standard / coarse_to_fine: camera prediction ---
    candidate_dirs_np = None
    candidate_source = "grid_probability"
    top_pose_count = _cfg_get(cfg, "top_pose_count", 5)

    if arrow_weights:
        candidate_source = arrow_source
        candidate_positions = np.asarray(arrow_positions, dtype=np.float64)
        candidate_weights = np.asarray(arrow_weights, dtype=np.float64)
        candidate_dirs_np = np.asarray(arrow_dirs, dtype=np.float64)
        top_fov = top_n_fov_poses(candidate_positions, candidate_weights,
                                  n=top_pose_count, rng=rng,
                                  directions=candidate_dirs_np)
        print(f"    top-{top_pose_count} FOV-weighted poses: {top_fov}")
    else:
        candidate_positions = cams
        candidate_weights = probs

    pred_dir = None
    try:
        pred_cam, selection_idx, selection_wts = select_prediction_point(
            candidate_positions, candidate_weights,
            strategy=prediction_strategy, rng=rng,
            bandwidth=cluster_bandwidth, max_points=max_cluster_points)
    except ValueError:
        pred_cam = pred_cam_prob
        selection_idx = [int(pred_idx)]
        selection_wts = np.asarray([1.0], dtype=np.float64)

    if candidate_dirs_np is not None and selection_idx:
        dir_vectors = candidate_dirs_np[selection_idx]
        weight_vec = selection_wts
        if weight_vec.shape[0] != len(selection_idx):
            weight_vec = np.ones(len(selection_idx), dtype=np.float64)
        weight_vec = np.clip(weight_vec, 0.0, None)
        if not np.any(weight_vec > 0):
            weight_vec = np.ones_like(weight_vec)
        weight_vec /= weight_vec.sum()
        mean_dir = np.sum(weight_vec[:, None] * dir_vectors, axis=0)
        norm_dir = float(np.linalg.norm(mean_dir))
        if norm_dir > 1e-6:
            pred_dir = mean_dir / norm_dir

    pred_source = f"{candidate_source}:{prediction_strategy}"
    metrics.distance_error = float(np.linalg.norm(pred_cam - gt_cam))

    # Standard mode: angular error + IoU
    if mode == EvalMode.STANDARD:
        if gt_dir is not None and pred_dir is not None:
            dot = float(np.clip(np.dot(gt_dir, pred_dir), -1.0, 1.0))
            metrics.angular_error_deg = float(math.degrees(math.acos(dot)))

        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        tri_pts = verts[tris_arr]
        tri_vecs = tri_pts[:, 1] - tri_pts[:, 0]
        tri_vecs_b = tri_pts[:, 2] - tri_pts[:, 0]
        tri_cross = np.cross(tri_vecs, tri_vecs_b)
        tri_areas = 0.5 * np.linalg.norm(tri_cross, axis=1)
        tri_centroids = tri_pts.mean(axis=1)

        iou_val, iou_err, gt_vis_set, pred_vis_set = compute_view_iou_error(
            gt_cam, gt_dir, pred_cam, pred_dir,
            hfov=hfov_rad, vfov=vfov_rad,
            rc=rc, geom_id=int(mesh_id),
            tri_pts=tri_pts, tri_centroids=tri_centroids,
            tri_areas=tri_areas, near=0.05, far=None)
        metrics.iou_error = iou_err
        if iou_val is not None:
            print(f"    view IoU: {iou_val:.3f} | IoU error: {iou_err:.3f}")
        else:
            print("    view IoU: n/a")

    print(f"    predicted camera ({pred_source}): {pred_cam.tolist()}")
    if pred_dir is not None:
        print(f"    approx. viewing direction: {pred_dir.tolist()}")

    # --- Optional visualisation ---
    show_heatmap = _cfg_get(cfg, "show_heatmap", False)
    show_arrows = _cfg_get(cfg, "show_arrows", False)
    show_3d = _cfg_get(cfg, "show_3d", False)

    if show_heatmap:
        plt.figure(figsize=(6.5, 6.2))
        sc = plt.scatter(cams[:, 0], cams[:, 1], c=probs, cmap="viridis", s=14)
        plt.colorbar(sc, label="Probability")
        if mode == EvalMode.COARSE_TO_FINE and refined_points.size:
            max_aw = float(np.max(arrow_all_weights)) if arrow_all_weights.size else 1.0
            if max_aw > 0:
                ref_colors = colormap(np.clip(refined_weights / max_aw, 0.0, 1.0))
            else:
                ref_colors = None
            plt.scatter(refined_points[:, 0], refined_points[:, 1],
                        c=ref_colors if ref_colors is not None else "none",
                        edgecolors="black", linewidths=0.6, s=32,
                        label=f"Refined grid ({arrow_step_used:.2f} m)")
        plt.axis("equal"); plt.xlabel("X (m)"); plt.ylabel("Y (m)")
        plt.title(f"{scene_id} · {metrics.frame_id} · grid {grid_step:.2f} m")
        add_heatmap_markers(gt_cam, pred_cam, label_pred=f"Pred ({pred_source})")
        plt.tight_layout(); plt.show()

    if show_arrows and arrow_weights:
        if mode == EvalMode.COARSE_TO_FINE and arrow_all_points.size:
            mask = ((arrow_all_weights > 0) &
                    (np.linalg.norm(arrow_all_dirs, axis=1) > 1e-8))
            pts_plot = arrow_all_points[mask]
            w_plot = arrow_all_weights[mask]
            d_plot = arrow_all_dirs[mask]
        else:
            pts_plot = np.asarray(arrow_positions)
            w_plot = np.asarray(arrow_weights, dtype=np.float32)
            d_plot = np.asarray(arrow_dirs)

        if pts_plot.size:
            arrow_len = _cfg_get(cfg, "arrow_len", 0.0)
            max_len = (0.9 * arrow_step_used) if arrow_len <= 0 else arrow_len
            W_np = w_plot.astype(np.float32)
            scale = np.where(W_np > 0, W_np / W_np.max(), 0.0)
            dirs_xy = np.asarray([d[:2] for d in d_plot], dtype=np.float32)
            norms = np.linalg.norm(dirs_xy, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            dirs_xy /= norms
            U_np = dirs_xy[:, 0] * max_len * scale
            V_np = dirs_xy[:, 1] * max_len * scale
            Qx = [float(p[0]) for p in pts_plot]
            Qy = [float(p[1]) for p in pts_plot]

            plt.figure(figsize=(7, 6.5))
            plt.quiver(Qx, Qy, U_np, V_np, W_np,
                       angles="xy", scale_units="xy", scale=1.0,
                       cmap="viridis", width=0.004, minlength=0.01)
            plt.colorbar(label="Max visible objects within FOV")
            plt.axis("equal"); plt.xlabel("X (m)"); plt.ylabel("Y (m)")
            plt.title(f"{scene_id} · FOV arrows "
                      f"(H={h_fov_deg:.0f}°, V={v_fov_deg:.0f}°)")
            add_arrow_markers(gt_cam, pred_cam)
            plt.tight_layout(); plt.show()

    if show_3d:
        _show_3d_scene(scene_id, scene_dir, mesh, obj2faces, obj_ids,
                       cams, probs, gt_cam, gt_dir, pred_cam, pred_dir,
                       pred_source, hfov_rad, vfov_rad, grid_step,
                       mode, metrics,
                       refined_points if mode == EvalMode.COARSE_TO_FINE else None,
                       refined_weights if mode == EvalMode.COARSE_TO_FINE else None,
                       arrow_all_weights if mode == EvalMode.COARSE_TO_FINE else None,
                       gt_vis_set if mode == EvalMode.STANDARD else None,
                       pred_vis_set if mode == EvalMode.STANDARD else None)

    return metrics


# ---------------------------------------------------------------------------
#  3D visualisation helper
# ---------------------------------------------------------------------------

def _show_3d_scene(scene_id, scene_dir, mesh, obj2faces, obj_ids,
                   cams, probs, gt_cam, gt_dir, pred_cam, pred_dir,
                   pred_source, hfov_rad, vfov_rad, grid_step,
                   mode, metrics,
                   refined_points, refined_weights, arrow_all_weights,
                   gt_vis_set, pred_vis_set):
    """Render the 3D Open3D scene with probability spheres and camera frustums."""
    from langloc.localization.visualization import GUI_INITIALISED
    import langloc.localization.visualization as viz_mod

    matched_set = {int(o) for o in obj_ids}
    frustum_scale = max(grid_step * 3.0, 0.6)

    try:
        from langloc.utils.mesh_segmentation import build_segmented_mesh
        mesh_vis, obj_stats = build_segmented_mesh(scene_dir, seed=42)
        colours = np.asarray(mesh_vis.vertex_colors)
        highlight = np.array([1.0, 0.3, 0.3], dtype=np.float64)
        for stats in obj_stats:
            oid = int(stats["object_id"])
            if oid in matched_set:
                idx = stats.get("vertex_indices")
                if idx is not None:
                    colours[idx] = np.clip(0.55 * colours[idx] + 0.45 * highlight, 0.0, 1.0)
        mesh_vis.vertex_colors = o3d.utility.Vector3dVector(colours)
        if not mesh_vis.has_vertex_normals():
            mesh_vis.compute_vertex_normals()
    except Exception:
        mesh_vis = colour_objects(mesh, obj2faces, obj_ids)
        obj_stats = []
    if not mesh_vis.has_vertex_normals():
        mesh_vis.compute_vertex_normals()

    from open3d.visualization import gui, rendering

    if not viz_mod.GUI_INITIALISED:
        gui.Application.instance.initialize()
        viz_mod.GUI_INITIALISED = True

    vis = o3d.visualization.O3DVisualizer(f"{scene_id} – localisation eval", 1280, 800)
    vis.show_settings = False

    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    vis.add_geometry("mesh", mesh_vis, material)

    text_added = set()
    if obj_stats:
        bbox_material = rendering.MaterialRecord()
        bbox_material.shader = "unlitLine"
        bbox_material.line_width = 1.5
        for stats in obj_stats:
            oid = int(stats["object_id"])
            label = stats.get("label") or f"id_{oid}"
            centroid = np.asarray(stats["centroid"]) if "centroid" in stats else None
            if centroid is not None and tuple(centroid) not in text_added:
                vis.add_3d_label(centroid, f"{oid}: {label}")
                text_added.add(tuple(centroid))
            if oid in matched_set and "bbox" in stats:
                vis.add_geometry(f"bbox_{oid}", stats["bbox"], bbox_material)

    prob_material = rendering.MaterialRecord()
    prob_material.shader = "defaultLit"
    prob_material.base_color = [1.0, 1.0, 1.0, 1.0]
    for idx_point, (point, colour) in enumerate(zip(cams, colormap(probs))):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
        s.translate(point)
        s.paint_uniform_color(colour)
        if not s.has_vertex_normals():
            s.compute_vertex_normals()
        vis.add_geometry(f"prob_{idx_point}", s, prob_material)

    # Coarse-to-fine: add refined grid points
    if (mode == EvalMode.COARSE_TO_FINE and refined_points is not None
            and refined_points.size):
        ref_material = rendering.MaterialRecord()
        ref_material.shader = "defaultLit"
        max_aw = float(np.max(arrow_all_weights)) if arrow_all_weights is not None and arrow_all_weights.size else 1.0
        if max_aw > 0 and refined_weights is not None and refined_weights.size:
            ref_colours = colormap(np.clip(refined_weights / max_aw, 0.0, 1.0))
        else:
            ref_colours = np.tile(np.array([[0.3, 0.85, 1.0]]), (len(refined_points), 1))
        for idx_point, point in enumerate(refined_points):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            s.translate(point)
            colour = ref_colours[idx_point] if idx_point < len(ref_colours) else np.array([0.3, 0.85, 1.0])
            s.paint_uniform_color(colour[:3])
            if not s.has_vertex_normals():
                s.compute_vertex_normals()
            vis.add_geometry(f"ref_grid_{idx_point}", s, ref_material)

    gt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    gt_sphere.translate(gt_cam)
    gt_sphere.paint_uniform_color([1.0, 0.0, 0.0])
    if not gt_sphere.has_vertex_normals():
        gt_sphere.compute_vertex_normals()
    vis.add_geometry("gt_cam", gt_sphere, material)
    vis.add_3d_label(gt_cam, "GT")

    pred_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.085)
    pred_sphere.translate(pred_cam)
    pred_sphere.paint_uniform_color([1.0, 0.9, 0.0])
    if not pred_sphere.has_vertex_normals():
        pred_sphere.compute_vertex_normals()
    vis.add_geometry("pred_cam", pred_sphere, material)
    vis.add_3d_label(pred_cam, f"Pred ({pred_source})")

    frustum_gt = create_camera_frustum(gt_cam, gt_dir, colour=(1.0, 0.0, 0.0),
                                       h_fov=hfov_rad, v_fov=vfov_rad, scale=frustum_scale)
    frustum_pred = create_camera_frustum(pred_cam, pred_dir, colour=(1.0, 0.9, 0.0),
                                         h_fov=hfov_rad, v_fov=vfov_rad, scale=frustum_scale)
    if frustum_gt is not None:
        fm = rendering.MaterialRecord()
        fm.shader = "unlitLine"; fm.line_width = 2.0; fm.base_color = [1.0, 0.0, 0.0, 1.0]
        vis.add_geometry("frustum_gt", frustum_gt, fm)
    if frustum_pred is not None:
        fm = rendering.MaterialRecord()
        fm.shader = "unlitLine"; fm.line_width = 2.0; fm.base_color = [1.0, 0.9, 0.0, 1.0]
        vis.add_geometry("frustum_pred", frustum_pred, fm)

    vis.reset_camera_to_default()
    gui.Application.instance.add_window(vis)

    # Standard mode: IoU overlap window
    if mode == EvalMode.STANDARD and gt_vis_set is not None and pred_vis_set is not None:
        vis_iou = o3d.visualization.O3DVisualizer(f"{scene_id} – IoU overlap", 1280, 800)
        vis_iou.show_settings = False
        base_mat = rendering.MaterialRecord()
        base_mat.shader = "defaultLitTransparency"
        base_mat.base_color = [0.8, 0.8, 0.8, 0.18]
        vis_iou.add_geometry("mesh_base", mesh, base_mat)

        for name, tri_set, colour, alpha in [
            ("iou_gt_only", gt_vis_set - pred_vis_set, (1.0, 0.0, 0.0), 0.65),
            ("iou_pred_only", pred_vis_set - gt_vis_set, (1.0, 0.85, 0.0), 0.65),
            ("iou_both", gt_vis_set & pred_vis_set, (1.0, 0.4, 0.0), 0.85),
        ]:
            if not tri_set:
                continue
            idx_arr = np.asarray(sorted(tri_set), dtype=np.int64)
            all_verts = np.asarray(mesh.vertices)
            sub_tris = np.asarray(mesh.triangles)[idx_arr]
            uniq, inv = np.unique(sub_tris.reshape(-1), return_inverse=True)
            sub = o3d.geometry.TriangleMesh()
            sub.vertices = o3d.utility.Vector3dVector(all_verts[uniq])
            sub.triangles = o3d.utility.Vector3iVector(inv.reshape(-1, 3))
            sub.paint_uniform_color(colour)
            if not sub.has_vertex_normals():
                sub.compute_vertex_normals()
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLitTransparency"
            mat.base_color = [*colour, alpha]
            vis_iou.add_geometry(name, sub, mat)

        vis_iou.add_geometry("gt_cam_iou", gt_sphere, material)
        vis_iou.add_geometry("pred_cam_iou", pred_sphere, material)
        if frustum_gt is not None:
            vis_iou.add_geometry("frustum_gt_iou", frustum_gt,
                                 rendering.MaterialRecord())
        if frustum_pred is not None:
            vis_iou.add_geometry("frustum_pred_iou", frustum_pred,
                                 rendering.MaterialRecord())
        vis_iou.reset_camera_to_default()
        gui.Application.instance.add_window(vis_iou)

    gui.Application.instance.run()


# ---------------------------------------------------------------------------
#  Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(cfg, *, graph_cfg=None) -> None:
    """Run the full evaluation pipeline over multiple scenes.

    Loads scene graphs, filters candidate scene IDs, loops over scenes
    calling :func:`evaluate_scene`, aggregates metrics, and writes
    output files (log, JSON metrics, or candidate poses depending on
    the evaluation mode).

    Args:
        cfg: Configuration namespace or dict-like object.  Must contain
            at minimum ``root``, ``graphs``, and ``mode``.
        graph_cfg: Optional graph configuration (from ``cfg.graph``) with
            ``max_dist``, ``embedding_type``, and ``use_attributes``.
    """
    mode = EvalMode(_cfg_get(cfg, "mode", "standard"))
    rng = np.random.default_rng(seed=_cfg_get(cfg, "seed", 0))
    graphs_dir = Path(_cfg_get(cfg, "graphs"))

    graph_kw = {}
    if graph_cfg is not None:
        graph_kw = dict(
            max_dist=graph_cfg.max_dist,
            embedding_type=graph_cfg.embedding_type,
            use_attributes=graph_cfg.use_attributes,
        )
    scenes = load_scene_graphs(graphs_dir, **graph_kw)
    root = Path(_cfg_get(cfg, "root"))

    candidate_ids = list(scenes.keys())
    visualize_scene_id = _cfg_get(cfg, "visualize_scene")
    scene_ids = _cfg_get(cfg, "scene_ids")

    if visualize_scene_id:
        if scene_ids:
            print("[WARN] --visualize_scene overrides --scene_ids.")
        if visualize_scene_id not in scenes:
            print(f"[ERROR] Scene '{visualize_scene_id}' not in processed graphs.")
            return
        candidate_ids = [visualize_scene_id]
    elif scene_ids:
        scene_set = set(scene_ids) if not isinstance(scene_ids, set) else scene_ids
        candidate_ids = [sid for sid in candidate_ids if sid in scene_set]
    else:
        query_root = ensure_query_root(_cfg_get(cfg, "query_root"), root)
        candidate_ids = [
            sid for sid in candidate_ids
            if (query_root / sid / "output" / "descriptions").exists()
            or (root / sid / "output" / "descriptions").exists()
        ]

    candidate_ids.sort()
    max_scenes = _cfg_get(cfg, "max_scenes")
    if max_scenes is not None:
        candidate_ids = candidate_ids[:int(max_scenes)]

    print(f"Evaluating {len(candidate_ids)} scene(s) in {mode.value} mode...\n")

    if mode == EvalMode.CANDIDATES:
        _run_candidates_mode(candidate_ids, scenes, mode, cfg, rng, graph_cfg=graph_cfg)
    else:
        _run_metrics_mode(candidate_ids, scenes, mode, cfg, rng, graph_cfg=graph_cfg)


def _run_candidates_mode(candidate_ids, scenes, mode, cfg, rng, *, graph_cfg=None):
    """Execute candidates-mode evaluation and write JSON output."""
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):
            return it

    results: List[Dict] = []
    skipped: List[Dict[str, str]] = []
    for sid in tqdm(candidate_ids, desc="Scenes", unit="scene"):
        record = evaluate_scene(sid, scenes[sid], mode, cfg, rng, graph_cfg=graph_cfg)
        if record is None:
            skipped.append({"scene_id": sid, "reason": "skipped"})
        else:
            results.append(record)

    output_json = _cfg_get(cfg, "output_json")
    if output_json is None:
        output_json = Path("eval") / "eval_pose_candidates.json"
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "scenes": results,
        "skipped": skipped,
        "args": {k: str(v) if isinstance(v, Path) else v
                 for k, v in (vars(cfg).items() if hasattr(cfg, '__dict__')
                              else dict(cfg).items())},
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote candidate poses to {output_path}")


def _run_metrics_mode(candidate_ids, scenes, mode, cfg, rng, *, graph_cfg=None):
    """Execute standard or coarse_to_fine evaluation with metrics aggregation."""
    params_text = format_args_section(cfg)

    metrics_list: List[SceneMetrics] = []
    for idx, sid in enumerate(candidate_ids, start=1):
        print(f"[{idx:03d}/{len(candidate_ids):03d}] {sid}")
        result = evaluate_scene(sid, scenes[sid], mode, cfg, rng, graph_cfg=graph_cfg)
        if result is None or not isinstance(result, SceneMetrics):
            continue
        metrics_list.append(result)

        print(f"    frame: {result.frame_id}")
        print(f"    matches: {result.matched_objects} | grid pts: {result.grid_points}")

        if mode == EvalMode.STANDARD:
            if result.hit_masses:
                hit_line = " | ".join(
                    f"hit@{r:.2f}m: {result.hit_masses.get(r, 0.0):.3f}"
                    for r in sorted(result.hit_masses))
                print(f"    {hit_line}")
            if result.mass_radii:
                mass_line = " | ".join(
                    f"R{p:.0f}%: {result.mass_radii.get(p, float('nan')):.3f} m"
                    for p in sorted(result.mass_radii))
                print(f"    mass-radius: {mass_line}")
            ang = "n/a" if result.angular_error_deg is None else f"{result.angular_error_deg:.2f}°"
            print(f"    dist_err: {result.distance_error:.3f} m | ang_err: {ang}")
        else:
            print(f"    gt_prob: {result.gt_prob:.4f} | nll: {result.nll:.3f}")
            hit_radius = _cfg_get(cfg, "hit_radius", 0.5)
            print(f"    hit@{hit_radius:.2f}m: {result.hit_mass:.3f} | "
                  f"dist_err: {result.distance_error:.3f} m")
        print()

    if not metrics_list:
        print("No scenes produced metrics.")
        return

    # Build table
    if mode == EvalMode.STANDARD:
        hit_radii = _cfg_get(cfg, "hit_radii", [0.75, 1.0, 1.5, 2.0, 2.5])
        mass_percentiles = _cfg_get(cfg, "mass_percentiles", [50.0, 90.0])
        topk_k = _cfg_get(cfg, "top_k_min_dist", 10)
        table_text = build_metrics_table_standard(metrics_list, hit_radii,
                                                  mass_percentiles, topk_k)
    else:
        hit_radius = _cfg_get(cfg, "hit_radius", 0.5)
        table_text = build_metrics_table_simple(metrics_list, hit_radius)

    if table_text:
        print("Scene-level summary table -------------------------------")
        print(table_text)
        print("---------------------------------------------------------\n")

    # Aggregate
    def agg(values):
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(np.median(arr))

    mean_err, med_err = agg([m.distance_error for m in metrics_list])
    agg_lines = ["Aggregate metrics ---------------------------------------"]

    if mode == EvalMode.STANDARD:
        hit_radii = _cfg_get(cfg, "hit_radii", [0.75, 1.0, 1.5, 2.0, 2.5])
        for r in sorted(set(float(x) for x in hit_radii)):
            vals = [m.hit_masses.get(r, 0.0) for m in metrics_list if m.hit_masses]
            if vals:
                m_h, med_h = agg(vals)
                agg_lines.append(f"  Hit@{r:.2f}m              : mean={m_h:.3f} | median={med_h:.3f}")
        mass_percentiles = _cfg_get(cfg, "mass_percentiles", [50.0, 90.0])
        for p in sorted(set(float(x) for x in mass_percentiles)):
            vals = [m.mass_radii.get(p, float("nan")) for m in metrics_list if m.mass_radii]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                m_r, med_r = agg(vals)
                agg_lines.append(f"  Mass-radius R{p:.0f}% (m): mean={m_r:.3f} | median={med_r:.3f}")
        topk_vals = [m.topk_min_dist for m in metrics_list if m.topk_min_dist is not None]
        if topk_vals:
            m_t, med_t = agg(topk_vals)
            agg_lines.append(f"  TopK min dist (m)    : mean={m_t:.3f} | median={med_t:.3f}")
        ang_vals = [m.angular_error_deg for m in metrics_list if m.angular_error_deg is not None]
        if ang_vals:
            m_a, med_a = agg(ang_vals)
            agg_lines.append(f"  Angular error (deg)  : mean={m_a:.2f} | median={med_a:.2f}")
        iou_vals = [m.iou_error for m in metrics_list if m.iou_error is not None]
        if iou_vals:
            m_i, med_i = agg(iou_vals)
            agg_lines.append(f"  View IoU error       : mean={m_i:.3f} | median={med_i:.3f}")
    else:
        m_gt, med_gt = agg([m.gt_prob for m in metrics_list if m.gt_prob is not None])
        m_nll, med_nll = agg([m.nll for m in metrics_list if m.nll is not None])
        m_hit, med_hit = agg([m.hit_mass for m in metrics_list if m.hit_mass is not None])
        agg_lines.append(f"  GT probability       : mean={m_gt:.4f} | median={med_gt:.4f}")
        agg_lines.append(f"  NLL (surprisal)      : mean={m_nll:.3f} | median={med_nll:.3f}")
        hit_radius = _cfg_get(cfg, "hit_radius", 0.5)
        agg_lines.append(f"  Hit@{hit_radius:.2f}m          : mean={m_hit:.3f} | median={med_hit:.3f}")

    agg_lines.append(f"  Distance error (m)   : mean={mean_err:.3f} | median={med_err:.3f}")
    agg_lines.append("---------------------------------------------------------\n")
    print("\n".join(agg_lines))

    # Write log
    log_file = _cfg_get(cfg, "log_file")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_sections = [params_text]
        if table_text:
            log_sections.extend(["Scene-level summary table", table_text])
        log_sections.append("\n".join(agg_lines))
        log_path.write_text("\n\n".join(log_sections).rstrip() + "\n")
        print(f"Metrics summary logged to {log_path}")

    # Save JSON metrics
    save_metrics = _cfg_get(cfg, "save_metrics")
    if save_metrics:
        save_path = Path(save_metrics)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if mode == EvalMode.STANDARD:
            payload = [{
                "scene_id": m.scene_id, "frame_id": m.frame_id,
                "hit_masses": {str(k): v for k, v in m.hit_masses.items()} if m.hit_masses else {},
                "mass_radii": {str(k): v for k, v in m.mass_radii.items()} if m.mass_radii else {},
                "topk_min_dist": m.topk_min_dist, "distance_error": m.distance_error,
                "angular_error_deg": m.angular_error_deg, "iou_error": m.iou_error,
                "grid_points": m.grid_points, "matched_objects": m.matched_objects,
            } for m in metrics_list]
        else:
            payload = [{
                "scene_id": m.scene_id, "frame_id": m.frame_id,
                "gt_prob": m.gt_prob, "nll": m.nll, "hit_mass": m.hit_mass,
                "distance_error": m.distance_error,
                "grid_points": m.grid_points, "matched_objects": m.matched_objects,
            } for m in metrics_list]
        save_path.write_text(json.dumps({"metrics": payload,
                                         "aggregate": "\n".join(agg_lines)}, indent=2))
        print(f"Metrics saved to {save_path}")
