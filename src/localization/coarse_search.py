"""Coarse-to-fine multi-level arrow search for localization.

Implements an iterative grid-refinement strategy: starting from a coarse
grid covering the entire scene, FOV-weighted arrow scores are computed
at each level; peaks are identified (optionally with NMS), and a finer
local grid is spawned around each peak for the next iteration.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

from src.localization.grid import (
    first_hit_is_object,
    grid_from_bounds,
)
from src.localization.visualization import (
    dir_to_yaw_pitch,
    best_fov_window,
    average_direction,
)


# ---------------------------------------------------------------------------
#  Arrow field computation
# ---------------------------------------------------------------------------

def arrow_field_from_visibility(cams: np.ndarray,
                                visible_dirs: List[List[np.ndarray]],
                                Nx: int,
                                Ny: int,
                                hfov: float,
                                vfov: float,
                                stride: int = 1) -> Tuple[List[np.ndarray],
                                                           List[np.ndarray],
                                                           List[float]]:
    """Compute a FOV-weighted arrow field from per-camera visibility directions.

    Iterates over the grid in a structured pattern (respecting *stride*),
    finds the best FOV window for each camera, and records the average
    viewing direction weighted by visible object count.

    Args:
        cams: Camera positions, shape ``(N, 3)``.
        visible_dirs: Per-camera list of unit direction vectors.
        Nx: Number of grid columns.
        Ny: Number of grid rows.
        hfov: Horizontal field-of-view in radians.
        vfov: Vertical field-of-view in radians.
        stride: Evaluate every *stride*-th grid cell in each dimension.

    Returns:
        A 3-tuple ``(positions, dirs, weights)`` of matching-length
        lists containing arrow positions, unit directions, and
        FOV-weighted counts.
    """
    arrow_positions: List[np.ndarray] = []
    arrow_dirs: List[np.ndarray] = []
    arrow_weights: List[float] = []

    stride = max(1, int(stride))
    for gy_i in range(0, Ny, stride):
        for gx_i in range(0, Nx, stride):
            idx = gy_i * Nx + gx_i
            if idx >= len(cams):
                continue
            dirs = np.asarray(visible_dirs[idx], dtype=np.float32)
            if dirs.size == 0:
                continue
            yaws = np.empty(len(dirs), dtype=np.float32)
            pits = np.empty(len(dirs), dtype=np.float32)
            for i, vec in enumerate(dirs):
                yaw, pit = dir_to_yaw_pitch(vec)
                yaws[i] = yaw
                pits[i] = pit
            sel, count = best_fov_window(yaws, pits, hfov, vfov)
            if count == 0:
                continue
            mdir = average_direction(dirs, sel)
            if mdir is None:
                continue
            arrow_positions.append(cams[idx])
            arrow_dirs.append(mdir)
            arrow_weights.append(float(count))
    return arrow_positions, arrow_dirs, arrow_weights


def arrow_weights_generic(cams: np.ndarray,
                          visible_dirs: List[List[np.ndarray]],
                          hfov: float,
                          vfov: float) -> Tuple[np.ndarray, List[Optional[np.ndarray]]]:
    """Compute per-camera FOV-weighted arrow scores for an arbitrary camera set.

    Unlike :func:`arrow_field_from_visibility`, this function does not
    assume a structured grid and processes every camera.

    Args:
        cams: Camera positions, shape ``(N, 3)``.
        visible_dirs: Per-camera list of unit direction vectors.
        hfov: Horizontal field-of-view in radians.
        vfov: Vertical field-of-view in radians.

    Returns:
        A 2-tuple ``(weights, dirs)`` where *weights* is a float64 array
        of shape ``(N,)`` and *dirs* is a list of unit direction arrays
        (or ``None`` for cameras with no valid FOV window).
    """
    weights = np.zeros(len(cams), dtype=np.float64)
    dirs: List[Optional[np.ndarray]] = [None] * len(cams)
    for idx, dirs_list in enumerate(visible_dirs):
        if not dirs_list:
            continue
        dirs_arr = np.asarray(dirs_list, dtype=np.float32)
        yaws = np.empty(len(dirs_arr), dtype=np.float32)
        pits = np.empty(len(dirs_arr), dtype=np.float32)
        for i, vec in enumerate(dirs_arr):
            yaw, pit = dir_to_yaw_pitch(vec)
            yaws[i] = yaw
            pits[i] = pit
        sel, count = best_fov_window(yaws, pits, hfov, vfov)
        if count == 0:
            continue
        mdir = average_direction(dirs_arr, sel)
        if mdir is None:
            continue
        weights[idx] = float(count)
        dirs[idx] = mdir
    return weights, dirs


# ---------------------------------------------------------------------------
#  Multi-level coarse-to-fine search
# ---------------------------------------------------------------------------

def coarse_to_fine_arrow_search(verts: np.ndarray,
                                centroids: Dict[int, np.ndarray],
                                rc: o3d.t.geometry.RaycastingScene,
                                tri2obj: np.ndarray,
                                z_eye: float,
                                hfov: float,
                                vfov: float,
                                base_step: float,
                                levels: int,
                                refine_factor: float,
                                keep_ratio: float,
                                top_k: int,
                                apply_nms: bool = True) -> Tuple[List[np.ndarray],
                                                     List[np.ndarray],
                                                     List[float],
                                                     float,
                                                     np.ndarray,
                                                     np.ndarray,
                                                     np.ndarray,
                                                     np.ndarray]:
    """Iteratively refine around the highest FOV-weighted arrows.

    Level 0 covers the full mesh with spacing ``base_step``.  At each
    subsequent level the top peaks are identified (optionally with
    non-maximum suppression), a local grid with smaller spacing is
    spawned around each peak, and arrow scores are recomputed.

    Args:
        verts: Mesh vertex positions, shape ``(V, 3)``.
        centroids: Object-ID-to-centroid mapping.
        rc: Pre-built Open3D raycasting scene.
        tri2obj: Per-triangle object-ID array.
        z_eye: Eye-height offset above mesh floor.
        hfov: Horizontal field-of-view in radians.
        vfov: Vertical field-of-view in radians.
        base_step: Initial (coarsest) grid spacing in metres.
        levels: Number of refinement levels (>= 1).
        refine_factor: Factor to shrink spacing each level (e.g. 2 halves it).
        keep_ratio: Keep arrows within ``ratio * max_weight`` for refinement.
        top_k: Maximum number of peaks to refine per level.
        apply_nms: Whether to apply non-maximum suppression when selecting
            refinement seeds.

    Returns:
        A 9-tuple containing:

        - ``final_positions``: Arrow positions at the finest level.
        - ``final_dirs``: Arrow directions at the finest level.
        - ``final_weights``: Arrow weights at the finest level.
        - ``current_step``: Grid spacing at the finest level.
        - ``refined_points``: All refined grid points (for visualisation).
        - ``refined_weights``: Weights of refined grid points.
        - ``all_points``: Every grid point sampled across all levels.
        - ``all_weights``: Weights for *all_points*.
        - ``all_dirs``: Directions for *all_points*.
    """
    from src.localization.grid import compute_visible_dirs

    xs, ys, zs = verts[:, 0], verts[:, 1], verts[:, 2]
    bounds = (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))
    z_base = float(zs.min())

    base_step = max(float(base_step), 1e-3)
    levels = max(1, int(levels))

    current_step = base_step
    current_points, _, _ = grid_from_bounds(bounds, current_step, z_base=z_base, z_eye=z_eye)
    visited: set[Tuple[float, float]] = set()
    for pt in current_points:
        visited.add((round(float(pt[0]), 4), round(float(pt[1]), 4)))

    refined_points: List[np.ndarray] = []
    refined_weights_list: List[float] = []
    final_positions: List[np.ndarray] = []
    final_dirs: List[np.ndarray] = []
    final_weights: List[float] = []

    all_points: List[np.ndarray] = []
    all_weights: List[float] = []
    all_dirs: List[Optional[np.ndarray]] = []

    for lvl in range(levels):
        if len(current_points) == 0:
            break

        vis_dirs = compute_visible_dirs(current_points, centroids, rc, tri2obj)
        weights_np, dirs_list = arrow_weights_generic(current_points,
                                                      vis_dirs,
                                                      hfov=hfov,
                                                      vfov=vfov)

        for i, pt in enumerate(current_points):
            all_points.append(pt)
            all_weights.append(float(weights_np[i]) if i < len(weights_np) else 0.0)
            all_dirs.append(dirs_list[i] if i < len(dirs_list) else None)

        valid_idx = [i for i, w in enumerate(weights_np) if w > 0 and dirs_list[i] is not None]
        final_positions = [current_points[i] for i in valid_idx]
        final_dirs = [dirs_list[i] for i in valid_idx if dirs_list[i] is not None]
        final_weights = [float(weights_np[i]) for i in valid_idx]

        if lvl == levels - 1:
            break

        if not len(weights_np):
            break
        max_w = float(weights_np.max()) if weights_np.size else 0.0
        if max_w <= 0.0:
            break

        order = np.argsort(-weights_np)
        peak_idx: List[int] = []
        if apply_nms:
            suppress_radius = current_step * 0.6
            for idx in order:
                if weights_np[idx] < keep_ratio * max_w:
                    break
                if top_k > 0 and len(peak_idx) >= top_k:
                    break
                keep = True
                for p in peak_idx:
                    if np.linalg.norm(current_points[idx][:2] - current_points[p][:2]) < suppress_radius:
                        keep = False
                        break
                if keep:
                    peak_idx.append(int(idx))
        else:
            for idx in order:
                if weights_np[idx] < keep_ratio * max_w:
                    break
                if top_k > 0 and len(peak_idx) >= top_k:
                    break
                peak_idx.append(int(idx))

        if not peak_idx:
            break

        next_step = current_step / refine_factor if refine_factor != 0 else current_step
        if next_step <= 0:
            break

        next_points: List[np.ndarray] = []
        for idx in peak_idx:
            centre = current_points[idx]
            cx, cy = float(centre[0]), float(centre[1])
            next_points.append(np.array([cx, cy, z_base + z_eye], dtype=np.float64))
            refined_points.append(np.array([cx, cy, z_base + z_eye], dtype=np.float64))
            refined_weights_list.append(float(weights_np[idx]))
            offsets = (-next_step, 0.0, next_step)
            for dx in offsets:
                for dy in offsets:
                    nx, ny = cx + dx, cy + dy
                    if nx < bounds[0] - 1e-6 or nx > bounds[1] + 1e-6:
                        continue
                    if ny < bounds[2] - 1e-6 or ny > bounds[3] + 1e-6:
                        continue
                    key = (round(nx, 4), round(ny, 4))
                    if key in visited:
                        continue
                    visited.add(key)
                    pt = np.array([nx, ny, z_base + z_eye], dtype=np.float64)
                    next_points.append(pt)
                    refined_points.append(pt)
                    refined_weights_list.append(float(weights_np[idx]))

        current_points = np.asarray(next_points, dtype=np.float64)
        current_step = next_step

    all_points_np = np.asarray(all_points, dtype=np.float64) if all_points else np.empty((0, 3), dtype=np.float64)
    all_weights_np = np.asarray(all_weights, dtype=np.float64) if all_weights else np.empty((0,), dtype=np.float64)
    all_dirs_np = np.zeros((len(all_dirs), 3), dtype=np.float64)
    for i, d in enumerate(all_dirs):
        if d is not None:
            all_dirs_np[i] = d
    refined_points_np = np.asarray(refined_points, dtype=np.float64) if refined_points else np.empty((0, 3), dtype=np.float64)
    refined_weights_np = np.asarray(refined_weights_list, dtype=np.float64) if refined_weights_list else np.empty((0,), dtype=np.float64)

    return (final_positions,
            final_dirs,
            final_weights,
            current_step,
            refined_points_np,
            refined_weights_np,
            all_points_np,
            all_weights_np,
            all_dirs_np)
