"""Visualization utilities for localization: colour mapping, FOV geometry, and rendering.

Provides colour-mapping helpers, angular/FOV utilities for computing
optimal viewing windows, and 2-D/3-D rendering functions for heatmaps,
arrow (quiver) plots, and Open3D scene visualisations.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
#  Module-level state
# ---------------------------------------------------------------------------

GUI_INITIALISED = False
"""Whether the Open3D GUI application has been initialised this session."""


# ---------------------------------------------------------------------------
#  Colour helpers
# ---------------------------------------------------------------------------

def colour_objects(mesh: o3d.geometry.TriangleMesh,
                   obj2faces: dict[int, np.ndarray],
                   focus: list[int],
                   base: tuple[float, float, float] = (0.55, 0.55, 0.55),
                   ) -> o3d.geometry.TriangleMesh:
    """Grey-out the mesh and assign random bright colours to selected objects.

    Args:
        mesh: Triangle mesh with vertex colours.
        obj2faces: Mapping from object ID to triangle-index arrays.
        focus: Object IDs to highlight.
        base: RGB tuple for unmatched vertices.

    Returns:
        The same *mesh* with updated vertex colours (modified in-place).
    """
    rng = np.random.default_rng(42)
    vcols = np.tile(base, (len(mesh.vertices), 1))
    tris = np.asarray(mesh.triangles)
    for oid in focus:
        col = rng.random(3)
        for fid in obj2faces.get(oid, []):
            for vid in tris[fid]:
                vcols[int(vid)] = col
    mesh.vertex_colors = o3d.utility.Vector3dVector(vcols)
    return mesh


def colormap(vals: np.ndarray) -> np.ndarray:
    """Map scalar values in ``[0, 1]`` to RGB using the *viridis* colourmap.

    Args:
        vals: 1-D array of values in ``[0, 1]``.

    Returns:
        An ``(N, 3)`` float array of RGB colours.
    """
    cmap = plt.get_cmap("viridis")
    return cmap(vals)[:, :3]


# ---------------------------------------------------------------------------
#  Angular / FOV utilities
# ---------------------------------------------------------------------------

def dir_to_yaw_pitch(v: np.ndarray) -> Tuple[float, float]:
    """Convert a 3-D direction vector to yaw and pitch angles.

    Yaw is measured around +Z (from +X toward +Y); pitch is elevation
    above the XY plane.

    Args:
        v: Direction vector, shape ``(3,)``.

    Returns:
        A ``(yaw, pitch)`` tuple in radians.
    """
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    yaw = math.atan2(y, x)
    pitch = math.atan2(z, math.hypot(x, y))
    return yaw, pitch


def best_fov_window(yaws: np.ndarray,
                    pitches: np.ndarray,
                    hfov: float, vfov: float) -> Tuple[np.ndarray, int]:
    """Find the axis-aligned yaw x pitch window that maximises visible object count.

    Performs a sweep over yaw (with wrap-around handling) and a nested
    sweep over pitch to find the rectangular angular window of size
    ``hfov x vfov`` containing the most objects.

    Args:
        yaws: Per-object yaw angles in radians, shape ``(K,)``.
        pitches: Per-object pitch angles in radians, shape ``(K,)``.
        hfov: Horizontal field-of-view width in radians.
        vfov: Vertical field-of-view width in radians.

    Returns:
        A 2-tuple ``(selected_indices, max_count)`` where
        *selected_indices* is an array of unique object indices inside
        the best window.
    """
    n = len(yaws)
    if n == 0:
        return np.array([], dtype=int), 0

    order = np.argsort(yaws)
    yaw_sorted = yaws[order]
    pit_sorted = pitches[order]
    idx_sorted = order

    yaw_ext = np.concatenate([yaw_sorted, yaw_sorted + 2 * math.pi])
    pit_ext = np.concatenate([pit_sorted, pit_sorted])
    idx_ext = np.concatenate([idx_sorted, idx_sorted])

    best_cnt = 0
    best_sel = np.array([], dtype=int)

    j = 0
    for s in range(n):
        y0 = yaw_ext[s]
        y1 = y0 + hfov
        j = max(j, s)
        while j < s + n and yaw_ext[j] <= y1 + 1e-9:
            j += 1

        if j <= s:
            continue

        cand_slice = slice(s, j)
        ps = pit_ext[cand_slice]
        ids = idx_ext[cand_slice]

        p_order = np.argsort(ps)
        ps = ps[p_order]
        ids = ids[p_order]

        t_end = 0
        for t_start in range(len(ps)):
            p0 = ps[t_start]
            p1 = p0 + vfov
            while t_end < len(ps) and ps[t_end] <= p1 + 1e-9:
                t_end += 1
            cnt = t_end - t_start
            if cnt > best_cnt:
                best_cnt = cnt
                best_sel = np.unique(ids[t_start:t_end])

    return best_sel, int(best_cnt)


def average_direction(unit_dirs: np.ndarray, sel: np.ndarray) -> Optional[np.ndarray]:
    """Compute the normalised vector average of selected unit directions.

    Args:
        unit_dirs: Array of unit direction vectors, shape ``(K, 3)``.
        sel: Indices into *unit_dirs* to average.

    Returns:
        A unit direction vector ``(3,)``, or ``None`` if the result is
        degenerate (empty selection or near-zero norm).
    """
    if sel.size == 0:
        return None
    m = unit_dirs[sel].mean(axis=0)
    n = np.linalg.norm(m)
    if n < 1e-8:
        return None
    return m / n


# ---------------------------------------------------------------------------
#  2-D plot markers
# ---------------------------------------------------------------------------

def add_heatmap_markers(gt_cam: np.ndarray,
                        pred_cam: np.ndarray,
                        label_gt: str = "GT",
                        label_pred: str = "Pred") -> None:
    """Add ground-truth and predicted camera markers to the current matplotlib figure.

    Args:
        gt_cam: Ground-truth camera position, shape ``(3,)`` (only X, Y used).
        pred_cam: Predicted camera position, shape ``(3,)`` (only X, Y used).
        label_gt: Legend label for the ground-truth marker.
        label_pred: Legend label for the predicted marker.
    """
    plt.scatter(gt_cam[0], gt_cam[1],
                c="red", marker="*", s=160,
                linewidths=1.2, edgecolors="black",
                label=label_gt)
    plt.scatter(pred_cam[0], pred_cam[1],
                c="orange", marker="o", s=80,
                linewidths=1.0, edgecolors="black",
                label=label_pred)
    plt.legend(loc="best")


def add_arrow_markers(gt_cam: np.ndarray,
                      pred_cam: np.ndarray) -> None:
    """Add ground-truth and predicted camera markers to an arrow (quiver) plot.

    Args:
        gt_cam: Ground-truth camera position, shape ``(3,)``.
        pred_cam: Predicted camera position, shape ``(3,)``.
    """
    plt.scatter([gt_cam[0]], [gt_cam[1]],
                c="red", marker="*", s=160,
                linewidths=1.0, edgecolors="black",
                label="GT")
    plt.scatter([pred_cam[0]], [pred_cam[1]],
                c="orange", marker="o", s=80,
                linewidths=1.0, edgecolors="black",
                label="Pred")
    plt.legend(loc="best")


# ---------------------------------------------------------------------------
#  3-D camera frustum
# ---------------------------------------------------------------------------

def create_camera_frustum(center: np.ndarray,
                          forward: Optional[np.ndarray],
                          colour: Tuple[float, float, float],
                          h_fov: float,
                          v_fov: float,
                          scale: float = 0.6) -> Optional[o3d.geometry.LineSet]:
    """Create a wireframe camera frustum for Open3D visualisation.

    Builds a simple line-based frustum with an apex at *center* and four
    base corners determined by the field-of-view angles.

    Args:
        center: Camera centre position, shape ``(3,)``.
        forward: Unit forward direction vector, shape ``(3,)``.  If
            ``None`` or degenerate, returns ``None``.
        colour: RGB colour tuple for the frustum lines.
        h_fov: Horizontal field-of-view in radians.
        v_fov: Vertical field-of-view in radians.
        scale: Frustum depth (distance from apex to base plane).

    Returns:
        An ``o3d.geometry.LineSet``, or ``None`` if the direction is
        unavailable or degenerate.
    """
    if forward is None:
        return None
    fwd = np.asarray(forward, dtype=np.float64)
    norm = np.linalg.norm(fwd)
    if norm < 1e-6:
        return None
    fwd /= norm

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(fwd, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = np.cross(fwd, up)
    r_norm = np.linalg.norm(right)
    if r_norm < 1e-6:
        return None
    right /= r_norm
    up = np.cross(right, fwd)

    depth = scale
    half_w = math.tan(h_fov / 2.0) * depth
    half_h = math.tan(v_fov / 2.0) * depth

    centre = np.asarray(center, dtype=np.float64)
    apex = centre
    base = centre + fwd * depth

    corners = [
        base + right * half_w + up * half_h,
        base - right * half_w + up * half_h,
        base - right * half_w - up * half_h,
        base + right * half_w - up * half_h,
    ]

    points = np.vstack([apex, *corners])
    lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ], dtype=np.int32)

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    colours = np.tile(np.asarray(colour, dtype=np.float64), (lines.shape[0], 1))
    frustum.colors = o3d.utility.Vector3dVector(colours)
    return frustum
