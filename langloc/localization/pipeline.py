"""Shared localization pipeline for visualization scripts.

Consolidates the mesh-loading, grid-sampling, ray-casting, probability
computation, and rendering logic shared by
``tools/viz/visualize_loc_prob.py`` and
``tools/viz/visualize_loc_from_query.py``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from langloc.localization.grid import load_scene, sample_grid, first_hit_is_object
from langloc.localization.visualization import (
    colour_objects,
    colormap,
    dir_to_yaw_pitch,
    best_fov_window,
    average_direction,
)


def run_loc_pipeline(
    scan_dir: Path,
    obj_ids: list[int],
    obj2faces: dict,
    mesh: o3d.geometry.TriangleMesh,
    tri2obj: np.ndarray,
    grid_step: float = 0.25,
    show_heatmap: bool = False,
    show_arrows: bool = False,
    show_3d: bool = False,
    h_fov_deg: float = 100.0,
    v_fov_deg: float = 60.0,
    arrow_stride: int = 2,
    arrow_len: float = 0.0,
    title_prefix: str = "",
) -> np.ndarray | None:
    """Run the full localization pipeline on a single scene.

    Args:
        scan_dir: Path to the 3RScan scene directory.
        obj_ids: Object IDs matched by the retrieval model.
        obj2faces: Mapping from object ID to triangle-face indices.
        mesh: Pre-loaded Open3D triangle mesh.
        tri2obj: Per-triangle object-ID array.
        grid_step: XY grid spacing in metres.
        show_heatmap: Whether to render a 2-D Matplotlib heatmap.
        show_arrows: Whether to render a FOV-weighted quiver plot.
        show_3d: Whether to open an Open3D 3-D viewer.
        h_fov_deg: Horizontal field-of-view in degrees.
        v_fov_deg: Vertical field-of-view in degrees.
        arrow_stride: Plot every Nth grid camera.
        arrow_len: Max arrow length in metres (0 = 0.9 * grid_step).
        title_prefix: Prefix for plot titles.

    Returns:
        Probability array over grid cameras, or ``None`` if no objects
        are visible.
    """
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    verts = np.asarray(mesh.vertices)
    cams, cam_linear_indices, Nx, Ny = sample_grid(
        verts, step=grid_step, mesh=mesh, return_indices=True
    )

    # Object centroids
    tris = np.asarray(mesh.triangles)
    centroids = {}
    for oid in obj_ids:
        faces = obj2faces.get(oid)
        if faces is not None and len(faces):
            centroids[oid] = verts[np.unique(tris[faces].ravel())].mean(0)

    if not centroids:
        print(f"    {title_prefix}no centroids for matched objects")
        return None

    # Visibility tally
    visible_dirs: list[list[np.ndarray]] = [[] for _ in range(len(cams))]
    for idx, cam in enumerate(cams):
        for oid, cen in centroids.items():
            if first_hit_is_object(cam, cen, oid, rc, tri2obj):
                d = cen - cam
                ln = np.linalg.norm(d)
                if ln > 1e-6:
                    visible_dirs[idx].append(d / ln)

    counts = np.array([len(v) for v in visible_dirs], dtype=np.int32)
    if counts.sum() == 0:
        print(f"    {title_prefix}matched objects not visible from any camera")
        return None
    probs = counts / counts.sum()

    # 2-D heatmap
    if show_heatmap:
        plt.figure(figsize=(6, 6))
        sc = plt.scatter(cams[:, 0], cams[:, 1], c=probs,
                         cmap="viridis", s=12)
        plt.colorbar(sc, label="probability")
        plt.title(f"{title_prefix}grid {grid_step} m")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    # FOV-weighted arrows
    if show_arrows:
        hfov = math.radians(h_fov_deg)
        vfov = math.radians(v_fov_deg)
        max_len = (0.9 * grid_step) if arrow_len <= 0 else arrow_len

        Qx, Qy, U, V, W = [], [], [], [], []
        stride = max(1, int(arrow_stride))
        grid_to_cam = {int(g): i for i, g in enumerate(cam_linear_indices)}
        for gy_i in range(0, Ny, stride):
            for gx_i in range(0, Nx, stride):
                linear_idx = gy_i * Nx + gx_i
                cam_idx = grid_to_cam.get(linear_idx)
                if cam_idx is None:
                    continue
                dirs = np.asarray(visible_dirs[cam_idx], dtype=np.float32)
                if dirs.size == 0:
                    continue
                yaws = np.empty(len(dirs), dtype=np.float32)
                pits = np.empty(len(dirs), dtype=np.float32)
                for i, v in enumerate(dirs):
                    y, p = dir_to_yaw_pitch(v)
                    yaws[i] = y
                    pits[i] = p
                sel, count = best_fov_window(yaws, pits, hfov, vfov)
                if count == 0:
                    continue
                mdir = average_direction(dirs, sel)
                if mdir is None:
                    continue
                xy = mdir[:2]
                nxy = np.linalg.norm(xy)
                if nxy < 1e-8:
                    continue
                xy_unit = xy / nxy
                Qx.append(cams[cam_idx, 0])
                Qy.append(cams[cam_idx, 1])
                U.append(float(xy_unit[0]))
                V.append(float(xy_unit[1]))
                W.append(count)

        if len(W):
            W = np.array(W, dtype=np.float32)
            scale_fac = np.where(W > 0, W / W.max(), 0.0)
            U = np.array(U) * max_len * scale_fac
            V = np.array(V) * max_len * scale_fac
            plt.figure(figsize=(7, 7))
            plt.quiver(Qx, Qy, U, V, W, angles="xy", scale_units="xy",
                       scale=1.0, cmap="viridis", width=0.004, minlength=0.01)
            plt.colorbar(label="max visible objects within FOV")
            plt.title(f"{title_prefix}FOV-weighted directions "
                      f"(H={h_fov_deg}\u00b0, V={v_fov_deg}\u00b0, stride={stride})")
            plt.axis("equal")
            plt.tight_layout()
            plt.show()
        else:
            print(f"    {title_prefix}arrows: no valid FOV windows")

    # 3-D viewer
    if show_3d:
        vis_mesh = colour_objects(mesh, obj2faces, list(centroids.keys()))
        spheres = []
        for p, col in zip(cams, colormap(probs)):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            s.translate(p)
            s.paint_uniform_color(col)
            spheres.append(s)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1280, height=800,
                          window_name=f"{title_prefix}localisation prob.")
        vis.add_geometry(vis_mesh)
        for s in spheres:
            vis.add_geometry(s)
        vis.get_render_option().point_size = 3
        vis.run()
        vis.destroy_window()

    return probs
