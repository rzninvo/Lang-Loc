#!/usr/bin/env python3
"""Generate D2 dataset slide figures: per-frame visibility + isometric frustum.

Produces two PDFs under ``--output``:

* ``visibility_perframe.pdf`` — perspective render from the frame's camera with
  visible-object faces painted in semantic colours (the rest desaturated). This
  is the "what the rasteriser sees" panel.
* ``visibility_isometric.pdf`` — third-person isometric render of the same
  coloured mesh with a wireframe camera frustum drawn from the frame's pose
  into the scene.

Usage::

    python -m tools.viz.fig_supp_visibility \\
        --dataset scannet --root ./data/scans \\
        --scan-id scene0002_00 --frame-id 000838 \\
        --output docs/figures
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import open3d as o3d
import open3d.core as o3c  # noqa: F401
from open3d.visualization import rendering

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Frustum line-set built from cylinders (so OffscreenRenderer can show it)
# ---------------------------------------------------------------------------

def _make_cylinder_between(start: np.ndarray, end: np.ndarray,
                           radius: float, color: np.ndarray
                           ) -> o3d.geometry.TriangleMesh:
    """Cylinder mesh aligned from start to end (copied from visualize_teaser)."""
    direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return o3d.geometry.TriangleMesh()
    direction /= length
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length,
                                                    resolution=12, split=1)
    cyl.compute_vertex_normals()
    cyl.paint_uniform_color(color)
    mid = (np.asarray(start) + np.asarray(end)) / 2.0
    z_axis = np.array([0.0, 0.0, 1.0])
    rot_axis = np.cross(z_axis, direction)
    rot_norm = np.linalg.norm(rot_axis)
    if rot_norm > 1e-6:
        rot_axis /= rot_norm
        angle = math.acos(np.clip(np.dot(z_axis, direction), -1, 1))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)
        cyl.rotate(R, center=np.zeros(3))
    elif np.dot(z_axis, direction) < 0:
        cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([1.0, 0.0, 0.0]) * math.pi), center=np.zeros(3))
    cyl.translate(mid)
    return cyl


def _make_sphere(center: np.ndarray, radius: float,
                 color: np.ndarray) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=16)
    s.compute_vertex_normals()
    s.paint_uniform_color(color)
    s.translate(center)
    return s


def frustum_world_corners(pose: np.ndarray,
                          fov_h_deg: float, fov_v_deg: float,
                          reach: float
                          ) -> np.ndarray:
    """Return the 5 frustum vertices in world coords: [apex, tl, tr, br, bl]."""
    pose = np.asarray(pose, dtype=np.float64)
    eye = pose[:3, 3]
    right = pose[:3, 0]
    down = pose[:3, 1]
    forward = pose[:3, 2]
    centre = eye + reach * forward
    x_max = reach * math.tan(math.radians(fov_h_deg / 2.0))
    y_max = reach * math.tan(math.radians(fov_v_deg / 2.0))
    tl = centre - x_max * right - y_max * down
    tr = centre + x_max * right - y_max * down
    br = centre + x_max * right + y_max * down
    bl = centre - x_max * right + y_max * down
    return np.stack([eye, tl, tr, br, bl], axis=0)


def build_frustum(pose: np.ndarray,
                  fov_h_deg: float, fov_v_deg: float,
                  reach: float = 2.2,
                  edge_radius: float = 0.008,
                  color: np.ndarray = np.array([0.95, 0.30, 0.18])
                  ) -> List[o3d.geometry.TriangleMesh]:
    """Cylinder-based wireframe frustum for the picker (and as a fallback)."""
    pts = frustum_world_corners(pose, fov_h_deg, fov_v_deg, reach)
    eye, tl, tr, br, bl = pts
    edges = [
        (eye, tl), (eye, tr), (eye, br), (eye, bl),
        (tl, tr), (tr, br), (br, bl), (bl, tl),
    ]
    geoms = [_make_cylinder_between(np.asarray(s), np.asarray(e),
                                    radius=edge_radius, color=color)
             for s, e in edges]
    geoms.append(_make_sphere(eye, radius=edge_radius * 4.0, color=color))
    return geoms


def build_filled_frustum_mesh(pose: np.ndarray,
                              fov_h_deg: float, fov_v_deg: float,
                              reach: float
                              ) -> o3d.geometry.TriangleMesh:
    """Filled triangle mesh for the frustum volume (4 side panels + base quad).

    Used together with the ``defaultLitTransparency`` shader in the
    O3DVisualizer picker so the volume actually looks translucent while the
    user is choosing the camera.
    """
    pts = frustum_world_corners(pose, fov_h_deg, fov_v_deg, reach)
    vertices = np.asarray(pts, dtype=np.float64)
    # apex=0, tl=1, tr=2, br=3, bl=4
    triangles = np.array([
        # 4 side panels
        [0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4],
        # base quad (back of the frustum)
        [1, 2, 3], [1, 3, 4],
    ], dtype=np.int32)
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(vertices)
    m.triangles = o3d.utility.Vector3iVector(triangles)
    m.compute_vertex_normals()
    m.compute_triangle_normals()
    return m


def color_visible_faces(mesh: o3d.geometry.TriangleMesh,
                        tri2obj: np.ndarray,
                        visible_face_ids: List[int],
                        semantic_colors: np.ndarray,
                        desat: float = 0.85
                        ) -> o3d.geometry.TriangleMesh:
    """Return a mesh whose vertices are duplicated per-face so visible faces can
    be coloured independently of their (possibly non-visible) neighbours.

    Faces in ``visible_face_ids`` are painted with their object's semantic
    colour (taken from ``semantic_colors`` at the face's first vertex).
    All other faces keep their original vertex colours, optionally desaturated.
    """
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    n_faces = len(tris)

    if mesh.has_vertex_colors():
        base = np.asarray(mesh.vertex_colors, dtype=np.float64)
    else:
        base = np.full((len(verts), 3), 0.45, dtype=np.float64)

    # Expand: each face gets its own 3 unique vertices.
    flat_ids = tris.reshape(-1)                 # (3 * n_faces,)
    expanded_verts = verts[flat_ids]            # (3 * n_faces, 3)
    expanded_colors = base[flat_ids].copy()     # (3 * n_faces, 3)
    new_tris = np.arange(3 * n_faces, dtype=np.int32).reshape(n_faces, 3)

    # Desaturate everything first
    if desat > 0:
        grey = expanded_colors.mean(axis=1, keepdims=True)
        expanded_colors = (1.0 - desat) * expanded_colors + desat * grey

    # Paint visible faces with their object's semantic colour
    vis_arr = np.asarray(list(visible_face_ids), dtype=np.int64)
    vis_arr = vis_arr[(vis_arr >= 0) & (vis_arr < n_faces)]
    if len(vis_arr) > 0:
        sem_per_face = semantic_colors[tris[vis_arr, 0]]   # (V, 3)
        for k in range(3):
            expanded_colors[3 * vis_arr + k] = sem_per_face

    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(expanded_verts)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_tris)
    new_mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.clip(expanded_colors, 0, 1))
    new_mesh.compute_vertex_normals()
    return new_mesh


def overlay_frustum_pdf(img: np.ndarray,
                        pose: np.ndarray,
                        intrinsic: o3d.camera.PinholeCameraIntrinsic,
                        extrinsic: np.ndarray,
                        fov_h_deg: float, fov_v_deg: float,
                        reach: float,
                        output_path: Path,
                        fill_color=(0.95, 0.30, 0.18),
                        fill_alpha: float = 0.22,
                        edge_alpha: float = 0.80,
                        edge_lw: float = 1.8,
                        dpi: int = 300) -> None:
    """Composite a translucent frustum (vector polygons) on a rendered image.

    The mesh stays raster; the frustum is one filled polygon (the 2D silhouette
    of the pyramid = convex hull of the projected vertices) plus the 8 wireframe
    edges, all as proper PDF vectors with alpha.
    """
    from matplotlib.patches import Polygon as MplPolygon

    H, W = img.shape[:2]

    # 5 frustum vertices in world coords: apex, tl, tr, br, bl
    pts_world = frustum_world_corners(pose, fov_h_deg, fov_v_deg, reach)

    # Accept either an Open3D PinholeCameraIntrinsic OR a 3x3 numpy array.
    # If we get the array form, we assume the K is already at the output
    # resolution (no rescale needed). If we get the Open3D form, rescale to
    # the rendered image size.
    if isinstance(intrinsic, np.ndarray):
        K = np.asarray(intrinsic, dtype=np.float64).copy()
    else:
        K = np.asarray(intrinsic.intrinsic_matrix, dtype=np.float64).copy()
        sx = W / intrinsic.width
        sy = H / intrinsic.height
        K[0, :] *= sx
        K[1, :] *= sy

    pts_h = np.hstack([pts_world, np.ones((5, 1))]).T          # (4, 5)
    pts_cam = (np.asarray(extrinsic, dtype=np.float64) @ pts_h)[:3, :]   # (3, 5)
    z = pts_cam[2, :]
    in_front = z > 1e-3
    proj = K @ pts_cam
    px = proj[0, :] / np.clip(z, 1e-6, None)
    py = proj[1, :] / np.clip(z, 1e-6, None)

    print(f"  Frustum projection: {int(in_front.sum())}/5 vertices in front of "
          f"camera; px range [{px[in_front].min():.0f}, {px[in_front].max():.0f}], "
          f"py range [{py[in_front].min():.0f}, {py[in_front].max():.0f}]; "
          f"image is {W}x{H}")

    # Bake the silhouette fill directly into the raster (guarantees the colour
    # actually renders — bypasses any PDF-alpha quirks of matplotlib).
    img_blended = img.copy()
    visible_idx = np.where(in_front)[0]
    hull_xy = None
    if len(visible_idx) >= 3:
        from matplotlib.path import Path as MplPath
        pts2d = np.column_stack([px[visible_idx], py[visible_idx]])
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts2d)
            hull_xy = pts2d[hull.vertices]
        except Exception as e:
            print(f"  ConvexHull failed ({e}); using polygon-as-given.")
            hull_xy = pts2d

        hull_path = MplPath(hull_xy)
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        pts_grid = np.column_stack([xx.ravel(), yy.ravel()])
        mask = hull_path.contains_points(pts_grid).reshape(H, W)
        n_pix = int(mask.sum())
        print(f"  Silhouette mask covers {n_pix} px ({100 * n_pix / (W * H):.1f}% of frame)")

        fill_rgb = np.array(fill_color, dtype=np.float32) * 255.0
        base = img_blended.astype(np.float32)
        base[mask] = (1.0 - fill_alpha) * base[mask] + fill_alpha * fill_rgb
        img_blended = np.clip(base, 0, 255).astype(np.uint8)

    # Now render via matplotlib: the fill is already baked into img_blended.
    # Edges + apex marker stay as PDF vectors on top.
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img_blended, zorder=1)
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    edges = [(0, 1), (0, 2), (0, 3), (0, 4),
             (1, 2), (2, 3), (3, 4), (4, 1)]
    for a, b in edges:
        if in_front[a] and in_front[b]:
            ax.plot([px[a], px[b]], [py[a], py[b]],
                    color=fill_color, alpha=edge_alpha,
                    linewidth=edge_lw, solid_capstyle="round", zorder=5)

    if in_front[0]:
        ax.plot(px[0], py[0], "o", color=fill_color,
                markersize=5, alpha=1.0,
                markeredgecolor="white", markeredgewidth=0.8, zorder=6)

    fig.savefig(output_path, format="pdf", dpi=dpi,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Also save a PNG sibling for easy visual verification
    png_path = output_path.with_suffix(".png")
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img_blended)
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    for a, b in edges:
        if in_front[a] and in_front[b]:
            ax.plot([px[a], px[b]], [py[a], py[b]],
                    color=fill_color, alpha=edge_alpha,
                    linewidth=edge_lw, solid_capstyle="round", zorder=5)
    if in_front[0]:
        ax.plot(px[0], py[0], "o", color=fill_color,
                markersize=5, alpha=1.0,
                markeredgecolor="white", markeredgewidth=0.8, zorder=6)
    fig.savefig(png_path, format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  Saved PNG sibling: {png_path}")


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _render_with_geoms(mesh: o3d.geometry.TriangleMesh,
                       extra_geoms: List[o3d.geometry.TriangleMesh],
                       width: int, height: int,
                       eye: np.ndarray | None = None,
                       centre: np.ndarray | None = None,
                       up: np.ndarray | None = None,
                       fov_deg: float = 55.0,
                       pinhole: o3d.camera.PinholeCameraParameters | None = None
                       ) -> np.ndarray:
    """Render mesh + extra geoms via OffscreenRenderer.

    If ``pinhole`` is provided, its intrinsic+extrinsic are used (rescaled to
    the target render resolution). Otherwise falls back to (eye, centre, up,
    fov_deg).
    """
    r = rendering.OffscreenRenderer(width, height)
    r.scene.set_background([1.0, 1.0, 1.0, 1.0])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    r.scene.add_geometry("mesh", mesh, mat)
    line_mat = rendering.MaterialRecord()
    line_mat.shader = "defaultLit"
    for i, g in enumerate(extra_geoms):
        r.scene.add_geometry(f"frustum_{i}", g, line_mat)
    r.scene.set_lighting(
        rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0.0, 0.0, 0.0))
    r.scene.scene.enable_sun_light(True)
    r.scene.scene.set_sun_light(
        direction=np.array([0.3, -1.0, -1.0], dtype=np.float32),
        color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        intensity=75000.0)
    if pinhole is not None:
        intr_in = pinhole.intrinsic
        ext = np.asarray(pinhole.extrinsic, dtype=np.float64)
        sx = width / intr_in.width
        sy = height / intr_in.height
        fx, fy = intr_in.get_focal_length()
        cx, cy = intr_in.get_principal_point()
        new_intr = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx * sx, fy * sy, cx * sx, cy * sy)
        r.setup_camera(new_intr, ext)
    else:
        r.setup_camera(fov_deg, centre, eye, up)
    img = np.asarray(r.render_to_image())
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img.astype(np.uint8)


def pick_camera_o3dvis(mesh: o3d.geometry.TriangleMesh,
                       filled_frustum: o3d.geometry.TriangleMesh,
                       edge_geoms: List[o3d.geometry.TriangleMesh],
                       fill_color=(0.95, 0.30, 0.18),
                       fill_alpha: float = 0.30,
                       width: int = 1280, height: int = 720
                       ) -> o3d.camera.PinholeCameraParameters | None:
    """O3DVisualizer-based picker — supports defaultLitTransparency, so the
    user actually sees the translucent red volume while choosing the camera.

    Click the **Capture** action button (in the toolbar at the top) to save
    the camera. Closing the window without clicking Capture aborts.
    """
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    captured: dict = {}

    app = gui.Application.instance
    try:
        app.initialize()
    except Exception:
        # Already initialised — that's fine
        pass

    title = ("Pick camera viewpoint — click 'Capture' (top toolbar) when ready, "
             "or close the window to abort.")
    vis = o3d.visualization.O3DVisualizer(title, width, height)
    try:
        vis.show_settings = False
    except Exception:
        pass
    try:
        vis.show_skybox(False)
    except Exception:
        pass
    try:
        vis.set_background([1.0, 1.0, 1.0, 1.0], None)
    except Exception:
        pass

    mat_lit = rendering.MaterialRecord()
    mat_lit.shader = "defaultLit"
    vis.add_geometry("mesh", mesh, mat_lit)

    mat_trans = rendering.MaterialRecord()
    mat_trans.shader = "defaultLitTransparency"
    mat_trans.base_color = [float(fill_color[0]), float(fill_color[1]),
                            float(fill_color[2]), float(fill_alpha)]
    vis.add_geometry("frustum_fill", filled_frustum, mat_trans)

    for i, g in enumerate(edge_geoms):
        vis.add_geometry(f"edge_{i}", g, mat_lit)

    vis.reset_camera_to_default()

    def on_capture(v):
        cam = v.scene.camera
        view_gl = np.asarray(cam.get_view_matrix(), dtype=np.float64)
        proj = np.asarray(cam.get_projection_matrix(), dtype=np.float64)

        # Filament/O3DVisualizer use the OpenGL convention (camera looks down -Z,
        # Y is up). PinholeCameraParameters.extrinsic, used by OffscreenRenderer,
        # uses the OpenCV convention (camera looks down +Z, Y is down). Convert
        # by flipping the Y and Z axes of the cam-space frame.
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        view = flip @ view_gl

        # Recover vertical FOV from the projection matrix:
        # proj[1, 1] = 1 / tan(fov_y / 2)
        fov_y_rad = 2.0 * float(np.arctan(1.0 / max(proj[1, 1], 1e-9)))
        fy = height / (2.0 * np.tan(fov_y_rad / 2.0))
        fx = fy
        cx, cy = width / 2.0, height / 2.0
        intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = intr
        params.extrinsic = view
        captured["params"] = params
        eye = -view[:3, :3].T @ view[:3, 3]
        print(f"  Camera captured at eye={eye.round(2)}. Closing viewer...")
        gui.Application.instance.quit()

    vis.add_action("Capture", on_capture)

    print("  Opening O3DVisualizer — drag to orbit, scroll to zoom, "
          "right-drag to pan. Click 'Capture' to save.")
    app.add_window(vis)
    app.run()

    return captured.get("params", None)


def pick_camera_with_geoms(mesh: o3d.geometry.TriangleMesh,
                           extra_geoms: List[o3d.geometry.TriangleMesh]
                           ) -> o3d.camera.PinholeCameraParameters | None:
    """Open an interactive Open3D viewer with the mesh + extra geoms.

    Navigate with the mouse (drag = orbit, right-drag = pan, scroll = zoom),
    then press **C** to capture the pinhole camera params and close.
    """
    captured: dict = {}

    def _capture(vis):
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        captured["params"] = params
        ext = np.asarray(params.extrinsic, dtype=np.float64)
        R = ext[:3, :3]
        t = ext[:3, 3]
        eye = -R.T @ t
        print(f"  Camera captured at eye={eye.round(2)}. Closing viewer...")
        vis.close()
        return False

    print("  Opening interactive viewer — navigate, press C to capture, "
          "or close the window to abort.")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Pick camera (press C to capture)",
                      width=1280, height=720)
    vis.add_geometry(mesh)
    for g in extra_geoms:
        vis.add_geometry(g)
    vis.register_key_callback(67, _capture)  # 'C'

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    opt.mesh_show_back_face = True

    vis.run()
    vis.destroy_window()

    return captured.get("params", None)


def save_pinhole_json(params: o3d.camera.PinholeCameraParameters,
                      path: Path) -> None:
    intr = params.intrinsic
    fx, fy = intr.get_focal_length()
    cx, cy = intr.get_principal_point()
    payload = {
        "intrinsic": {
            "width": intr.width, "height": intr.height,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        },
        "extrinsic": np.asarray(params.extrinsic, dtype=np.float64).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2))


def load_pinhole_json(path: Path) -> o3d.camera.PinholeCameraParameters:
    payload = json.loads(path.read_text())
    intr = o3d.camera.PinholeCameraIntrinsic(
        payload["intrinsic"]["width"], payload["intrinsic"]["height"],
        payload["intrinsic"]["fx"], payload["intrinsic"]["fy"],
        payload["intrinsic"]["cx"], payload["intrinsic"]["cy"])
    p = o3d.camera.PinholeCameraParameters()
    p.intrinsic = intr
    p.extrinsic = np.array(payload["extrinsic"], dtype=np.float64)
    return p


def _render_perspective_at_pose(mesh: o3d.geometry.TriangleMesh,
                                pose: np.ndarray,
                                width: int, height: int,
                                fov_deg: float = 60.0) -> np.ndarray:
    r = rendering.OffscreenRenderer(width, height)
    r.scene.set_background([1.0, 1.0, 1.0, 1.0])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    r.scene.add_geometry("mesh", mesh, mat)
    r.scene.set_lighting(
        rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0.0, 0.0, 0.0))
    r.scene.scene.enable_sun_light(True)
    r.scene.scene.set_sun_light(
        direction=np.array([0.3, -1.0, -1.0], dtype=np.float32),
        color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        intensity=75000.0)
    c2w = np.asarray(pose, dtype=np.float64)
    extrinsic = np.linalg.inv(c2w)
    fy = height / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    fx = fy
    cx, cy = width / 2.0, height / 2.0
    intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    r.setup_camera(intr, extrinsic)
    img = np.asarray(r.render_to_image())
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img.astype(np.uint8)


def _save_pdf(img: np.ndarray, pdf_path: Path, dpi: int = 300) -> None:
    """Save the image as a tight PDF and a same-named PNG sibling."""
    H, W = img.shape[:2]
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    fig.savefig(pdf_path, format="pdf", dpi=dpi,
                bbox_inches="tight", pad_inches=0)
    fig.savefig(pdf_path.with_suffix(".png"), format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _isometric_camera(mesh: o3d.geometry.TriangleMesh, up_axis: str,
                      eye_pose: np.ndarray):
    """Pick an isometric eye/centre/up that frames the room and the camera pose."""
    bbox = mesh.get_axis_aligned_bounding_box()
    centre = np.asarray(bbox.get_center(), dtype=np.float64)
    extent = np.asarray(bbox.get_max_bound()) - np.asarray(bbox.get_min_bound())
    diag = float(np.linalg.norm(extent))

    up_idx = {"x_up": 0, "y_up": 1, "z_up": 2}[up_axis]
    up = np.eye(3)[up_idx]

    floor_axes = [i for i in range(3) if i != up_idx]
    offset = np.zeros(3)
    offset[floor_axes[0]] = diag * 0.55
    offset[floor_axes[1]] = -diag * 0.55
    offset[up_idx] = diag * 0.55
    eye = centre + offset
    return eye, centre, up


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="D2 visibility demo figures.")
    ap.add_argument("--dataset", choices=["3rscan", "scannet"], required=True)
    ap.add_argument("--root", type=Path, required=True,
                    help="Dataset root (e.g. ./data/scans)")
    ap.add_argument("--scan-id", "--scan_id", dest="scan_id", required=True)
    ap.add_argument("--frame-id", "--frame_id", dest="frame_id",
                    type=str, required=True,
                    help="Frame id (zero-padded), e.g. '000838'.")
    ap.add_argument("--output", type=Path, default=Path("docs/figures"))
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--persp-width", "--persp_width", dest="persp_width",
                    type=int, default=1296)
    ap.add_argument("--persp-height", "--persp_height", dest="persp_height",
                    type=int, default=968)
    ap.add_argument("--desat", type=float, default=0.85,
                    help="How much to grey-out non-visible mesh (0=keep, 1=full grey).")
    ap.add_argument("--fov-h", "--fov_h", dest="fov_h", type=float, default=58.0)
    ap.add_argument("--fov-v", "--fov_v", dest="fov_v", type=float, default=45.0)
    ap.add_argument("--frustum-reach", "--frustum_reach", dest="frustum_reach",
                    type=float, default=3.0,
                    help="How far the frustum extends in metres.")
    ap.add_argument("--frustum-alpha", "--frustum_alpha",
                    dest="frustum_alpha", type=float, default=0.22,
                    help="Fill alpha of the translucent frustum silhouette (0-1). "
                         "Default 0.22; bump higher for stronger color.")
    ap.add_argument("--interactive", action="store_true",
                    help="Open an Open3D viewer (modern O3DVisualizer with a "
                         "translucent frustum) for the user to pick the "
                         "isometric camera; click the Capture toolbar button. "
                         "Captured camera is cached to "
                         "{output}/{scan_id}_iso_camera.json.")
    ap.add_argument("--classic-picker", "--classic_picker",
                    dest="classic_picker", action="store_true",
                    help="Force the classic Open3D Visualizer picker (press C "
                         "to capture). Use only if the modern O3DVisualizer "
                         "can't open on this system.")
    ap.add_argument("--camera-json", "--camera_json", dest="camera_json",
                    type=Path, default=None,
                    help="Reuse a previously-saved camera JSON (skip the "
                         "interactive picker). Default: try "
                         "{output}/{scan_id}_iso_camera.json.")
    return ap.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from tools.viz.visualize_teaser import (
        load_scene_any, detect_up_axis, load_semantic_vertex_colors,
    )

    scan_dir = args.root / args.scan_id
    print(f"[1/4] Loading scene: {args.scan_id}")
    mesh, tri2obj, obj2faces = load_scene_any(scan_dir, dataset=args.dataset)
    up_axis = detect_up_axis(mesh)

    # 2. Read this frame's visibility cache entry
    cache_path = scan_dir / "output" / "cache" / f"{args.scan_id}.json"
    print(f"[2/4] Reading visibility cache: {cache_path}")
    cache = json.loads(cache_path.read_text())
    fr = next((f for f in cache if str(f.get("fid")) == args.frame_id), None)
    if fr is None:
        raise KeyError(f"Frame {args.frame_id!r} not found in {cache_path}")
    visible_face_ids = [int(x) for x in fr.get("visible_face_ids", [])]
    print(f"  {len(visible_face_ids)} faces visible in frame {args.frame_id}; "
          f"{len(fr['visible_objects'])} objects passed the cascade")

    # 3. Build face-level coloured mesh (only rasterised faces get instance colours)
    sem_colors = load_semantic_vertex_colors(scan_dir, args.scan_id, args.dataset)
    if sem_colors is None:
        raise FileNotFoundError("Semantic vertex colours not found "
                                "(needed for instance colouring).")
    coloured = color_visible_faces(mesh, tri2obj, visible_face_ids,
                                   sem_colors, desat=args.desat)

    # 4. Frame's pose
    desc_path = scan_dir / "output" / "descriptions" / f"{args.frame_id}.json"
    desc = json.loads(desc_path.read_text())
    pose = np.array(desc["scene_pose"], dtype=np.float64)

    # 5. Per-frame perspective render (what the rasteriser sees)
    print(f"[3/4] Rendering per-frame view ({args.persp_width}x{args.persp_height})")
    persp = _render_perspective_at_pose(coloured, pose,
                                        width=args.persp_width,
                                        height=args.persp_height,
                                        fov_deg=args.fov_v + 5)
    out_persp = args.output / f"visibility_perframe_{args.scan_id}.pdf"
    _save_pdf(persp, out_persp)
    print(f"  Saved: {out_persp}")

    # 6. Isometric render with translucent frustum overlay
    print(f"[4/4] Rendering isometric view ({args.width}x{args.height}) "
          f"+ translucent frustum overlay")
    cached_cam_path = (args.camera_json
                       or args.output / f"{args.scan_id}_iso_camera.json")
    pinhole = None
    if args.interactive:
        # Build the picker geometries:
        #   - cylinder edges (visible in any picker)
        #   - a filled frustum mesh used by the modern O3DVisualizer with the
        #     defaultLitTransparency shader so the user sees true translucency.
        picker_edges = build_frustum(pose,
                                     fov_h_deg=args.fov_h,
                                     fov_v_deg=args.fov_v,
                                     reach=args.frustum_reach)
        picker_fill = build_filled_frustum_mesh(pose,
                                                fov_h_deg=args.fov_h,
                                                fov_v_deg=args.fov_v,
                                                reach=args.frustum_reach)
        if args.classic_picker:
            pinhole = pick_camera_with_geoms(coloured, picker_edges)
        else:
            try:
                pinhole = pick_camera_o3dvis(
                    coloured, picker_fill, picker_edges,
                    fill_alpha=max(args.frustum_alpha * 1.4, 0.20))
            except Exception as e:
                print(f"  O3DVisualizer picker failed ({e}); "
                      "falling back to classic picker.")
                pinhole = pick_camera_with_geoms(coloured, picker_edges)
        if pinhole is not None:
            save_pinhole_json(pinhole, cached_cam_path)
            print(f"  Cached camera: {cached_cam_path}")
    elif cached_cam_path.exists():
        pinhole = load_pinhole_json(cached_cam_path)
        print(f"  Reusing cached camera: {cached_cam_path}")

    # Render mesh-only (no frustum) at the chosen viewpoint
    if pinhole is not None:
        iso_mesh = _render_with_geoms(coloured, [],
                                      width=args.width, height=args.height,
                                      pinhole=pinhole)
        iso_intrinsic = pinhole.intrinsic
        iso_extrinsic = np.asarray(pinhole.extrinsic, dtype=np.float64)
    else:
        eye, centre, up = _isometric_camera(coloured, up_axis, pose)
        iso_mesh = _render_with_geoms(coloured, [],
                                      width=args.width, height=args.height,
                                      eye=eye, centre=centre, up=up,
                                      fov_deg=55.0)
        # Build the equivalent intrinsic/extrinsic for the projection
        fy = args.height / (2.0 * math.tan(math.radians(55.0 / 2.0)))
        fx = fy
        iso_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            args.width, args.height, fx, fy,
            args.width / 2.0, args.height / 2.0)
        # world→cam from look-at: forward = centre - eye
        f = centre - eye; f /= np.linalg.norm(f)
        s = np.cross(f, up); s /= max(np.linalg.norm(s), 1e-9)
        u = np.cross(s, f)
        R = np.stack([s, -u, f], axis=0)
        t = -R @ eye
        iso_extrinsic = np.eye(4)
        iso_extrinsic[:3, :3] = R
        iso_extrinsic[:3, 3] = t

    out_iso = args.output / f"visibility_isometric_{args.scan_id}.pdf"
    overlay_frustum_pdf(
        iso_mesh, pose, iso_intrinsic, iso_extrinsic,
        fov_h_deg=args.fov_h, fov_v_deg=args.fov_v,
        reach=args.frustum_reach,
        output_path=out_iso,
        fill_alpha=args.frustum_alpha,
    )
    print(f"  Saved: {out_iso}")

    print("Done.")


if __name__ == "__main__":
    main()
