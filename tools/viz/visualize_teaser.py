#!/usr/bin/env python3
"""ECCV teaser visualization: perspective render, top-down view, and localization heatmap overlay.

Supports both 3RScan and ScanNet scenes. Accepts a natural-language query
(``--query``) or loads a per-frame description JSON (``--frame-id``), runs
the full localization pipeline, and composites a probability heatmap onto
the top-down view. Optionally produces a scene graph visualization.

Usage::

    # Manual query (requires OPENAI_API_KEY):
    python -m tools.viz.visualize_teaser \
        --dataset 3rscan \
        --root ./data/3RScan \
        --scan-id 0ad2d3a1-79e2-2212-9b99-a96495d9f7fe \
        --query "I can see a sofa on my left and a bookshelf in front of me" \
        --graphs-3dssg ./data/3DSSG/graphs.pt \
        --output ./teaser_output

    # From per-frame description JSON (no API key needed):
    python -m tools.viz.visualize_teaser \
        --dataset scannet \
        --root ./data/scans \
        --scan-id scene0000_01 \
        --frame-id 001158 \
        --graphs-3dssg ./data/3DSSG/graphs.pt \
        --output ./teaser_output \
        --scene-graph --direction-field
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Load .env from project root
_dotenv_path = Path(__file__).resolve().parents[2] / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path)

import numpy as np
import torch
import open3d as o3d
import open3d.core as o3c
from open3d.visualization import rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------

def detect_up_axis(mesh: o3d.geometry.TriangleMesh) -> str:
    """Auto-detect the vertical (up) axis from mesh vertex extents.

    Indoor rooms are wider than tall, so the axis with the smallest
    extent is the height axis.
    """
    verts = np.asarray(mesh.vertices)
    extents = verts.max(0) - verts.min(0)
    up_idx = int(np.argmin(extents))
    axis_name = ["x_up", "y_up", "z_up"][up_idx]
    print(f"  Auto-detected up axis: {axis_name} "
          f"(extents X={extents[0]:.1f}m, Y={extents[1]:.1f}m, Z={extents[2]:.1f}m)")
    return axis_name


def load_scene_3rscan(scan_dir: Path):
    """Load a 3RScan scene. Delegates to langloc.localization.grid.load_scene."""
    from langloc.localization.grid import load_scene
    return load_scene(scan_dir)


def load_scene_scannet(scan_dir: Path, scan_id: str):
    """Load a ScanNet scene mesh and build per-triangle / per-object maps.

    Reads the *_vh_clean_2.ply mesh, segmentation JSON, and aggregation JSON
    to produce the same (mesh, tri2obj, obj2faces) tuple as the 3RScan loader.

    Args:
        scan_dir: Path to the scene directory (e.g. ``data/scans/scene0000_00``).
        scan_id: Scene identifier string (e.g. ``scene0000_00``).

    Returns:
        (mesh, tri2obj, obj2faces) matching the 3RScan loader interface.
    """
    ply = scan_dir / f"{scan_id}_vh_clean_2.ply"
    if not ply.exists():
        raise FileNotFoundError(f"ScanNet mesh not found: {ply}")
    mesh = o3d.io.read_triangle_mesh(str(ply))
    mesh.compute_vertex_normals()

    # Load segmentation
    segs_json = scan_dir / f"{scan_id}_vh_clean_2.0.010000.segs.json"
    if not segs_json.exists():
        segs_json = scan_dir / f"{scan_id}_vh_clean_2.segs.json"
    if not segs_json.exists():
        raise FileNotFoundError(f"Segs JSON not found for {scan_id}")

    agg_json = scan_dir / f"{scan_id}.aggregation.json"
    if not agg_json.exists():
        raise FileNotFoundError(f"Aggregation JSON not found for {scan_id}")

    segs = json.loads(segs_json.read_text())
    agg = json.loads(agg_json.read_text())

    vert_seg = np.array(segs["segIndices"], dtype=np.int32)

    seg_to_obj: Dict[int, int] = {}
    for g in agg["segGroups"]:
        oid = int(g["objectId"])
        for s in g["segments"]:
            seg_to_obj[int(s)] = oid

    # Per-vertex object ID
    v_oid = np.array([seg_to_obj.get(int(s), 0) for s in vert_seg], dtype=np.int32)

    # Per-triangle object ID (majority vote)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    tri2obj = np.array([np.bincount(v_oid[t]).argmax() for t in tris],
                       dtype=np.int32)

    obj2faces: Dict[int, np.ndarray] = {}
    for fid, oid in enumerate(tri2obj):
        if oid != 0:
            obj2faces.setdefault(int(oid), []).append(fid)
    obj2faces = {k: np.asarray(v, dtype=np.int32) for k, v in obj2faces.items()}

    return mesh, tri2obj, obj2faces


def load_semantic_vertex_colors(scan_dir: Path, scan_id: str,
                                dataset: str) -> np.ndarray | None:
    """Load per-vertex semantic instance colours from the labels PLY.

    For ScanNet: ``{scan_id}_vh_clean_2.labels.ply``
    For 3RScan: returns None (no separate labels file).

    Returns:
        (N, 3) float64 array in [0, 1] or None.
    """
    if dataset == "scannet":
        labels_ply = scan_dir / f"{scan_id}_vh_clean_2.labels.ply"
    else:
        return None
    if not labels_ply.exists():
        return None
    m = o3d.io.read_triangle_mesh(str(labels_ply))
    if not m.has_vertex_colors():
        return None
    return np.asarray(m.vertex_colors, dtype=np.float64)


def color_matched_objects(mesh: o3d.geometry.TriangleMesh,
                          tri2obj: np.ndarray,
                          matched_obj_ids: List[int],
                          semantic_colors: np.ndarray,
                          desat: float = 0.0) -> o3d.geometry.TriangleMesh:
    """Return a copy of the mesh with matched objects painted in their semantic
    instance colours and the rest optionally desaturated.

    Args:
        desat: Desaturation strength for non-matched vertices (0.0 = original
               colours, 1.0 = fully grey). Matched objects always use their
               semantic colours at full saturation.
    """
    import copy
    mesh_colored = copy.deepcopy(mesh)

    verts = np.asarray(mesh_colored.vertices)
    tris = np.asarray(mesh_colored.triangles, dtype=np.int32)
    n_verts = len(verts)

    if mesh_colored.has_vertex_colors():
        base = np.asarray(mesh_colored.vertex_colors, dtype=np.float64).copy()
    else:
        base = np.full((n_verts, 3), 0.45, dtype=np.float64)

    # Desaturate non-matched regions
    if desat > 0:
        grey = base.mean(axis=1, keepdims=True)
        new_colors = (1.0 - desat) * base + desat * grey
    else:
        new_colors = base

    # Paint matched objects with their semantic colours
    matched_set = set(matched_obj_ids)
    for fid, oid in enumerate(tri2obj):
        if int(oid) in matched_set:
            for vi in tris[fid]:
                new_colors[vi] = semantic_colors[vi]

    mesh_colored.vertex_colors = o3d.utility.Vector3dVector(np.clip(new_colors, 0, 1))
    return mesh_colored


# ---------------------------------------------------------------------------
# Perspective render (Open3D OffscreenRenderer)
# ---------------------------------------------------------------------------

def _up_vector(up_axis: str) -> np.ndarray:
    """Return the unit up-direction vector for the given axis convention."""
    return {"x_up": np.array([1., 0., 0.]),
            "y_up": np.array([0., 1., 0.]),
            "z_up": np.array([0., 0., 1.])}[up_axis]


def _auto_camera(mesh: o3d.geometry.TriangleMesh, up_axis: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a reasonable default camera: eye, centre, up."""
    bbox = mesh.get_axis_aligned_bounding_box()
    centre = np.asarray(bbox.get_center(), dtype=np.float64)
    extent = np.asarray(bbox.get_max_bound()) - np.asarray(bbox.get_min_bound())
    diag = float(np.linalg.norm(extent))

    up = _up_vector(up_axis)
    # Place camera above and to the side: offset along each floor axis + up
    up_idx = {"x_up": 0, "y_up": 1, "z_up": 2}[up_axis]
    offset = np.array([diag * 0.6, diag * 0.6, diag * 0.6])
    offset[up_idx] = diag * 0.45  # slightly above centre
    # Negate one floor axis to get a nicer angle
    floor_axes = [i for i in range(3) if i != up_idx]
    offset[floor_axes[1]] *= -0.8
    eye = centre + offset

    return eye, centre, up


def pick_camera_interactive(mesh: o3d.geometry.TriangleMesh,
                            up_axis: str):
    """Open an interactive Open3D viewer for the user to pick a camera viewpoint.

    The user navigates to their desired viewpoint, then presses **C** to
    capture the full pinhole camera parameters and close the window.

    Returns:
        o3d.camera.PinholeCameraParameters captured from the viewer, or None.
    """
    captured = {}

    def _capture_callback(vis):
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

    print("  Opening interactive viewer — navigate to desired viewpoint, then press C to capture.")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Pick camera viewpoint (press C to capture)", width=1280, height=720)
    vis.add_geometry(mesh)
    vis.register_key_callback(67, _capture_callback)  # 'C' key

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    opt.mesh_show_back_face = True

    vis.run()
    vis.destroy_window()

    return captured.get("params", None)


def render_perspective(mesh: o3d.geometry.TriangleMesh,
                       width: int, height: int,
                       up_axis: str,
                       camera_json: str | None = None,
                       interactive: bool = False,
                       frame_pose: np.ndarray | None = None) -> np.ndarray:
    """High-quality perspective render using Open3D OffscreenRenderer.

    Args:
        interactive: If True, opens an interactive viewer for the user to pick
            the camera viewpoint before rendering.
        frame_pose: 4x4 cam-to-world SE(3) matrix from a description JSON.
            If provided, renders from this exact viewpoint.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    r = rendering.OffscreenRenderer(width, height)
    scene = r.scene

    scene.set_background([1.0, 1.0, 1.0, 1.0])

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    scene.add_geometry("mesh", mesh, mat)

    scene.set_lighting(
        rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0.0, 0.0, 0.0))
    scene.scene.enable_sun_light(True)
    scene.scene.set_sun_light(
        direction=np.array([0.3, -1.0, -1.0], dtype=np.float32),
        color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        intensity=75000.0,
    )

    pinhole = None
    if frame_pose is not None:
        # Use frame's camera pose: cam2world → invert to get world2cam (extrinsic)
        c2w = np.asarray(frame_pose, dtype=np.float64)
        extrinsic = np.linalg.inv(c2w)
        # Build a reasonable intrinsic for the render resolution
        fov_deg = 60.0
        fy = height / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
        fx = fy  # square pixels
        cx, cy = width / 2.0, height / 2.0
        intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        r.setup_camera(intr, extrinsic)
    elif camera_json and Path(camera_json).exists():
        cam = json.loads(Path(camera_json).read_text())
        eye = np.array(cam["eye"], dtype=np.float64)
        centre = np.array(cam["center"], dtype=np.float64)
        up = np.array(cam["up"], dtype=np.float64)
        fov_deg = 60.0
        r.setup_camera(fov_deg, centre, eye, up)
    elif interactive:
        pinhole = pick_camera_interactive(mesh, up_axis)
        if pinhole is None:
            print("  No camera captured — falling back to auto camera.")

    if pinhole is not None:
        # Use the exact pinhole intrinsic + extrinsic from the interactive viewer
        intr = pinhole.intrinsic
        ext = np.asarray(pinhole.extrinsic, dtype=np.float64)
        # Rebuild intrinsic at the target render resolution
        sx = width / intr.width
        sy = height / intr.height
        fx = intr.get_focal_length()[0] * sx
        fy = intr.get_focal_length()[1] * sy
        cx = intr.get_principal_point()[0] * sx
        cy = intr.get_principal_point()[1] * sy
        new_intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        r.setup_camera(new_intr, ext)
    elif frame_pose is None and not (camera_json and Path(camera_json).exists()):
        eye, centre, up = _auto_camera(mesh, up_axis)
        fov_deg = 60.0
        r.setup_camera(fov_deg, centre, eye, up)

    img = r.render_to_image()
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr.astype(np.uint8)


# ---------------------------------------------------------------------------
# Top-down render (reusing topdown_3rscan utilities)
# ---------------------------------------------------------------------------

def render_topdown(mesh: o3d.geometry.TriangleMesh,
                   up_axis: str,
                   topdown_size: int,
                   floor_pct: float = 0.2,
                   ceiling_pct: float = 95.0,
                   cutoff_m: float = 2.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a top-down view with ceiling removed.

    Returns:
        (image, intrinsic_4x4, extrinsic_4x4) where image is (H, W, 3) uint8.
    """
    from langloc.baselines.topdown_3rscan import (
        filter_faces_by_height, build_filtered_mesh
    )

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    if colors.shape != (len(vertices), 3) or not np.isfinite(colors).all() or np.allclose(colors, 0.0):
        colors = np.tile(np.array([[0.45, 0.45, 0.45]], dtype=np.float32), (len(vertices), 1))

    height_axis = {"x_up": 0, "y_up": 1, "z_up": 2}[up_axis]
    # plane name for camera vectors: project onto the two non-height axes
    plane = {0: "yz", 1: "xz", 2: "xy"}[height_axis]

    face_mask = filter_faces_by_height(
        vertices, faces, height_axis,
        floor_percentile=floor_pct,
        ceiling_percentile=ceiling_pct,
        cutoff_above_ground_m=cutoff_m,
    )
    filtered = build_filtered_mesh(mesh, vertices, faces, face_mask, colors)

    if len(filtered.triangles) == 0:
        raise RuntimeError("All faces removed by height filtering — try wider percentiles")

    # Render using legacy Visualizer (correct intrinsic/extrinsic for heatmap projection)
    size = max(1024, topdown_size)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="topdown", width=size, height=size, visible=False)
    vis.add_geometry(filtered)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    opt.mesh_show_back_face = True
    opt.light_on = True

    bbox = filtered.get_axis_aligned_bounding_box()
    lookat = bbox.get_center()

    if plane == "xy":
        front = np.array([0.0, 0.0, 1.0])
        up = np.array([0.0, 1.0, 0.0])
    elif plane == "xz":
        front = np.array([0.0, 1.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
    else:
        front = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])

    ctr = vis.get_view_control()
    ctr.set_lookat(lookat)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(0.5)

    vis.poll_events()
    vis.update_renderer()

    params = ctr.convert_to_pinhole_camera_parameters()
    intrinsic = np.asarray(params.intrinsic.intrinsic_matrix, dtype=np.float64)
    extrinsic = np.asarray(params.extrinsic, dtype=np.float64)

    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    img_arr = (np.asarray(img) * 255).astype(np.uint8)
    return img_arr, intrinsic, extrinsic


# ---------------------------------------------------------------------------
# Localization pipeline
# ---------------------------------------------------------------------------

def _sample_grid_any_axis(verts: np.ndarray, step: float,
                          up_axis: str, eye_height: float = 1.6,
                          mesh: o3d.geometry.TriangleMesh | None = None,
                          ) -> np.ndarray:
    """Sample a dense floor grid with cameras at eye_height above the floor.

    Unlike grid.sample_grid (which assumes z_up), this supports any up axis.
    When *mesh* is provided, grid points that fall outside the scene geometry
    are pruned via downward raycasting.
    """
    up_idx = {"x_up": 0, "y_up": 1, "z_up": 2}[up_axis]
    floor_axes = [i for i in range(3) if i != up_idx]
    a0, a1 = floor_axes

    g0 = np.arange(verts[:, a0].min(), verts[:, a0].max() + 1e-4, step)
    g1 = np.arange(verts[:, a1].min(), verts[:, a1].max() + 1e-4, step)
    v0, v1 = np.meshgrid(g0, g1, indexing="xy")
    n = v0.size

    height = verts[:, up_idx].min() + eye_height
    cams = np.empty((n, 3), dtype=np.float64)
    cams[:, a0] = v0.ravel()
    cams[:, a1] = v1.ravel()
    cams[:, up_idx] = height

    # Filter: keep only grid points inside the scene mesh
    if mesh is not None:
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

        # Cast a ray downward from each camera position
        down = np.zeros(3, dtype=np.float32)
        down[up_idx] = -1.0
        dirs = np.tile(down, (n, 1))
        rays = np.hstack([cams.astype(np.float32), dirs])
        hits = rc.cast_rays(o3d.core.Tensor(rays))
        t_hit = hits["t_hit"].numpy()

        # A point is inside the scene if the downward ray hits something
        # within a reasonable distance (eye_height + margin)
        inside = np.isfinite(t_hit) & (t_hit > 0) & (t_hit < eye_height + 1.0)
        cams = cams[inside]
        print(f"  Grid: {n} total → {len(cams)} inside scene "
              f"({len(cams)/n*100:.0f}%)")

    return cams


def _per_node_best_match(qg, sg, k: int) -> List[int]:
    """Match each query node to its best scene-graph node, then optionally
    fill remaining slots with global top-k (avoiding duplicates).

    This prevents a single repeated label (e.g. 9 "wall" nodes) from
    dominating all k slots.
    """
    import torch.nn.functional as F

    qf, _, _ = qg.to_pyg()
    sf, _, _ = sg.to_pyg()
    qf = F.normalize(torch.tensor(np.asarray(qf), dtype=torch.float32), dim=1)
    sf = F.normalize(torch.tensor(np.asarray(sf), dtype=torch.float32), dim=1)

    sim = qf @ sf.T  # (|Q|, |S|)
    sids = list(sg.nodes)

    picks = []
    # Phase 1: best match per query node
    for qi in range(sim.size(0)):
        best_si = int(sim[qi].argmax())
        sid = sids[best_si]
        if sid not in picks:
            picks.append(sid)

    # Phase 2: fill remaining slots from global ranking
    if len(picks) < k:
        topv, topi = torch.topk(sim.flatten(), min(k * 3, sim.numel()))
        S = sf.size(0)
        for idx in topi.tolist():
            sid = sids[idx % S]
            if sid not in picks:
                picks.append(sid)
            if len(picks) >= k:
                break

    return picks[:k]


def _build_edge_lookup(sg) -> Dict[Tuple[int, int], Tuple[str, np.ndarray]]:
    """Build a fast edge lookup from a SceneGraph's edge lists.

    Returns:
        dict mapping ``(from_id, to_id)`` to ``(relation_str, relation_embedding)``.
    """
    lookup: Dict[Tuple[int, int], Tuple[str, np.ndarray]] = {}
    if not sg.edge_idx or len(sg.edge_idx) < 2:
        return lookup
    for i, (f, t) in enumerate(zip(sg.edge_idx[0], sg.edge_idx[1])):
        emb = np.asarray(sg.edge_features[i], dtype=np.float32) if i < len(sg.edge_features) else None
        lookup[(int(f), int(t))] = (sg.edge_relations[i], emb)
    return lookup


def _relation_bonus(q_edge_lookup: Dict, s_edge_lookup: Dict,
                    assignment: Dict[int, int]) -> float:
    """Score how well a trial assignment preserves query-graph relations.

    For each query edge ``(qi, qj)``, checks if the assigned scene nodes
    ``(si, sj)`` share a matching relation in the scene graph, using cosine
    similarity of relation embeddings.

    Returns:
        Average relation similarity in [0, 1], or 0.0 if no edges can be checked.
    """
    total, checked = 0.0, 0
    for (qi, qj), (q_rel, q_emb) in q_edge_lookup.items():
        si = assignment.get(qi)
        sj = assignment.get(qj)
        if si is None or sj is None:
            continue
        # Check both directions in the scene graph
        s_edge = s_edge_lookup.get((si, sj)) or s_edge_lookup.get((sj, si))
        checked += 1
        if s_edge is not None and q_emb is not None and s_edge[1] is not None:
            q_n = q_emb / (np.linalg.norm(q_emb) + 1e-8)
            s_n = s_edge[1] / (np.linalg.norm(s_edge[1]) + 1e-8)
            total += max(float(q_n @ s_n), 0.0)
    return total / checked if checked > 0 else 0.0


def _relation_aware_match(qg, sg, k: int, alpha: float = 0.5) -> List[int]:
    """Match query nodes to scene-graph nodes using both node similarity
    and relation consistency.

    For each query node, considers the top-C candidates by cosine similarity,
    then re-ranks by ``(1-alpha)*node_sim + alpha*relation_bonus``.
    Greedy: most-confident nodes are assigned first to anchor the graph.

    Args:
        alpha: Blend weight. 0 = pure node similarity (same as
            ``_per_node_best_match``), 1 = pure relation consistency.
    """
    import torch.nn.functional as F

    qf, _, _ = qg.to_pyg()
    sf, _, _ = sg.to_pyg()
    qf = F.normalize(torch.tensor(np.asarray(qf), dtype=torch.float32), dim=1)
    sf = F.normalize(torch.tensor(np.asarray(sf), dtype=torch.float32), dim=1)

    sim = qf @ sf.T  # (|Q|, |S|)
    q_ids = list(qg.nodes)
    s_ids = list(sg.nodes)

    q_edge_lookup = _build_edge_lookup(qg)
    s_edge_lookup = _build_edge_lookup(sg)

    C = min(10, len(s_ids))
    topk_vals, topk_idx = sim.topk(C, dim=1)  # (|Q|, C)

    # Greedy assignment: anchor most-confident query nodes first
    assignment: Dict[int, int] = {}
    used_sids: set = set()
    q_order = sorted(range(len(q_ids)), key=lambda qi: -float(sim[qi].max()))

    for qi in q_order:
        qid = q_ids[qi]
        best_score = -float('inf')
        best_sid = None

        for c in range(C):
            si = int(topk_idx[qi, c])
            sid = s_ids[si]
            if sid in used_sids:
                continue

            node_sim = float(topk_vals[qi, c])

            # Tentatively assign and score relation consistency
            trial = dict(assignment)
            trial[qid] = sid
            r_bonus = _relation_bonus(q_edge_lookup, s_edge_lookup, trial)

            combined = (1.0 - alpha) * node_sim + alpha * r_bonus
            if combined > best_score:
                best_score = combined
                best_sid = sid

        if best_sid is not None:
            assignment[qid] = best_sid
            used_sids.add(best_sid)

    picks = list(assignment.values())

    # Phase 2: fill remaining slots from global ranking
    if len(picks) < k:
        topv, topi = torch.topk(sim.flatten(), min(k * 3, sim.numel()))
        S = sf.size(0)
        for idx in topi.tolist():
            sid = s_ids[idx % S]
            if sid not in picks:
                picks.append(sid)
            if len(picks) >= k:
                break

    return picks[:k]


def run_localization(mesh: o3d.geometry.TriangleMesh,
                     tri2obj: np.ndarray,
                     obj2faces: Dict[int, np.ndarray],
                     query: str = None,
                     query_sg=None,
                     graphs_3dssg: str = None,
                     scan_id: str = "",
                     embedding_type: str = "word2vec",
                     top_k: int = None,
                     grid_step: float = 0.25,
                     h_fov_deg: float = 100.0,
                     v_fov_deg: float = 60.0,
                     up_axis: str = "z_up",
                     score_tau: float = 0.0) -> Tuple[np.ndarray, np.ndarray, List[int], np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Run the full localization pipeline from a text query or pre-built SceneGraph.

    Args:
        query: Natural-language query string (uses text_to_scenegraph, needs OPENAI_API_KEY).
        query_sg: Pre-built SceneGraph (e.g. from frame_to_scenegraph). Takes priority over query.

    Returns:
        (cams, probs, matched_obj_ids, pred_pos, pred_dir, cam_dirs)
        where cam_dirs is (N, 3) average viewing direction per camera (zero if no visible objects).
    """
    from langloc.localization.grid import first_hit_is_object
    from langloc.localization.visualization import dir_to_yaw_pitch, best_fov_window, average_direction
    from tools.viz.visualize_loc_from_query import load_scene_graph_for_scan

    # Build or use provided query scene graph
    if query_sg is not None:
        qg = query_sg
    elif query is not None:
        from langloc.graph_matching.single_inference import text_to_scenegraph
        qg = text_to_scenegraph(query, embedding_type=embedding_type,
                                use_attributes=True, scene_id="query_teaser",
                                debug=False)
    else:
        raise ValueError("Either query or query_sg must be provided")

    # Load target scene graph
    sg = load_scene_graph_for_scan(graphs_3dssg, scan_id,
                                    max_dist=2.0, embedding_type=embedding_type,
                                    use_attributes=True)

    # Recompute word2vec embeddings to match the current spaCy model.
    # The pre-generated .pt file may have been built with a different model,
    # causing cosine similarities to be near zero.
    if embedding_type == "word2vec":
        from langloc.graphs.create_text_embeddings import create_embedding_nlp
        for nid, node in sg.nodes.items():
            fresh_label = create_embedding_nlp(node.label)
            node.label_features = fresh_label
            fresh_attrs = [create_embedding_nlp(a) for a in (node.attributes or [])]
            node.attribute_features = fresh_attrs
            node.features = node.set_features(fresh_label, fresh_attrs, use_attributes=True)

    # Auto top_k: match exactly as many objects as the query mentions.
    if top_k is None:
        top_k = len(qg.nodes)

    # Object matching: per-query-node best match instead of global top-k.
    # Global top-k can be dominated by duplicate labels (e.g. 9 "wall" nodes).
    # Instead, for each query node, pick the best matching scene node.
    obj_ids = _relation_aware_match(qg, sg, top_k)
    if not obj_ids:
        print("  No cosine matches found between query and scene.")
        return np.empty((0, 3)), np.array([]), [], None, None, np.empty((0, 3))

    print(f"  Matched {len(obj_ids)} objects: {obj_ids}")

    # Grid sampling + raycasting
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    verts = np.asarray(mesh.vertices)
    cams = _sample_grid_any_axis(verts, step=grid_step, up_axis=up_axis,
                                  mesh=mesh)

    tris = np.asarray(mesh.triangles)
    centroids: Dict[int, np.ndarray] = {}
    for oid in obj_ids:
        faces = obj2faces.get(oid)
        if faces is not None and len(faces):
            centroids[oid] = verts[np.unique(tris[faces].ravel())].mean(0)

    if not centroids:
        print("  No centroids for matched objects")
        return cams, np.zeros(len(cams)), obj_ids, None, None, np.zeros((len(cams), 3))

    # Weight objects by distinctiveness: generic labels (wall, floor,
    # ceiling) are visible from almost anywhere and don't help localize.
    # Each object contributes its weight to the score instead of a flat 1.
    _GENERIC_LOC_LABELS = frozenset({
        "wall", "floor", "ceiling", "object", "doorframe", "door"})
    oid_label = {oid: sg.nodes[oid].label for oid in obj_ids if oid in sg.nodes}
    oid_weight: Dict[int, float] = {}
    for oid in centroids:
        lbl = oid_label.get(oid, "").lower().strip()
        oid_weight[oid] = 0.1 if lbl in _GENERIC_LOC_LABELS else 1.0

    # Visibility tally — weighted by object distinctiveness
    visible_dirs: List[List[np.ndarray]] = [[] for _ in range(len(cams))]
    visible_weights: List[List[float]] = [[] for _ in range(len(cams))]
    for idx, cam in enumerate(cams):
        for oid, cen in centroids.items():
            if first_hit_is_object(cam, cen, oid, rc, tri2obj):
                d = cen - cam
                ln = np.linalg.norm(d)
                if ln > 1e-6:
                    visible_dirs[idx].append(d / ln)
                    visible_weights[idx].append(oid_weight[oid])

    counts = np.array([len(v) for v in visible_dirs], dtype=np.int32)
    if counts.sum() == 0:
        print("  Matched objects not visible from any grid camera")
        return cams, np.zeros(len(cams)), obj_ids, None, None, np.zeros((len(cams), 3))

    # Compute FOV-constrained arrow weights with distinctiveness weighting.
    # arrow_weights_generic gives the FOV selection mask; we sum the per-
    # object weights within that mask instead of raw count.
    from langloc.localization.coarse_search import arrow_weights_generic
    hfov = math.radians(h_fov_deg)
    vfov = math.radians(v_fov_deg)
    _, arrow_d = arrow_weights_generic(cams, visible_dirs, hfov, vfov)

    # Recompute weighted arrow scores using per-object distinctiveness
    arrow_w = np.zeros(len(cams), dtype=np.float64)
    for idx in range(len(cams)):
        if not visible_dirs[idx]:
            continue
        dirs_arr = np.asarray(visible_dirs[idx], dtype=np.float32)
        w_arr = np.array(visible_weights[idx], dtype=np.float64)
        yaws = np.array([dir_to_yaw_pitch(d)[0] for d in dirs_arr])
        pits = np.array([dir_to_yaw_pitch(d)[1] for d in dirs_arr])
        sel, count = best_fov_window(yaws, pits, hfov, vfov)
        if count > 0:
            arrow_w[idx] = float(w_arr[sel].sum())

    # Weighted visibility as fallback
    weighted_counts = np.array([sum(w) for w in visible_weights], dtype=np.float64)

    if arrow_w.sum() > 0:
        probs = arrow_w.copy()
    else:
        probs = weighted_counts

    visible_mask = probs > 0
    if score_tau > 0 and visible_mask.any():
        logits = probs[visible_mask] / score_tau
        logits -= logits.max()
        softmax_vals = np.exp(logits)
        probs[:] = 0.0
        probs[visible_mask] = softmax_vals
    probs /= probs.sum()

    # Per-camera average viewing direction (from arrow field, not raw average)
    cam_dirs = np.zeros((len(cams), 3), dtype=np.float64)
    for idx in range(len(cams)):
        if arrow_d[idx] is not None:
            cam_dirs[idx] = arrow_d[idx]
        elif visible_dirs[idx]:
            avg = np.mean(visible_dirs[idx], axis=0)
            n = np.linalg.norm(avg)
            if n > 1e-6:
                cam_dirs[idx] = avg / n

    # Predicted position (highest probability)
    best_idx = int(np.argmax(probs))
    pred_pos = cams[best_idx]

    # Predicted direction (FOV window at best position)
    pred_dir = None
    if arrow_d[best_idx] is not None:
        pred_dir = arrow_d[best_idx]
    else:
        dirs = visible_dirs[best_idx]
        if dirs:
            dirs_arr = np.array(dirs, dtype=np.float32)
            yaws = np.array([dir_to_yaw_pitch(d)[0] for d in dirs_arr])
            pits = np.array([dir_to_yaw_pitch(d)[1] for d in dirs_arr])
            sel, _ = best_fov_window(yaws, pits, hfov, vfov)
            pred_dir_vec = average_direction(dirs_arr, sel)
            if pred_dir_vec is not None:
                pred_dir = pred_dir_vec

    return cams, probs, obj_ids, pred_pos, pred_dir, cam_dirs


# ---------------------------------------------------------------------------
# Dialogue-based pose refinement (oracle mode)
# ---------------------------------------------------------------------------

def run_dialogue_refinement(
    cams: np.ndarray,
    probs: np.ndarray,
    cam_dirs: np.ndarray,
    pred_pos: np.ndarray,
    pred_dir: np.ndarray,
    gt_pos: np.ndarray,
    gt_dir: np.ndarray,
    dataset_root: Path,
    scan_id: str,
    max_rounds: int = 12,
    backend_name: str = "a1",
) -> Tuple[np.ndarray, np.ndarray, int, List[Tuple[str, str, str]]]:
    """Run oracle dialogue to refine the predicted pose.

    Uses the existing dialogue system in oracle mode: the GT frame's visible
    labels and spatial relations provide ground-truth answers to targeted
    yes/no questions, and the posterior over grid candidates is updated via
    Bayes' rule until convergence.

    Args:
        cams: Grid candidate positions, shape ``(N, 3)``.
        probs: Prior probability over candidates, shape ``(N,)``.
        cam_dirs: Per-candidate viewing directions, shape ``(N, 3)``.
        pred_pos: Initial predicted position from fine localization.
        pred_dir: Initial predicted direction from fine localization.
        gt_pos: Ground-truth position (for oracle answers).
        gt_dir: Ground-truth direction.
        dataset_root: Root of the dataset (e.g. ``/path/to/3RScan``).
        scan_id: Scene identifier.
        max_rounds: Maximum number of dialogue rounds.
        backend_name: Which backend to use (``"a1"``, ``"a2"``, or ``"a3"``).

    Returns:
        Tuple of ``(refined_pos, refined_dir, n_questions, dialogue_log)``
        where *dialogue_log* is a list of ``(question_text, answer, top_prob)``
        tuples tracking each round.
    """
    from langloc.dialogue.backends import CandidateBackendA1, FrameBackendA3, ParticleBackendA2
    from langloc.dialogue.candidates import extract_candidates
    from langloc.dialogue.dialogue_config import DialogueConfig
    from langloc.dialogue.dialogue_runner import nearest_frame_to_gt, oracle_answer
    from langloc.dialogue.frame_mapping import build_cand_to_frame_map, top_frames_by_mapping
    from langloc.dialogue.math_utils import _normalize, c2f_to_dense, pose_errors
    from langloc.dialogue.question_pool import Question, build_pools, compute_label_idf
    from langloc.dialogue.question_selection import pick_next_question_system
    from langloc.dialogue.scene_data import DEFAULT_ALIASES, load_scene_data
    from langloc.dialogue.semantics import frame_label_salience, frame_relations, rel_item_to_tuple, relation_phrase

    cfg = DialogueConfig(
        answer_mode="oracle",
        max_rounds=max_rounds,
        min_rounds=2,
        conf_threshold=0.85,
        auto_relax=True,
        candidate_set="grid",
        include_predicted_pose=True,
        pred_candidate_prior=0.15,
        k_nn=5,
        sigma=0.5,
        use_direction=True,
        dir_temp=0.25,
        max_pool_frames=30,
        question_strategy="ig",
        cache_answers=False,
        # Oracle mode: ask everything, no P(yes) threshold filtering
        ask_min_p=0.0,
        ask_max_p=1.0,
        # Only viewpoint-dependent spatial relations — size/height comparisons
        # (bigger_than, smaller_than, higher_than, lower_than) are true from
        # every viewpoint and provide zero positional information.
        allowed_rels=["left", "right", "front", "behind", "close_by"],
    )

    # Load scene data (frames with semantics)
    scene = load_scene_data(dataset_root, scan_id, dict(DEFAULT_ALIASES))
    frames_all = scene.frames

    # Build candidate set from grid points
    # Filter to candidates with nonzero probability
    valid_mask = probs > 0
    if valid_mask.sum() == 0:
        print("  [dialogue] No valid candidates, skipping.")
        return pred_pos, pred_dir, 0, [], probs.copy()

    cand_pos = cams[valid_mask].astype(np.float32)
    cand_prior = probs[valid_mask].astype(np.float64)
    cand_prior = cand_prior / max(float(cand_prior.sum()), 1e-12)

    # Candidate directions
    cd = cam_dirs[valid_mask].astype(np.float32)
    norms = np.linalg.norm(cd, axis=1, keepdims=True)
    has_dir = norms.squeeze() > 1e-6
    if has_dir.any():
        cd[has_dir] = cd[has_dir] / norms[has_dir]
        cand_dir = cd
    else:
        cand_dir = None

    # For A1/A2: inject frame positions as additional candidates so
    # there are always hypotheses near GT even when the heatmap missed it.
    if backend_name in ("a1", "a2"):
        frame_base_prior = 0.02  # 2% per frame → 20% total for 10 frames
        n_fr = len(scene.frame_pos)
        fr_pos = scene.frame_pos.astype(np.float32)
        cand_pos = np.concatenate([cand_pos, fr_pos], axis=0)
        fr_prior = np.full(n_fr, frame_base_prior, dtype=np.float64)
        total_fr_prior = frame_base_prior * n_fr
        cand_prior = np.concatenate([cand_prior * (1.0 - total_fr_prior), fr_prior])
        cand_prior = np.clip(cand_prior, 0, None)
        cand_prior = cand_prior / max(float(cand_prior.sum()), 1e-12)
        fr_dir = scene.frame_dir.astype(np.float32)
        fr_dir = fr_dir / np.maximum(np.linalg.norm(fr_dir, axis=1, keepdims=True), 1e-6)
        if cand_dir is not None:
            cand_dir = np.concatenate([cand_dir, fr_dir], axis=0)
        else:
            # Grid had no directions — start from frame directions
            cand_dir = np.concatenate([np.zeros((len(cand_pos) - n_fr, 3), dtype=np.float32), fr_dir], axis=0)
        print(f"  [dialogue] Injected {n_fr} frame-position candidates "
              f"(base prior={frame_base_prior:.2f} each, total={total_fr_prior:.1%})")

    # Optionally append predicted pose as extra candidate
    if cfg.include_predicted_pose and pred_pos is not None:
        extra_pos = pred_pos[None, :].astype(np.float32)
        cand_pos = np.concatenate([cand_pos, extra_pos], axis=0)
        extra_prior = np.array([cfg.pred_candidate_prior], dtype=np.float64)
        cand_prior = np.concatenate([cand_prior * (1.0 - cfg.pred_candidate_prior), extra_prior])
        cand_prior = cand_prior / max(float(cand_prior.sum()), 1e-12)
        if cand_dir is not None and pred_dir is not None:
            extra_dir = (pred_dir / max(float(np.linalg.norm(pred_dir)), 1e-6))[None, :].astype(np.float32)
            cand_dir = np.concatenate([cand_dir, extra_dir], axis=0)

    # Candidate-to-frame mapping
    c2f_map = build_cand_to_frame_map(
        cand_pos=cand_pos,
        cand_dir=cand_dir,
        frame_pos=scene.frame_pos,
        frame_dir=scene.frame_dir,
        k_nn=cfg.k_nn,
        sigma=cfg.sigma,
        use_direction=cfg.use_direction and (cand_dir is not None),
        dir_temp=cfg.dir_temp,
    )
    frame_subset = top_frames_by_mapping(c2f_map, max_frames=cfg.max_pool_frames)
    frames_pool = [frames_all[i] for i in frame_subset]

    # Dense mapping matrix
    W = c2f_to_dense(c2f_map, num_cands=len(cand_pos), num_frames=len(scene.frame_pos))
    W = W[:, frame_subset]

    pool_pos = np.asarray([scene.frame_pos[i] for i in frame_subset], dtype=np.float64)
    pool_dir = np.asarray([_normalize(scene.frame_dir[i]) for i in frame_subset], dtype=np.float64)

    pool_label_dicts = [frame_label_salience(fr) for fr in frames_pool]
    pool_rel_sets = [frame_relations(fr) for fr in frames_pool]

    label_pool, rel_pool = build_pools(
        frames_all=frames_all,
        frame_subset=frame_subset,
        max_rel_pool=cfg.max_rel_pool,
        rel_min_salience=cfg.rel_min_salience,
        rel_unique_only=cfg.rel_unique_only,
        allowed_rels=cfg.allowed_rels,
    )
    ignore = set(x.strip().lower() for x in cfg.ignore_labels)
    label_pool = [lab for lab in label_pool if lab not in ignore]

    idf = compute_label_idf(label_pool, pool_label_dicts)

    # Oracle GT frame
    gt_frame_idx = nearest_frame_to_gt(gt_pos, frames_all, scene)
    fr_gt = frames_all[gt_frame_idx]
    oracle_label_dict = frame_label_salience(fr_gt)
    oracle_rel_set = frame_relations(fr_gt)

    # Create backend
    if backend_name == "a1":
        backend = CandidateBackendA1(
            cand_pos=cand_pos,
            cand_dir=cand_dir,
            cand_prior=cand_prior,
            c2f_pool=W,
            frame_label_dicts=pool_label_dicts,
            frame_rel_sets=pool_rel_sets,
            frame_dirs=pool_dir,
            alpha_label=cfg.alpha_label,
            alpha_rel=cfg.alpha_rel,
            p_u_label=cfg.p_u_label,
            p_u_rel=cfg.p_u_rel,
            p_u_unanswerable=cfg.p_u_unanswerable,
            vis_tau=cfg.vis_tau,
            ans_tau=cfg.ans_tau,
        )
    elif backend_name == "a3":
        pf0 = (W.T @ cand_prior).reshape(-1)
        pf0 = pf0 / max(float(pf0.sum()), 1e-12)
        backend = FrameBackendA3(
            p0=pf0,
            frames_pool=frames_pool,
            frame_label_dicts=pool_label_dicts,
            frame_rel_sets=pool_rel_sets,
            frame_pos=pool_pos,
            frame_dir=pool_dir,
            alpha_label=cfg.alpha_label,
            alpha_rel=cfg.alpha_rel,
            p_u_label=cfg.p_u_label,
            p_u_rel=cfg.p_u_rel,
            p_u_unanswerable=cfg.p_u_unanswerable,
            vis_tau=cfg.vis_tau,
            ans_tau=cfg.ans_tau,
        )
    else:
        backend = ParticleBackendA2(
            cand_pos=cand_pos,
            cand_dir=cand_dir,
            cand_prior=cand_prior,
            frame_label_dicts=pool_label_dicts,
            frame_rel_sets=pool_rel_sets,
            frame_pos=pool_pos,
            frame_dir=pool_dir,
            n_particles=cfg.n_particles,
            k_nn=cfg.p_k_nn,
            sigma=cfg.p_sigma,
            jitter_pos=cfg.p_jitter,
            alpha_label=cfg.alpha_label,
            alpha_rel=cfg.alpha_rel,
            p_u_label=cfg.p_u_label,
            p_u_rel=cfg.p_u_rel,
            p_u_unanswerable=cfg.p_u_unanswerable,
            vis_tau=cfg.vis_tau,
            ans_tau=cfg.ans_tau,
            seed=cfg.seed,
        )

    # Run dialogue
    # A1/A2: skip relation questions — with only ~10 frames the
    # candidate-to-frame mapping makes relations non-discriminative
    # (e.g. "sink behind wall" is true from most viewpoints, boosting
    # all candidates equally and diffusing the posterior).
    if backend_name in ("a1", "a2"):
        questions = [Question("label", i) for i in range(len(label_pool))]
    else:
        questions = ([Question("rel", i) for i in range(len(rel_pool))]
                     + [Question("label", i) for i in range(len(label_pool))])

    dialogue_log: List[Tuple[str, str, str]] = []
    asked = 0

    print(f"  [dialogue] Starting oracle dialogue ({backend_name.upper()}, "
          f"max {max_rounds} rounds, {len(label_pool)} labels, {len(rel_pool)} relations)")

    for r in range(max_rounds):
        tp = backend.top_prob()
        if r + 1 >= cfg.min_rounds and tp >= cfg.conf_threshold:
            print(f"  [dialogue] Round {r+1}: confident (topP={tp:.3f} >= {cfg.conf_threshold})")
            break

        q = pick_next_question_system(backend_name, backend, questions,
                                       label_pool, rel_pool, idf, cfg)
        if q is None and cfg.auto_relax:
            old_min, old_max, old_ans = cfg.ask_min_p, cfg.ask_max_p, cfg.rel_min_answerable
            try:
                cfg.ask_min_p, cfg.ask_max_p = 0.01, 0.99
                q = pick_next_question_system(backend_name, backend, questions,
                                               label_pool, rel_pool, idf, cfg)
                if q is None:
                    cfg.rel_min_answerable = 0.0
                    q = pick_next_question_system(backend_name, backend, questions,
                                                   label_pool, rel_pool, idf, cfg)
            finally:
                cfg.ask_min_p, cfg.ask_max_p, cfg.rel_min_answerable = old_min, old_max, old_ans

        if q is None:
            print(f"  [dialogue] Round {r+1}: no more questions (topP={tp:.3f})")
            break

        # Render question text
        if q.qtype == "label":
            lab = label_pool[q.idx]
            q_text = f"Do you see a {lab}?"
        else:
            s, rel, o = rel_item_to_tuple(rel_pool[q.idx])
            q_text = f"Is {s} {relation_phrase(rel)} {o}?"

        # Oracle answer
        ans = oracle_answer(q, label_pool, rel_pool, oracle_label_dict, oracle_rel_set, cfg)

        print(f"  [dialogue] Round {r+1}: {q_text} -> {ans} (topP={tp:.3f})")
        dialogue_log.append((q_text, ans, f"{tp:.3f}"))

        if ans not in ("y", "n", "u"):
            continue

        asked += 1
        if q.qtype == "label":
            backend.update_label(label_pool[q.idx], ans)
        else:
            backend.update_rel(rel_item_to_tuple(rel_pool[q.idx]), ans)

        questions = [qq for qq in questions if not (qq.qtype == q.qtype and qq.idx == q.idx)]

    # Extract refined pose
    map_pos, map_dir, mean_pos, mean_dir = backend.predict_pose()
    # A3 snaps to best frame (MAP); A1/A2 use posterior-weighted mean
    # because the MAP stays stuck at the predicted-pose candidate while
    # the mean gets pulled toward GT as label evidence accumulates.
    if backend_name == "a3":
        refined_pos = map_pos
        refined_dir = map_dir
    else:
        refined_pos = mean_pos
        refined_dir = mean_dir

    # Compute errors for logging
    if gt_pos is not None:
        pre_pos_err, pre_rot_err = pose_errors(pred_pos, pred_dir, gt_pos, gt_dir)
        map_pos_err, map_rot_err = pose_errors(map_pos, map_dir, gt_pos, gt_dir)
        mean_pos_err, mean_rot_err = pose_errors(mean_pos, mean_dir, gt_pos, gt_dir)
        post_pos_err, post_rot_err = pose_errors(refined_pos, refined_dir, gt_pos, gt_dir)
        print(f"  [dialogue] Before: pos_err={pre_pos_err:.3f}m, rot_err={pre_rot_err:.1f}deg")
        print(f"  [dialogue] After (MAP):  pos_err={map_pos_err:.3f}m, rot_err={map_rot_err:.1f}deg")
        print(f"  [dialogue] After (mean): pos_err={mean_pos_err:.3f}m, rot_err={mean_rot_err:.1f}deg")
        using = "MAP" if backend_name == "a3" else "mean"
        print(f"  [dialogue] Using: {using} → pos_err={post_pos_err:.3f}m, rot_err={post_rot_err:.1f}deg")
        print(f"  [dialogue] Questions asked: {asked}")
        # Diagnostic: closest candidate to GT
        if hasattr(backend, 'cand_pos'):
            d_gt = np.linalg.norm(backend.cand_pos - gt_pos[None, :], axis=1)
            print(f"  [dialogue] Closest candidate to GT: {d_gt.min():.3f}m (idx={int(d_gt.argmin())})")

    # Return posterior mapped back to the original grid for heatmap rendering.
    # The first n_grid entries of the posterior correspond to cams[valid_mask].
    posterior = backend.posterior_vector().copy()
    n_grid = int(valid_mask.sum())
    grid_posterior = np.zeros(len(probs), dtype=np.float64)
    grid_posterior[valid_mask] = posterior[:n_grid]
    total = grid_posterior.sum()
    if total > 1e-12:
        grid_posterior = grid_posterior / total

    return refined_pos, refined_dir, asked, dialogue_log, grid_posterior


# ---------------------------------------------------------------------------
# Heatmap overlay on top-down view
# ---------------------------------------------------------------------------

def _set_eccv_rc():
    """Configure matplotlib for ECCV publication style (Computer Modern serif)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman",
                        "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
    })


def _add_title(img: np.ndarray, title: str, dpi: int = 300) -> np.ndarray:
    """Wrap an image in a matplotlib figure with an ECCV-style title."""
    _set_eccv_rc()
    H, W = img.shape[:2]
    fig_w, fig_h = W / dpi, H / dpi
    title_fs = max(10, W / dpi * 1.8)
    # Generous vertical padding for the title
    title_h_in = title_fs * 2.5 / 72  # points → inches
    total_h = fig_h + title_h_in
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, total_h), dpi=dpi)
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(title, fontsize=title_fs, pad=title_fs * 0.8)
    fig.subplots_adjust(left=0, right=1, top=1 - title_h_in / total_h, bottom=0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def _project_to_topdown(points_3d: np.ndarray,
                        intrinsic: np.ndarray,
                        extrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D world points to 2D top-down image coordinates.

    Args:
        points_3d: (N, 3) world coordinates.
        intrinsic: 3x3 camera intrinsic matrix.
        extrinsic: 4x4 world-to-camera extrinsic matrix.

    Returns:
        (px, py) pixel coordinate arrays of shape (N,).
    """
    pts_h = np.hstack([points_3d, np.ones((len(points_3d), 1))]).T  # (4, N)
    pts_cam = extrinsic @ pts_h  # (4, N)
    pts_2d = intrinsic @ pts_cam[:3, :]  # (3, N)
    px = pts_2d[0] / (pts_2d[2] + 1e-12)
    py = pts_2d[1] / (pts_2d[2] + 1e-12)
    return px, py


def overlay_heatmap(topdown_img: np.ndarray,
                    intrinsic: np.ndarray,
                    extrinsic: np.ndarray,
                    cams: np.ndarray,
                    probs: np.ndarray,
                    pred_pos: np.ndarray | None,
                    pred_dir: np.ndarray | None,
                    query: str,
                    alpha: float = 0.65,
                    cmap_name: str = "inferno",
                    sigma: float = 0.0,
                    h_fov_deg: float = 100.0,
                    up_axis: str = "z_up",
                    dpi: int = 300,
                    gt_pos: np.ndarray | None = None,
                    gt_dir: np.ndarray | None = None,
                    dialogue_pos: np.ndarray | None = None,
                    dialogue_dir: np.ndarray | None = None,
                    anchor_probs: np.ndarray | None = None) -> np.ndarray:
    """Composite a publication-quality probability heatmap onto the top-down image.

    Uses KDE-based density estimation over projected grid positions, weighted
    by visibility probability, with gaussian smoothing and perceptually-uniform
    colormaps. Alpha compositing produces a clean, ECCV/CVPR-ready overlay.

    Args:
        topdown_img: (H, W, 3) uint8 top-down render.
        intrinsic: 3x3 camera intrinsic matrix.
        extrinsic: 4x4 world-to-camera extrinsic matrix.
        cams: (N, 3) grid camera positions.
        probs: (N,) per-camera probability values.
        pred_pos: Predicted best camera position (3,) or None.
        pred_dir: Predicted heading direction (3,) or None.
        query: Natural-language query string (for annotation).
        alpha: Heatmap overlay opacity (0..1). Higher = more visible heatmap.
        cmap_name: Matplotlib colormap name (viridis, plasma, inferno, magma).
        sigma: Additional gaussian blur sigma in pixels. 0 = auto-compute from data.
        dpi: Output DPI.

    Returns:
        (H, W, 3) uint8 composited image.
    """
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    H, W = topdown_img.shape[:2]

    # Project 3D grid cameras to top-down image pixels
    px, py = _project_to_topdown(cams, intrinsic, extrinsic)

    # Filter to in-bounds points. When anchor_probs is provided, use its
    # nonzero mask to define the interpolation footprint (keeps the grid
    # extent even when dialogue pushes many probs toward zero).
    coverage = anchor_probs if anchor_probs is not None else probs
    mask = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (coverage > 0)
    px_v, py_v = px[mask].astype(np.float64), py[mask].astype(np.float64)
    probs_v = probs[mask].astype(np.float64)

    if len(probs_v) < 3:
        return topdown_img.copy()

    # ---- Direct interpolation of raw probabilities ----
    # Interpolate actual probability values onto a dense image grid
    # (not KDE density estimation, which spreads probability everywhere)
    grid_res = min(512, H, W)
    gx = np.linspace(0, W - 1, grid_res)
    gy = np.linspace(0, H - 1, grid_res)
    gx_mesh, gy_mesh = np.meshgrid(gx, gy)

    points = np.column_stack([px_v, py_v])
    prob_field = griddata(points, probs_v, (gx_mesh, gy_mesh),
                          method="cubic", fill_value=0.0)
    prob_field = np.clip(prob_field, 0, None)  # cubic can go negative

    # Light smoothing
    if sigma <= 0:
        sigma = max(grid_res * 0.01, 1.5)
    prob_field = gaussian_filter(prob_field, sigma=sigma)

    # Normalise to [0, 1] by max
    p_max = prob_field.max()
    if p_max > 1e-12:
        prob_field = prob_field / p_max
    else:
        prob_field = np.zeros_like(prob_field)

    # Upscale to full image resolution
    prob_pil = Image.fromarray((prob_field * 255).astype(np.uint8), mode="L")
    prob_full = np.asarray(
        prob_pil.resize((W, H), Image.LANCZOS), dtype=np.float64) / 255.0

    # Apply colourmap
    cmap = plt.get_cmap(cmap_name)
    heatmap_rgba = cmap(prob_full)  # (H, W, 4) float in [0, 1]
    heatmap_rgb = heatmap_rgba[:, :, :3]

    # Only overlay where probability is meaningful (> 2% of max).
    # When anchor_probs is given, compute coverage from the anchor grid
    # so regions with dialogue-prob ~0 still show up as dark (not hidden).
    if anchor_probs is not None:
        anchor_v = anchor_probs[mask].astype(np.float64)
        anc_field = griddata(points, anchor_v, (gx_mesh, gy_mesh),
                             method="cubic", fill_value=0.0)
        anc_field = np.clip(anc_field, 0, None)
        anc_full = np.asarray(
            Image.fromarray((np.clip(anc_field / max(anc_field.max(), 1e-12), 0, 1) * 255
                             ).astype(np.uint8), mode="L"
                            ).resize((W, H), Image.LANCZOS), dtype=np.float64) / 255.0
        prob_mask = anc_full > 0.02
    else:
        prob_mask = prob_full > 0.02
    mask_float = gaussian_filter(prob_mask.astype(np.float64), sigma=3.0)
    mask_float = np.clip(mask_float * alpha, 0, 1)

    # Alpha composite: blend heatmap over top-down image
    base = topdown_img.astype(np.float64) / 255.0
    blend = mask_float[..., None]
    composite = (1.0 - blend) * base + blend * heatmap_rgb
    composite = (np.clip(composite, 0, 1) * 255).astype(np.uint8)

    # ---- Overlay FOV wedge and markers with matplotlib ----
    _set_eccv_rc()
    title_fs = max(10, W / dpi * 1.8)
    title_h_in = title_fs * 2.5 / 72
    fig_w = W / dpi
    fig_h = H / dpi + title_h_in
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(composite)

    if pred_pos is not None:
        pred_px_arr, pred_py_arr = _project_to_topdown(
            pred_pos[None, :], intrinsic, extrinsic)
        pred_px, pred_py = float(pred_px_arr[0]), float(pred_py_arr[0])


        if -W * 0.1 <= pred_px < W * 1.1 and -H * 0.1 <= pred_py < H * 1.1:
            if pred_dir is not None and h_fov_deg > 0:
                # Draw a transparent FOV wedge
                # Compute the FOV reach in pixels by projecting points along
                # the FOV edges into the top-down image
                fov_reach_m = 2.5  # how far the FOV wedge extends in metres
                half_fov = math.radians(h_fov_deg / 2.0)

                # Get the heading angle on the floor plane
                up_idx = {"x_up": 0, "y_up": 1, "z_up": 2}.get(up_axis, 2)
                floor_axes = [i for i in range(3) if i != up_idx]
                a0, a1 = floor_axes
                heading = math.atan2(pred_dir[a1], pred_dir[a0])

                # Generate wedge boundary points in 3D
                n_arc = 30
                angles = np.linspace(heading - half_fov, heading + half_fov, n_arc)
                arc_pts_3d = np.zeros((n_arc, 3), dtype=np.float64)
                arc_pts_3d[:, up_idx] = pred_pos[up_idx]
                arc_pts_3d[:, a0] = pred_pos[a0] + fov_reach_m * np.cos(angles)
                arc_pts_3d[:, a1] = pred_pos[a1] + fov_reach_m * np.sin(angles)

                arc_px, arc_py = _project_to_topdown(arc_pts_3d, intrinsic, extrinsic)

                # Build wedge polygon: origin → arc → back to origin
                from matplotlib.patches import Polygon as MplPolygon
                wedge_xy = [(pred_px, pred_py)]
                for x, y in zip(arc_px, arc_py):
                    wedge_xy.append((float(x), float(y)))
                wedge_xy.append((pred_px, pred_py))

                wedge = MplPolygon(
                    wedge_xy, closed=True,
                    facecolor="#00e676", edgecolor="#00c853",
                    alpha=0.30, linewidth=1.2, zorder=4)
                ax.add_patch(wedge)

            # Predicted position dot
            ax.plot(pred_px, pred_py, marker="o", color="#00e676",
                    markersize=7, markeredgecolor="white",
                    markeredgewidth=1.2, zorder=5, label="Predicted")

    # Ground-truth overlay (red)
    if gt_pos is not None:
        gt_px_arr, gt_py_arr = _project_to_topdown(
            gt_pos[None, :], intrinsic, extrinsic)
        gt_px_v, gt_py_v = float(gt_px_arr[0]), float(gt_py_arr[0])

        if 0 <= gt_px_v < W and 0 <= gt_py_v < H:
            if gt_dir is not None and h_fov_deg > 0:
                from matplotlib.patches import Polygon as MplPolygon
                fov_reach_m = 2.5
                half_fov = math.radians(h_fov_deg / 2.0)
                up_idx_gt = {"x_up": 0, "y_up": 1, "z_up": 2}.get(up_axis, 2)
                floor_axes_gt = [i for i in range(3) if i != up_idx_gt]
                a0g, a1g = floor_axes_gt
                heading_gt = math.atan2(gt_dir[a1g], gt_dir[a0g])
                n_arc = 30
                angles_gt = np.linspace(heading_gt - half_fov, heading_gt + half_fov, n_arc)
                arc_pts = np.zeros((n_arc, 3), dtype=np.float64)
                arc_pts[:, up_idx_gt] = gt_pos[up_idx_gt]
                arc_pts[:, a0g] = gt_pos[a0g] + fov_reach_m * np.cos(angles_gt)
                arc_pts[:, a1g] = gt_pos[a1g] + fov_reach_m * np.sin(angles_gt)
                arc_px_gt, arc_py_gt = _project_to_topdown(arc_pts, intrinsic, extrinsic)
                wedge_xy_gt = [(gt_px_v, gt_py_v)]
                for wx, wy in zip(arc_px_gt, arc_py_gt):
                    wedge_xy_gt.append((float(wx), float(wy)))
                wedge_xy_gt.append((gt_px_v, gt_py_v))
                wedge_gt = MplPolygon(
                    wedge_xy_gt, closed=True,
                    facecolor="#ef5350", edgecolor="#c62828",
                    alpha=0.30, linewidth=1.2, zorder=4)
                ax.add_patch(wedge_gt)

            ax.plot(gt_px_v, gt_py_v, marker="o", color="#ef5350",
                    markersize=7, markeredgecolor="white",
                    markeredgewidth=1.2, zorder=5, label="Ground Truth")

    # Dialogue-refined pose overlay (blue)
    if dialogue_pos is not None:
        dlg_px_arr, dlg_py_arr = _project_to_topdown(
            dialogue_pos[None, :], intrinsic, extrinsic)
        dlg_px, dlg_py = float(dlg_px_arr[0]), float(dlg_py_arr[0])

        if -W * 0.1 <= dlg_px < W * 1.1 and -H * 0.1 <= dlg_py < H * 1.1:
            if dialogue_dir is not None and h_fov_deg > 0:
                from matplotlib.patches import Polygon as MplPolygon
                fov_reach_m = 2.5
                half_fov = math.radians(h_fov_deg / 2.0)
                up_idx_d = {"x_up": 0, "y_up": 1, "z_up": 2}.get(up_axis, 2)
                floor_axes_d = [i for i in range(3) if i != up_idx_d]
                a0d, a1d = floor_axes_d
                heading_d = math.atan2(dialogue_dir[a1d], dialogue_dir[a0d])
                n_arc = 30
                angles_d = np.linspace(heading_d - half_fov, heading_d + half_fov, n_arc)
                arc_pts_d = np.zeros((n_arc, 3), dtype=np.float64)
                arc_pts_d[:, up_idx_d] = dialogue_pos[up_idx_d]
                arc_pts_d[:, a0d] = dialogue_pos[a0d] + fov_reach_m * np.cos(angles_d)
                arc_pts_d[:, a1d] = dialogue_pos[a1d] + fov_reach_m * np.sin(angles_d)
                arc_px_d, arc_py_d = _project_to_topdown(arc_pts_d, intrinsic, extrinsic)
                wedge_xy_d = [(dlg_px, dlg_py)]
                for wx, wy in zip(arc_px_d, arc_py_d):
                    wedge_xy_d.append((float(wx), float(wy)))
                wedge_xy_d.append((dlg_px, dlg_py))
                wedge_d = MplPolygon(
                    wedge_xy_d, closed=True,
                    facecolor="#42a5f5", edgecolor="#1565c0",
                    alpha=0.30, linewidth=1.2, zorder=4)
                ax.add_patch(wedge_d)

            ax.plot(dlg_px, dlg_py, marker="D", color="#42a5f5",
                    markersize=7, markeredgecolor="white",
                    markeredgewidth=1.2, zorder=6, label="Dialogue-Refined")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, aspect=30)
    cbar.set_label("Probability", fontsize=max(8, W / dpi * 1.2))
    cbar.ax.tick_params(labelsize=max(6, W / dpi))

    # Legend (only when there are multiple markers to distinguish)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 1:
        leg = ax.legend(loc="lower left", fontsize=max(6, W / dpi * 0.9),
                        framealpha=0.85, edgecolor="#888888",
                        handletextpad=0.3, borderpad=0.3)
        leg.set_zorder(10)

    # Title
    ax.set_title("Localization Probability Heatmap",
                 fontsize=title_fs, pad=title_fs * 0.8)

    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.subplots_adjust(left=0, right=0.93, top=1 - title_h_in / fig_h, bottom=0)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)

    if buf.shape[:2] != (H, W):
        buf = np.asarray(Image.fromarray(buf).resize((W, H), Image.LANCZOS))

    return buf


def overlay_poses_only(topdown_img: np.ndarray,
                       intrinsic: np.ndarray,
                       extrinsic: np.ndarray,
                       pred_pos: np.ndarray | None,
                       pred_dir: np.ndarray | None,
                       gt_pos: np.ndarray | None,
                       gt_dir: np.ndarray | None,
                       h_fov_deg: float = 100.0,
                       up_axis: str = "z_up",
                       dpi: int = 300) -> np.ndarray:
    """Draw only GT and predicted pose markers on the top-down view (no heatmap)."""
    H, W = topdown_img.shape[:2]
    _set_eccv_rc()
    title_fs = max(10, W / dpi * 1.8)
    title_h_in = title_fs * 2.5 / 72
    fig_w = W / dpi
    fig_h = H / dpi + title_h_in
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(topdown_img)

    up_idx = {"x_up": 0, "y_up": 1, "z_up": 2}.get(up_axis, 2)
    floor_axes = [i for i in range(3) if i != up_idx]
    a0, a1 = floor_axes

    def _draw_pose(pos, direction, color, edge_color, marker, label, zorder):
        px_arr, py_arr = _project_to_topdown(pos[None, :], intrinsic, extrinsic)
        px_v, py_v = float(px_arr[0]), float(py_arr[0])
        if not (-W * 0.1 <= px_v < W * 1.1 and -H * 0.1 <= py_v < H * 1.1):
            return
        if direction is not None and h_fov_deg > 0:
            from matplotlib.patches import Polygon as MplPolygon
            fov_reach_m = 2.5
            half_fov = math.radians(h_fov_deg / 2.0)
            heading = math.atan2(direction[a1], direction[a0])
            n_arc = 30
            angles = np.linspace(heading - half_fov, heading + half_fov, n_arc)
            arc_pts = np.zeros((n_arc, 3), dtype=np.float64)
            arc_pts[:, up_idx] = pos[up_idx]
            arc_pts[:, a0] = pos[a0] + fov_reach_m * np.cos(angles)
            arc_pts[:, a1] = pos[a1] + fov_reach_m * np.sin(angles)
            arc_px, arc_py = _project_to_topdown(arc_pts, intrinsic, extrinsic)
            wedge_xy = [(px_v, py_v)]
            for wx, wy in zip(arc_px, arc_py):
                wedge_xy.append((float(wx), float(wy)))
            wedge_xy.append((px_v, py_v))
            wedge = MplPolygon(wedge_xy, closed=True,
                               facecolor=color, edgecolor=edge_color,
                               alpha=0.30, linewidth=1.2, zorder=zorder)
            ax.add_patch(wedge)
        ax.plot(px_v, py_v, marker=marker, color=color,
                markersize=7, markeredgecolor="white",
                markeredgewidth=1.2, zorder=zorder + 1, label=label)

    if pred_pos is not None:
        _draw_pose(pred_pos, pred_dir, "#00e676", "#00c853", "o", "Predicted", 4)
    if gt_pos is not None:
        _draw_pose(gt_pos, gt_dir, "#ef5350", "#c62828", "o", "Ground Truth", 6)

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 1:
        leg = ax.legend(loc="lower left", fontsize=max(6, W / dpi * 0.9),
                        framealpha=0.85, edgecolor="#888888",
                        handletextpad=0.3, borderpad=0.3)
        leg.set_zorder(10)

    ax.set_title("GT vs Predicted Pose",
                 fontsize=title_fs, pad=title_fs * 0.8)
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.subplots_adjust(left=0, right=1, top=1 - title_h_in / fig_h, bottom=0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    if buf.shape[:2] != (H, W):
        buf = np.asarray(Image.fromarray(buf).resize((W, H), Image.LANCZOS))
    return buf


# ---------------------------------------------------------------------------
# Direction field overlay on top-down view
# ---------------------------------------------------------------------------

def overlay_direction_field(topdown_img: np.ndarray,
                            intrinsic: np.ndarray,
                            extrinsic: np.ndarray,
                            cams: np.ndarray,
                            probs: np.ndarray,
                            cam_dirs: np.ndarray,
                            pred_pos: np.ndarray | None = None,
                            pred_dir: np.ndarray | None = None,
                            h_fov_deg: float = 100.0,
                            up_axis: str = "z_up",
                            stride: int = 1,
                            dpi: int = 300,
                            gt_pos: np.ndarray | None = None,
                            gt_dir: np.ndarray | None = None) -> np.ndarray:
    """Overlay a direction field on the top-down image.

    Each grid camera with non-zero probability is drawn as a short oriented
    line segment. Color encodes the heading angle (HSV colour wheel) and
    opacity/thickness encodes the probability.

    Args:
        topdown_img: (H, W, 3) uint8 top-down render.
        intrinsic: 3x3 camera intrinsic matrix.
        extrinsic: 4x4 world-to-camera extrinsic matrix.
        cams: (N, 3) grid camera positions.
        probs: (N,) per-camera probability.
        cam_dirs: (N, 3) per-camera average viewing direction.
        up_axis: Vertical axis convention.
        stride: Draw every Nth camera (1 = all).
        dpi: Output DPI.

    Returns:
        (H, W, 3) uint8 composited image.
    """
    import matplotlib.colors as mcolors

    H, W = topdown_img.shape[:2]
    up_idx = {"x_up": 0, "y_up": 1, "z_up": 2}.get(up_axis, 2)
    floor_axes = [i for i in range(3) if i != up_idx]
    a0, a1 = floor_axes

    # Project cameras to 2D
    px, py = _project_to_topdown(cams, intrinsic, extrinsic)

    # Filter: in-bounds, non-zero prob, non-zero direction
    dir_norms = np.linalg.norm(cam_dirs, axis=1)
    mask = ((px >= 0) & (px < W) & (py >= 0) & (py < H)
            & (probs > 0) & (dir_norms > 1e-6))
    indices = np.where(mask)[0]
    if stride > 1:
        indices = indices[::stride]

    if len(indices) == 0:
        return topdown_img.copy()

    # Compute segment length in pixels (proportional to grid spacing)
    # Use median distance between neighbouring projected points
    all_px, all_py = px[mask], py[mask]
    if len(all_px) > 1:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([all_px, all_py]))
        dists, _ = tree.query(np.column_stack([all_px, all_py]), k=2)
        seg_len = float(np.median(dists[:, 1])) * 0.45
    else:
        seg_len = 10.0
    seg_len = max(seg_len, 3.0)

    # Build line segments and colors
    from matplotlib.collections import LineCollection
    segments = []
    prob_norms = []
    p_max = probs[indices].max()
    cmap = plt.get_cmap("inferno")

    for idx in indices:
        cx, cy = float(px[idx]), float(py[idx])
        # Project the direction tip to get 2D direction
        tip_3d = cams[idx] + cam_dirs[idx] * 0.5
        tip_px, tip_py = _project_to_topdown(tip_3d[None, :], intrinsic, extrinsic)
        dx_2d = float(tip_px[0]) - cx
        dy_2d = float(tip_py[0]) - cy
        dn = math.sqrt(dx_2d**2 + dy_2d**2)
        if dn < 1e-3:
            continue
        dx_2d, dy_2d = dx_2d / dn, dy_2d / dn

        prob_norm = probs[idx] / p_max if p_max > 0 else 0

        # Line segment centred on camera position
        half = seg_len * 0.5
        x0, y0 = cx - half * dx_2d, cy - half * dy_2d
        x1, y1 = cx + half * dx_2d, cy + half * dy_2d
        segments.append([(x0, y0), (x1, y1)])
        prob_norms.append(prob_norm)

    # Color by probability: inferno colormap (dark = low, yellow = high)
    colors = [cmap(p) for p in prob_norms]
    linewidths = [1.0] * len(prob_norms)

    if not segments:
        return topdown_img.copy()

    # Render with matplotlib
    _set_eccv_rc()
    title_fs = max(10, min(W, H) / dpi * 1.8)
    title_h_in = title_fs * 2.5 / 72
    fig_w = W / dpi
    fig_h = H / dpi + title_h_in
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(topdown_img)

    lc = LineCollection(segments, colors=colors, linewidths=linewidths,
                        capstyle="round", zorder=3)
    ax.add_collection(lc)

    # --- FOV wedge + predicted pose (grey, low opacity) ---
    if pred_pos is not None:
        pred_px_arr, pred_py_arr = _project_to_topdown(
            pred_pos[None, :], intrinsic, extrinsic)
        pred_px, pred_py = float(pred_px_arr[0]), float(pred_py_arr[0])


        if -W * 0.1 <= pred_px < W * 1.1 and -H * 0.1 <= pred_py < H * 1.1:
            if pred_dir is not None and h_fov_deg > 0:
                from matplotlib.patches import Polygon as MplPolygon
                fov_reach_m = 2.5
                half_fov = math.radians(h_fov_deg / 2.0)
                heading = math.atan2(pred_dir[a1], pred_dir[a0])
                n_arc = 30
                angles = np.linspace(heading - half_fov, heading + half_fov, n_arc)
                arc_pts_3d = np.zeros((n_arc, 3), dtype=np.float64)
                arc_pts_3d[:, up_idx] = pred_pos[up_idx]
                arc_pts_3d[:, a0] = pred_pos[a0] + fov_reach_m * np.cos(angles)
                arc_pts_3d[:, a1] = pred_pos[a1] + fov_reach_m * np.sin(angles)
                arc_px, arc_py = _project_to_topdown(arc_pts_3d, intrinsic, extrinsic)

                wedge_xy = [(pred_px, pred_py)]
                for wx, wy in zip(arc_px, arc_py):
                    wedge_xy.append((float(wx), float(wy)))
                wedge_xy.append((pred_px, pred_py))

                wedge = MplPolygon(
                    wedge_xy, closed=True,
                    facecolor="#9e9e9e", edgecolor="#757575",
                    alpha=0.40, linewidth=1.0, zorder=4)
                ax.add_patch(wedge)

            # Predicted position dot
            ax.plot(pred_px, pred_py, marker="o", color="#616161",
                    markersize=6, markeredgecolor="white",
                    markeredgewidth=1.0, zorder=5)

    # --- Ground-truth overlay (red, low opacity) ---
    if gt_pos is not None:
        gt_px_arr, gt_py_arr = _project_to_topdown(
            gt_pos[None, :], intrinsic, extrinsic)
        gt_px_v, gt_py_v = float(gt_px_arr[0]), float(gt_py_arr[0])

        if 0 <= gt_px_v < W and 0 <= gt_py_v < H:
            if gt_dir is not None and h_fov_deg > 0:
                from matplotlib.patches import Polygon as MplPolygon
                fov_reach_m = 2.5
                half_fov = math.radians(h_fov_deg / 2.0)
                heading_gt = math.atan2(gt_dir[a1], gt_dir[a0])
                n_arc = 30
                angles_gt = np.linspace(heading_gt - half_fov, heading_gt + half_fov, n_arc)
                arc_pts = np.zeros((n_arc, 3), dtype=np.float64)
                arc_pts[:, up_idx] = gt_pos[up_idx]
                arc_pts[:, a0] = gt_pos[a0] + fov_reach_m * np.cos(angles_gt)
                arc_pts[:, a1] = gt_pos[a1] + fov_reach_m * np.sin(angles_gt)
                arc_px_gt, arc_py_gt = _project_to_topdown(arc_pts, intrinsic, extrinsic)
                wedge_xy_gt = [(gt_px_v, gt_py_v)]
                for wx, wy in zip(arc_px_gt, arc_py_gt):
                    wedge_xy_gt.append((float(wx), float(wy)))
                wedge_xy_gt.append((gt_px_v, gt_py_v))
                wedge_gt = MplPolygon(
                    wedge_xy_gt, closed=True,
                    facecolor="#ef5350", edgecolor="#c62828",
                    alpha=0.30, linewidth=1.0, zorder=4)
                ax.add_patch(wedge_gt)

            ax.plot(gt_px_v, gt_py_v, marker="o", color="#ef5350",
                    markersize=6, markeredgecolor="white",
                    markeredgewidth=1.0, zorder=5)

    # Colorbar for probability
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, aspect=30)
    cbar.set_label("Probability", fontsize=max(8, W / dpi * 1.2))
    cbar.ax.tick_params(labelsize=max(6, W / dpi))

    # Title
    ax.set_title("Predicted Viewing Direction Field",
                 fontsize=title_fs, pad=title_fs * 0.8)

    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.subplots_adjust(left=0, right=0.93, top=1 - title_h_in / fig_h, bottom=0)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)

    if buf.shape[:2] != (H, W):
        buf = np.asarray(Image.fromarray(buf).resize((W, H), Image.LANCZOS))

    return buf


# ---------------------------------------------------------------------------
# Scene graph visualization (Open3D OffscreenRenderer)
# ---------------------------------------------------------------------------

_SPATIAL_RELATIONS = {"left", "right", "front", "behind"}


def _avg_vertex_color(mesh: o3d.geometry.TriangleMesh,
                      obj2faces: Dict[int, np.ndarray],
                      oid: int) -> np.ndarray:
    """Compute average vertex colour (float64 RGB in [0,1]) for an object."""
    faces = obj2faces.get(oid)
    if faces is None or len(faces) == 0:
        return np.array([0.6, 0.6, 0.6], dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    vert_ids = np.unique(tris[faces].ravel())
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
        return np.clip(colors[vert_ids].mean(axis=0), 0, 1)
    return np.array([0.6, 0.6, 0.6], dtype=np.float64)


def _make_sphere(center: np.ndarray, radius: float,
                 color: np.ndarray, resolution: int = 20) -> o3d.geometry.TriangleMesh:
    """Create a colored sphere mesh at the given center."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    sphere.translate(center)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    return sphere


def _make_cylinder_between(start: np.ndarray, end: np.ndarray,
                           radius: float,
                           color: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Create a cylinder mesh connecting two 3D points."""
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return o3d.geometry.TriangleMesh()
    direction = vec / length

    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length,
                                                     resolution=12, split=1)
    cyl.compute_vertex_normals()
    cyl.paint_uniform_color(color)

    # Open3D cylinders are along +Z centered at origin; align to direction
    mid = (start + end) / 2.0
    z_axis = np.array([0.0, 0.0, 1.0])
    rot_axis = np.cross(z_axis, direction)
    rot_norm = np.linalg.norm(rot_axis)
    if rot_norm > 1e-6:
        rot_axis /= rot_norm
        angle = math.acos(np.clip(np.dot(z_axis, direction), -1, 1))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)
        cyl.rotate(R, center=np.zeros(3))
    elif np.dot(z_axis, direction) < 0:
        # 180° flip
        cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([1.0, 0.0, 0.0]) * math.pi), center=np.zeros(3))

    cyl.translate(mid)
    return cyl


def render_scene_graph(mesh: o3d.geometry.TriangleMesh,
                       graphs_3dssg: dict,
                       scan_id: str,
                       matched_obj_ids: List[int],
                       obj2faces: Dict[int, np.ndarray],
                       up_axis: str,
                       width: int = 1920,
                       height: int = 1080,
                       camera_json: str | None = None,
                       interactive: bool = False,
                       desat: float = 0.5,
                       gt_pos: np.ndarray | None = None,
                       gt_dir: np.ndarray | None = None,
                       gt_pullback: float = 1.0,
                       labels_ply: str | None = None) -> np.ndarray:
    """Render a scene graph overlay using Open3D OffscreenRenderer.

    Draws the mesh with matched objects highlighted, places instance-colored
    spheres at matched object centroids, and draws cylinder edges for
    spatial relations between matched objects.

    Args:
        mesh: The scene mesh.
        graphs_3dssg: Pre-loaded 3DSSG graphs dict (from torch.load).
        scan_id: Scene identifier.
        matched_obj_ids: Object IDs matched by the localization pipeline.
        obj2faces: Mapping from object ID to face indices.
        up_axis: Vertical axis convention.
        width: Render width in pixels.
        height: Render height in pixels.
        camera_json: Optional JSON file with {eye, center, up}.
        interactive: If True, open viewer to pick camera before rendering.
        desat: Desaturation of non-matched regions (0=original, 1=grey).

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    # Look up scene in pre-loaded graph data
    g3d_all = graphs_3dssg
    g = None
    for key in [scan_id, f"3RScan/{scan_id}", scan_id.replace("3RScan/", "")]:
        if key in g3d_all:
            g = g3d_all[key]
            break
    if g is None:
        print(f"  Warning: scan_id '{scan_id}' not found in 3DSSG — skipping scene graph.")
        return render_perspective(mesh, width, height, up_axis)

    objects = g.get("objects", {})
    matched_set = set(matched_obj_ids)

    # Compute centroids from OBBs
    obj_centroids: Dict[int, np.ndarray] = {}
    for oid_str, obj in objects.items():
        oid = int(oid_str) if isinstance(oid_str, str) else oid_str
        obb = obj.get("obb")
        if obb is not None:
            obj_centroids[oid] = np.array(obb["centroid"], dtype=np.float64)

    # --- Build the scene ---
    r = rendering.OffscreenRenderer(width, height)
    scene = r.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])

    # Color matched objects with semantic instance colours on the mesh
    # Build tri2obj from obj2faces so color_matched_objects can paint per-face
    n_tris = len(np.asarray(mesh.triangles))
    tri2obj_sg = np.zeros(n_tris, dtype=np.int32)
    for oid, faces in obj2faces.items():
        if faces is not None and len(faces) > 0:
            tri2obj_sg[faces] = int(oid)

    # Load semantic colors: prefer labels PLY, fall back to mesh vertex colors
    sem_colors = None
    if labels_ply and Path(labels_ply).exists():
        lm = o3d.io.read_triangle_mesh(labels_ply)
        if lm.has_vertex_colors():
            sem_colors = np.asarray(lm.vertex_colors, dtype=np.float64)
    if sem_colors is None and mesh.has_vertex_colors():
        sem_colors = np.asarray(mesh.vertex_colors, dtype=np.float64).copy()

    if sem_colors is not None and matched_obj_ids:
        render_mesh = color_matched_objects(mesh, tri2obj_sg,
                                             matched_obj_ids, sem_colors,
                                             desat=desat)
    else:
        render_mesh = mesh

    mesh_mat = rendering.MaterialRecord()
    mesh_mat.shader = "defaultLitTransparency"
    mesh_mat.base_color = [1.0, 1.0, 1.0, 0.85]
    scene.add_geometry("mesh", render_mesh, mesh_mat)
    scene.view.set_post_processing(True)

    def _obj_label_color(oid: int) -> np.ndarray:
        """Get the dominant label color for an object from the labels PLY."""
        if sem_colors is None:
            return _avg_vertex_color(mesh, obj2faces, oid)
        faces = obj2faces.get(oid)
        if faces is None or len(faces) == 0:
            return np.array([0.6, 0.6, 0.6], dtype=np.float64)
        tris = np.asarray(mesh.triangles, dtype=np.int32)
        vert_ids = np.unique(tris[faces].ravel())
        # Use the most frequent color (mode) rather than average
        vc = sem_colors[vert_ids]
        # Quantize to avoid float noise, then find mode
        quantized = (vc * 255).astype(np.uint8)
        keys = quantized[:, 0].astype(np.uint32) << 16 | quantized[:, 1].astype(np.uint32) << 8 | quantized[:, 2].astype(np.uint32)
        values, counts = np.unique(keys, return_counts=True)
        best = values[counts.argmax()]
        return np.array([(best >> 16) & 0xFF, (best >> 8) & 0xFF, best & 0xFF],
                        dtype=np.float64) / 255.0

    # Sphere markers at matched object centroids — vivid opaque colors
    sphere_radius = 0.07

    def _intensify_color(rgb: np.ndarray) -> np.ndarray:
        """Push colour to maximum saturation and full brightness."""
        import colorsys
        h, s, v = colorsys.rgb_to_hsv(float(rgb[0]), float(rgb[1]), float(rgb[2]))
        s = 1.0   # full saturation
        v = 1.0   # full brightness
        return np.array(colorsys.hsv_to_rgb(h, s, v), dtype=np.float64)

    for i, oid in enumerate(matched_obj_ids):
        if oid not in obj_centroids:
            continue
        base_color = _obj_label_color(oid)
        intense = _intensify_color(base_color)
        sphere = _make_sphere(obj_centroids[oid], sphere_radius, intense)
        sph_mat = rendering.MaterialRecord()
        sph_mat.shader = "defaultLit"
        sph_mat.base_color = [1.0, 1.0, 1.0, 1.0]
        scene.add_geometry(f"sphere_{i}", sphere, sph_mat)

    # Cylinder edges for spatial relations between matched objects
    edge_lists = g.get("edge_lists", {})
    from_ids = edge_lists.get("from", [])
    to_ids = edge_lists.get("to", [])
    relations = edge_lists.get("relation", [])

    edge_color = np.array([0.15, 0.15, 0.15])
    cyl_mat = rendering.MaterialRecord()
    cyl_mat.shader = "defaultLit"
    seen_pairs: set = set()
    edge_i = 0
    for fi, ti, rel in zip(from_ids, to_ids, relations):
        fi, ti = int(fi), int(ti)
        if rel not in _SPATIAL_RELATIONS:
            continue
        pair_key = frozenset({fi, ti})
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        if fi not in obj_centroids or ti not in obj_centroids:
            continue
        # Only draw edges between nodes that both have spheres
        if fi not in matched_set or ti not in matched_set:
            continue
        cyl = _make_cylinder_between(obj_centroids[fi], obj_centroids[ti],
                                      radius=0.012, color=edge_color)
        if len(cyl.vertices) > 0:
            scene.add_geometry(f"edge_{edge_i}", cyl, cyl_mat)
            edge_i += 1

    # Lighting
    scene.set_lighting(
        rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0.0, 0.0, 0.0))
    scene.scene.enable_sun_light(True)
    scene.scene.set_sun_light(
        direction=np.array([0.3, -1.0, -1.0], dtype=np.float32),
        color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        intensity=120000.0)

    # Camera — use GT pose pulled back if available, else fallback
    if gt_pos is not None and gt_dir is not None:
        # Place camera behind GT position (pull back along negative view dir)
        up_v = _up_vector(up_axis)
        eye = gt_pos - gt_dir * gt_pullback
        centre = gt_pos + gt_dir  # look past GT position along view dir
        r.setup_camera(60.0, centre, eye, up_v)
    elif camera_json and Path(camera_json).exists():
        cam = json.loads(Path(camera_json).read_text())
        eye = np.array(cam["eye"], dtype=np.float64)
        centre = np.array(cam["center"], dtype=np.float64)
        up_v = np.array(cam["up"], dtype=np.float64)
        r.setup_camera(60.0, centre, eye, up_v)
    elif interactive:
        pinhole = pick_camera_interactive(render_mesh, up_axis)
        if pinhole is not None:
            intr = pinhole.intrinsic
            ext = np.asarray(pinhole.extrinsic, dtype=np.float64)
            sx, sy = width / intr.width, height / intr.height
            new_intr = o3d.camera.PinholeCameraIntrinsic(
                width, height,
                intr.get_focal_length()[0] * sx, intr.get_focal_length()[1] * sy,
                intr.get_principal_point()[0] * sx, intr.get_principal_point()[1] * sy)
            r.setup_camera(new_intr, ext)
        else:
            eye, centre, up_v = _auto_camera(render_mesh, up_axis)
            r.setup_camera(60.0, centre, eye, up_v)
    else:
        eye, centre, up_v = _auto_camera(render_mesh, up_axis)
        r.setup_camera(60.0, centre, eye, up_v)

    img = r.render_to_image()
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # --- Overlay text labels using matplotlib ---
    result = arr.astype(np.uint8)
    if matched_obj_ids:
        _set_eccv_rc()
        H, W = result.shape[:2]
        dpi = 150
        fig, ax = plt.subplots(1, 1, figsize=(W / dpi, H / dpi), dpi=dpi)
        ax.imshow(result)
        ax.set_axis_off()
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

        # Get view/projection matrices from the renderer's camera
        view_mat = np.asarray(scene.camera.get_view_matrix())
        proj_mat = np.asarray(scene.camera.get_projection_matrix())

        for oid in matched_obj_ids:
            if oid not in obj_centroids:
                continue
            oid_key = str(oid) if str(oid) in objects else oid
            obj = objects.get(oid_key)
            label = obj.get("label", f"obj_{oid}") if obj else f"obj_{oid}"
            color = _obj_label_color(oid)

            # Project centroid: world → clip → NDC → pixel
            pt = np.append(obj_centroids[oid], 1.0)
            cam_pt = view_mat @ pt
            clip_pt = proj_mat @ cam_pt
            if abs(clip_pt[3]) < 1e-8:
                continue
            ndc = clip_pt[:3] / clip_pt[3]
            px = (ndc[0] + 1.0) * 0.5 * W
            py = (1.0 - ndc[1]) * 0.5 * H
            if not (0 <= px < W and 0 <= py < H):
                continue
            # Place label above sphere
            label_y = py - sphere_radius * H * 0.04
            ax.text(px, label_y, label, ha="center", va="bottom",
                    fontsize=max(7, W / dpi * 0.7), fontweight="bold",
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              alpha=0.75, ec="none"))

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        result = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        if result.shape[:2] != (H, W):
            result = np.asarray(Image.fromarray(result).resize((W, H), Image.LANCZOS))

    return result


# ---------------------------------------------------------------------------
# Scene retrieval — find other scenes matching query objects + relations
# ---------------------------------------------------------------------------

_GENERIC_LABELS = frozenset({
    "wall", "floor", "ceiling", "door", "doorframe", "window", "object"})
_SPATIAL_RELS = frozenset({"left", "right", "front", "behind"})


def find_matching_scenes(
    sg_all: dict,
    source_scan_id: str,
    query_labels: set[str],
    query_relations: set[tuple[str, str, str]],
    top_k: int = 4,
) -> List[dict]:
    """Find scenes whose objects and spatial relations best match a query.

    Args:
        sg_all: All scene graphs (scan_id → graph dict).
        source_scan_id: Scene to exclude (the query's own scene).
        query_labels: Set of non-generic object labels from the query frame.
        query_relations: Set of (subject_label, object_label, relation) tuples.
        top_k: Number of results to return.

    Returns:
        List of dicts sorted by score, each with keys:
        scan_id, score, matched_labels, matched_relations.
    """
    results = []
    for scan_id, g in sg_all.items():
        if scan_id == source_scan_id:
            continue
        objects = g.get("objects", {})
        edge_lists = g.get("edge_lists", {})

        # Scene label set (non-generic)
        scene_labels: set[str] = set()
        for obj in objects.values():
            l = obj["label"]
            if l not in _GENERIC_LABELS:
                scene_labels.add(l)

        shared_labels = query_labels & scene_labels
        if len(shared_labels) < 2:
            continue

        # Scene spatial edge set
        from_ids = edge_lists.get("from", [])
        to_ids = edge_lists.get("to", [])
        relations = edge_lists.get("relation", [])
        scene_edges: set[tuple[str, str, str]] = set()
        for fi, ti, rel in zip(from_ids, to_ids, relations):
            if rel not in _SPATIAL_RELS:
                continue
            fl = objects.get(str(int(fi)), {}).get("label", "")
            tl = objects.get(str(int(ti)), {}).get("label", "")
            scene_edges.add((fl, tl, rel))

        shared_rels = query_relations & scene_edges
        obj_score = len(shared_labels) / max(len(query_labels), 1)
        rel_score = len(shared_rels) / max(len(query_relations), 1)
        total = obj_score * 0.5 + rel_score * 0.5

        results.append({
            "scan_id": scan_id,
            "score": total,
            "matched_labels": shared_labels,
            "matched_relations": shared_rels,
        })

    results.sort(key=lambda x: -x["score"])
    return results[:top_k]


def find_best_frame(scan_dir: Path, query_labels: set[str]) -> dict | None:
    """Find the frame in a scene that shows the most query objects.

    Args:
        scan_dir: Path to the scene directory (e.g. data/scans/scene0006_01).
        query_labels: Non-generic object labels to look for.

    Returns:
        The frame dict with the best coverage, or None if no descriptions exist.
    """
    desc_dir = scan_dir / "output" / "descriptions"
    if not desc_dir.exists():
        return None

    best_frame = None
    best_coverage = -1

    for f in desc_dir.glob("[0-9]*.json"):
        try:
            data = json.loads(f.read_text())
            if not isinstance(data, dict):
                continue
        except Exception:
            continue

        vis = data.get("visible_objects", {})
        vis_labels: set[str] = set()
        for obj in vis.values():
            if isinstance(obj, dict):
                vis_labels.add(obj.get("label", ""))

        coverage = len(query_labels & vis_labels)
        if coverage > best_coverage:
            best_coverage = coverage
            best_frame = data

    return best_frame


# ---------------------------------------------------------------------------
# Combined teaser figure
# ---------------------------------------------------------------------------

def compose_teaser(perspective_img: np.ndarray,
                   heatmap_img: np.ndarray,
                   query: str,
                   dpi: int = 300,
                   dirfield_img: np.ndarray | None = None,
                   scene_graph_img: np.ndarray | None = None) -> np.ndarray:
    """Compose a multi-panel teaser figure.

    Layout (top row):  Query text | Scene Graph | Perspective View
    Layout (bottom row): Heatmap | Direction Field

    If scene_graph_img or dirfield_img are None, those panels are skipped
    and the layout adapts.

    Returns:
        (H, W, 3) uint8 array.
    """
    from matplotlib.lines import Line2D

    _set_eccv_rc()

    # Build the panel list
    # Top row: query + scene graph + perspective
    # Bottom row: heatmap + direction field
    has_sg = scene_graph_img is not None
    has_df = dirfield_img is not None

    if has_sg and has_df:
        # 2 rows, 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=dpi,
                                  gridspec_kw={"width_ratios": [1.2, 2, 2],
                                               "height_ratios": [1, 1]})
        # Top row
        ax_query = axes[0, 0]
        ax_sg = axes[0, 1]
        ax_persp = axes[0, 2]
        # Bottom row
        ax_heat = axes[1, 0]
        ax_dir = axes[1, 1]
        ax_legend = axes[1, 2]
    elif has_sg:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi,
                                  gridspec_kw={"width_ratios": [1.2, 2, 2]})
        ax_query = axes[0]
        ax_sg = axes[1]
        ax_persp = axes[2]
        ax_heat = None
        ax_dir = None
        ax_legend = None
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi,
                                  gridspec_kw={"width_ratios": [2, 2, 1.2]})
        ax_persp = axes[0]
        ax_heat = axes[1]
        ax_query = axes[2]
        ax_sg = None
        ax_dir = None
        ax_legend = None

    # --- Query text panel ---
    ax_query.set_xlim(0, 1)
    ax_query.set_ylim(0, 1)
    ax_query.set_axis_off()
    props = dict(boxstyle="round,pad=0.6", facecolor="#e8f0fe",
                 edgecolor="#4285f4", linewidth=1.5, alpha=0.95)
    ax_query.text(0.5, 0.5, f"\u201c{query}\u201d",
                  transform=ax_query.transAxes, fontsize=9,
                  verticalalignment="center", horizontalalignment="center",
                  bbox=props, wrap=True, style="italic")
    ax_query.set_title("Query", fontsize=11, fontweight="bold")

    # --- Perspective panel ---
    ax_persp.imshow(perspective_img)
    ax_persp.set_title("Ground Truth View", fontsize=11, fontweight="bold")
    ax_persp.set_axis_off()

    # --- Scene graph panel ---
    if ax_sg is not None and has_sg:
        ax_sg.imshow(scene_graph_img)
        ax_sg.set_title("Scene Graph", fontsize=11, fontweight="bold")
        ax_sg.set_axis_off()

    # --- Heatmap panel ---
    if ax_heat is not None:
        ax_heat.imshow(heatmap_img)
        ax_heat.set_title("Localization Heatmap", fontsize=11, fontweight="bold")
        ax_heat.set_axis_off()

        # Add pose legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff1744',
                   markersize=8, label='Ground Truth'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#00e676',
                   markersize=8, label='Predicted'),
        ]
        ax_heat.legend(handles=legend_elements, loc="lower right",
                       fontsize=8, framealpha=0.85, edgecolor="grey")

    # --- Direction field panel ---
    if ax_dir is not None and has_df:
        ax_dir.imshow(dirfield_img)
        ax_dir.set_title("Direction Field", fontsize=11, fontweight="bold")
        ax_dir.set_axis_off()

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff1744',
                   markersize=8, label='Ground Truth'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
                   markersize=8, label='Predicted'),
        ]
        ax_dir.legend(handles=legend_elements, loc="lower right",
                      fontsize=8, framealpha=0.85, edgecolor="grey")

    # --- Legend / empty panel (bottom-right if 2x3 layout) ---
    if ax_legend is not None:
        ax_legend.set_axis_off()

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="ECCV teaser visualization: perspective + top-down + localization heatmap.")

    ap.add_argument("--dataset", choices=["3rscan", "scannet"], required=True,
                    help="Dataset type.")
    ap.add_argument("--root", type=Path, required=True,
                    help="Dataset root directory.")
    ap.add_argument("--scan-id", "--scan_id", dest="scan_id", required=True,
                    help="Scene identifier.")
    ap.add_argument("--up-axis", "--up_axis", dest="up_axis",
                    choices=["x_up", "y_up", "z_up"], default=None,
                    help="Override vertical axis. If omitted, auto-detected from mesh extents.")
    ap.add_argument("--query", type=str, default=None,
                    help="Natural-language query for localization (requires OPENAI_API_KEY). "
                         "If omitted, use --frame-id to load from a description JSON.")
    ap.add_argument("--frame-id", "--frame_id", dest="frame_id", type=str, default=None,
                    help="Frame ID to load description from (e.g. '001158'). "
                         "Loads from {root}/{scan_id}/output/descriptions/{frame_id}.json. "
                         "If neither --query nor --frame-id is given, lists available frames.")
    ap.add_argument("--desc-dir", "--desc_dir", dest="desc_dir", type=Path, default=None,
                    help="Override description directory. "
                         "Default: {root}/{scan_id}/output/descriptions/")
    ap.add_argument("--graphs-3dssg", "--graphs_3dssg", dest="graphs_3dssg",
                    type=str, default=None,
                    help="Path to 3DSSG graphs .pt file (required for localization).")
    ap.add_argument("--scene-graph", "--scene_graph", dest="scene_graph",
                    action="store_true",
                    help="Produce an Open3D scene graph visualization (PNG).")
    ap.add_argument("--num-retrieval-scenes", "--num_retrieval_scenes",
                    dest="num_retrieval_scenes", type=int, default=0,
                    help="Find N other scenes matching the query objects+relations "
                         "and render scene graph visualizations for them. 0=disabled.")
    ap.add_argument("--output", type=Path, default=Path("teaser_output"),
                    help="Output directory.")

    # Render settings
    ap.add_argument("--width", type=int, default=1920, help="Perspective render width.")
    ap.add_argument("--height", type=int, default=1080, help="Perspective render height.")
    ap.add_argument("--topdown-size", "--topdown_size", dest="topdown_size",
                    type=int, default=2048, help="Top-down render size (square).")
    ap.add_argument("--camera-json", "--camera_json", dest="camera_json",
                    type=str, default=None,
                    help="JSON file with {eye, center, up} for perspective camera.")
    ap.add_argument("--interactive", action="store_true",
                    help="Open an interactive Open3D viewer to pick the perspective camera. "
                         "Navigate to the desired viewpoint and press C to capture.")
    ap.add_argument("--desat", type=float, default=0.0,
                    help="Desaturation of non-matched mesh regions (0.0 = original colours, "
                         "1.0 = fully grey). Matched objects keep full semantic colours.")

    # Localization
    ap.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=None,
                    help="Number of matched objects (default: auto = number of query-graph nodes).")
    ap.add_argument("--grid-step", "--grid_step", dest="grid_step",
                    type=float, default=0.25, help="Grid spacing in metres.")
    ap.add_argument("--h-fov", "--h_fov", dest="h_fov", type=float, default=100.0,
                    help="Horizontal FOV in degrees.")
    ap.add_argument("--v-fov", "--v_fov", dest="v_fov", type=float, default=60.0,
                    help="Vertical FOV in degrees.")
    ap.add_argument("--score-tau", "--score_tau", dest="score_tau",
                    type=float, default=0.0,
                    help="Softmax temperature for probability sharpening. "
                         "0 = raw counts (default). Lower values (e.g. 0.5) = peakier heatmap, "
                         "higher values (e.g. 2.0) = smoother.")
    ap.add_argument("--embedding-type", "--embedding_type", dest="embedding_type",
                    type=str, default="word2vec",
                    help="Embedding backend for scene graphs.")

    # Top-down filtering
    ap.add_argument("--floor-pct", "--floor_pct", dest="floor_pct",
                    type=float, default=0.2)
    ap.add_argument("--ceiling-pct", "--ceiling_pct", dest="ceiling_pct",
                    type=float, default=95.0)
    ap.add_argument("--cutoff-m", "--cutoff_m", dest="cutoff_m",
                    type=float, default=2.1,
                    help="Max height above ground in metres.")

    # Heatmap
    ap.add_argument("--heatmap-alpha", "--heatmap_alpha", dest="heatmap_alpha",
                    type=float, default=0.65,
                    help="Heatmap overlay opacity (0-1). Higher = more visible.")
    ap.add_argument("--heatmap-cmap", "--heatmap_cmap", dest="heatmap_cmap",
                    type=str, default="inferno",
                    choices=["viridis", "plasma", "inferno", "magma", "cividis"],
                    help="Colormap for the heatmap overlay.")
    ap.add_argument("--heatmap-sigma", "--heatmap_sigma", dest="heatmap_sigma",
                    type=float, default=0.0,
                    help="Gaussian blur sigma for heatmap (0 = auto).")
    ap.add_argument("--dpi", type=int, default=300, help="Output DPI.")

    # Direction field
    ap.add_argument("--direction-field", "--direction_field", dest="direction_field",
                    action="store_true",
                    help="Produce a direction field overlay on the top-down view.")
    ap.add_argument("--dir-stride", "--dir_stride", dest="dir_stride",
                    type=int, default=1,
                    help="Draw every Nth camera in the direction field (1 = all).")

    # Combined figure
    ap.add_argument("--combined", action="store_true",
                    help="Also produce a combined 3-panel teaser figure.")

    # Dialogue refinement
    ap.add_argument("--dialogue", action="store_true",
                    help="Run oracle dialogue to refine the predicted pose.")
    ap.add_argument("--dialogue-backend", "--dialogue_backend", dest="dialogue_backend",
                    choices=["a1", "a2", "a3"], default="a3",
                    help="Dialogue backend (default: a3 = frame posterior).")
    ap.add_argument("--dialogue-max-rounds", "--dialogue_max_rounds",
                    dest="dialogue_max_rounds", type=int, default=12,
                    help="Max dialogue rounds (default: 12).")

    return ap.parse_args()


def _load_frame_description(desc_dir: Path, frame_id: str | None) -> dict:
    """Load a frame description JSON, with interactive selection if needed.

    Args:
        desc_dir: Directory containing per-frame JSON files.
        frame_id: Specific frame ID, or None to prompt interactively.

    Returns:
        Parsed frame dict.
    """
    if not desc_dir.exists():
        raise FileNotFoundError(f"Description directory not found: {desc_dir}")

    json_files = sorted(
        p for p in desc_dir.glob("*.json")
        if p.stem != "all_descriptions"
    )
    if not json_files:
        raise FileNotFoundError(f"No description JSONs in {desc_dir}")

    if frame_id is not None:
        # Direct lookup
        target = desc_dir / f"{frame_id}.json"
        if not target.exists():
            raise FileNotFoundError(
                f"Frame description not found: {target}\n"
                f"Available: {[p.stem for p in json_files]}")
        return json.loads(target.read_text())

    # Interactive selection
    print(f"\nAvailable frames in {desc_dir}:")
    print("-" * 70)
    previews = []
    for p in json_files:
        data = json.loads(p.read_text())
        desc = data.get("description", "")
        preview = desc[:75] + "..." if len(desc) > 75 else desc
        n_objs = len(data.get("visible_objects", {}))
        previews.append((p.stem, n_objs, preview))
        print(f"  {p.stem}  ({n_objs} objects)  {preview}")
    print("-" * 70)
    choice = input(f"Enter frame ID [{json_files[0].stem}]: ").strip()
    if not choice:
        choice = json_files[0].stem
    target = desc_dir / f"{choice}.json"
    if not target.exists():
        raise FileNotFoundError(f"Frame '{choice}' not found in {desc_dir}")
    return json.loads(target.read_text())


def main() -> None:
    args = parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    scan_dir = args.root / args.scan_id
    if not scan_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scan_dir}")

    # --- 1. Load scene ---
    print(f"[1/6] Loading {args.dataset} scene: {args.scan_id}")
    if args.dataset == "3rscan":
        mesh, tri2obj, obj2faces = load_scene_3rscan(scan_dir)
    else:
        mesh, tri2obj, obj2faces = load_scene_scannet(scan_dir, args.scan_id)

    up_axis = args.up_axis if args.up_axis else detect_up_axis(mesh)

    # Pre-load 3DSSG graphs once (avoids segfault from repeated torch.load with Open3D)
    sg_all_data = None
    if args.graphs_3dssg:
        sg_all_data = torch.load(args.graphs_3dssg, map_location="cpu", weights_only=False)

    # --- 2. Determine query source ---
    query_sg = None
    query_text = None
    gt_pos = None
    gt_dir = None
    frame_data = None

    if args.query:
        # Manual text query (requires OPENAI_API_KEY)
        if not args.graphs_3dssg:
            raise ValueError("--graphs-3dssg is required when --query is provided")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set. Required for text_to_scenegraph().")
        query_text = args.query

    else:
        # Load from description JSON (--frame-id or interactive selection)
        desc_dir = args.desc_dir or (scan_dir / "output" / "descriptions")
        if desc_dir.exists():
            frame_data = _load_frame_description(desc_dir, args.frame_id)

            query_text = frame_data.get("description", "")
            print(f"[2/6] Loaded frame {frame_data.get('image_index', '?')}: \"{query_text}\"")

            # Build SceneGraph from frame's visible objects + spatial relations
            from langloc.localization.frame_io import frame_to_scenegraph, camera_center_from_pose
            query_sg, meta = frame_to_scenegraph(
                frame_data, embedding_type="word2vec", use_attributes=True)

            # Extract GT camera pose
            scene_pose = frame_data.get("scene_pose")
            if scene_pose is not None:
                gt_pos = camera_center_from_pose(scene_pose)
                # GT viewing direction: negative Z axis of camera in world frame
                pose_mat = np.array(scene_pose, dtype=np.float64)
                gt_dir = pose_mat[:3, 2]  # cam2world OpenCV: camera looks along +Z
                gt_dir_norm = np.linalg.norm(gt_dir)
                if gt_dir_norm > 1e-6:
                    gt_dir = gt_dir / gt_dir_norm
                else:
                    gt_dir = None
                print(f"  GT position: [{gt_pos[0]:.2f}, {gt_pos[1]:.2f}, {gt_pos[2]:.2f}]")
        elif args.frame_id is not None:
            raise FileNotFoundError(f"Description directory not found: {desc_dir}")
        else:
            print("[2/6] No query and no descriptions found — skipping localization.")

    # --- Pre-compute retrieval matches before Open3D rendering ---
    # (Open3D Filament causes heap corruption; extract pure-Python data first)
    retrieval_matches = None
    if args.num_retrieval_scenes > 0 and frame_data is not None and sg_all_data is not None:
        vis = frame_data.get("visible_objects", {})
        _ret_labels: set[str] = set()
        for obj in vis.values():
            if isinstance(obj, dict):
                l = obj.get("label", "")
                if l not in _GENERIC_LABELS:
                    _ret_labels.add(l)

        rels = frame_data.get("spatial_relations", [])
        _ret_rels: set[tuple[str, str, str]] = set()
        for r in rels:
            if isinstance(r, dict) and r.get("relation") in _SPATIAL_RELS:
                s, o = r.get("subject", ""), r.get("object", "")
                if s not in _GENERIC_LABELS and o not in _GENERIC_LABELS:
                    _ret_rels.add((s, o, r["relation"]))

        retrieval_matches = {
            "query_labels": _ret_labels,
            "query_relations": _ret_rels,
            "matches": find_matching_scenes(
                sg_all_data, args.scan_id, _ret_labels, _ret_rels,
                top_k=args.num_retrieval_scenes),
        }

    # --- 3. Run localization ---
    loc_results = None
    if query_sg is not None or (args.query and query_text):
        if not args.graphs_3dssg:
            raise ValueError("--graphs-3dssg is required for localization")

        print(f"[3/6] Running localization...")
        cams, probs, obj_ids, pred_pos, pred_dir, cam_dirs = run_localization(
            mesh, tri2obj, obj2faces,
            query=args.query if query_sg is None else None,
            query_sg=query_sg,
            graphs_3dssg=args.graphs_3dssg,
            scan_id=args.scan_id,
            embedding_type=args.embedding_type,
            top_k=args.top_k,
            grid_step=args.grid_step,
            h_fov_deg=args.h_fov,
            v_fov_deg=args.v_fov,
            up_axis=up_axis,
            score_tau=args.score_tau,
        )
        loc_results = (cams, probs, obj_ids, pred_pos, pred_dir, cam_dirs)
    else:
        print("[3/6] No query — skipping localization.")

    # --- 3b. Dialogue refinement (optional) ---
    dialogue_results = None
    if args.dialogue and loc_results is not None and gt_pos is not None:
        cams, probs, obj_ids, pred_pos, pred_dir, cam_dirs = loc_results
        if pred_pos is not None and pred_dir is not None:
            print(f"[3b/6] Running dialogue refinement...")
            (refined_pos, refined_dir, n_questions, dialogue_log,
             dlg_grid_posterior) = run_dialogue_refinement(
                cams=cams,
                probs=probs,
                cam_dirs=cam_dirs,
                pred_pos=pred_pos,
                pred_dir=pred_dir,
                gt_pos=gt_pos,
                gt_dir=gt_dir,
                dataset_root=args.root,
                scan_id=args.scan_id,
                max_rounds=args.dialogue_max_rounds,
                backend_name=args.dialogue_backend,
            )
            dialogue_results = (refined_pos, refined_dir, n_questions, dialogue_log,
                                dlg_grid_posterior)
        else:
            print("[3b/6] No predicted pose — skipping dialogue.")
    elif args.dialogue and gt_pos is None:
        print("[3b/6] No GT pose available for oracle dialogue — skipping.")

    # --- 4. Color matched objects with semantic instance colours ---
    render_mesh = mesh  # default: original mesh
    if loc_results is not None and loc_results[2]:
        semantic_colors = load_semantic_vertex_colors(scan_dir, args.scan_id, args.dataset)
        if semantic_colors is not None:
            print(f"[4/6] Coloring {len(loc_results[2])} matched objects with semantic colours")
            render_mesh = color_matched_objects(mesh, tri2obj, loc_results[2], semantic_colors,
                                                desat=args.desat)
        else:
            print("[4/6] No labels PLY found — using original mesh colours")
    else:
        print("[4/6] No matched objects — using original mesh colours")

    # --- 5. Perspective render ---
    # Use the frame's camera pose if available (from --frame-id)
    frame_pose = None
    if frame_data is not None and frame_data.get("scene_pose") is not None:
        frame_pose = np.array(frame_data["scene_pose"], dtype=np.float64)
        print(f"[5/6] Rendering perspective view from frame pose ({args.width}x{args.height})")
    else:
        print(f"[5/6] Rendering perspective view ({args.width}x{args.height})")
    persp_img = render_perspective(render_mesh, args.width, args.height, up_axis,
                                   camera_json=args.camera_json,
                                   interactive=args.interactive,
                                   frame_pose=frame_pose)
    persp_titled = _add_title(persp_img, "Ground Truth View", dpi=args.dpi)
    persp_path = out / f"{args.scan_id}_gt.png"
    Image.fromarray(persp_titled).save(persp_path)
    print(f"  Saved: {persp_path}")

    # --- 6. Top-down render + overlays ---
    print(f"[6/6] Rendering top-down view ({args.topdown_size}x{args.topdown_size})")
    topdown_img, intrinsic, extrinsic = render_topdown(
        render_mesh, up_axis, args.topdown_size,
        floor_pct=args.floor_pct, ceiling_pct=args.ceiling_pct,
        cutoff_m=args.cutoff_m)
    topdown_titled = _add_title(topdown_img, "Top-Down View", dpi=args.dpi)
    topdown_path = out / f"{args.scan_id}_topdown.png"
    Image.fromarray(topdown_titled).save(topdown_path)
    print(f"  Saved: {topdown_path}")

    cam_path = out / f"{args.scan_id}_topdown_camera.npz"
    np.savez(cam_path, intrinsic=intrinsic, extrinsic=extrinsic)
    print(f"  Saved camera params: {cam_path}")

    # Predicted / refined pose perspective render
    if loc_results is not None:
        cams, probs, obj_ids, pred_pos, pred_dir, cam_dirs = loc_results
        if pred_pos is not None and pred_dir is not None:
            print(f"  pred_pos: [{pred_pos[0]:.2f}, {pred_pos[1]:.2f}, {pred_pos[2]:.2f}]")
            print(f"  pred_dir: [{pred_dir[0]:.3f}, {pred_dir[1]:.3f}, {pred_dir[2]:.3f}]")
            if gt_dir is not None:
                print(f"  gt_dir:   [{gt_dir[0]:.3f}, {gt_dir[1]:.3f}, {gt_dir[2]:.3f}]")

            up_vec = _up_vector(up_axis)
            r = rendering.OffscreenRenderer(args.width, args.height)
            r.scene.set_background([1.0, 1.0, 1.0, 1.0])
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            r.scene.add_geometry("mesh", render_mesh, mat)
            r.scene.set_lighting(
                rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0, 0, 0))
            r.scene.scene.enable_sun_light(True)
            r.scene.scene.set_sun_light(
                direction=np.array([0.3, -1.0, -1.0], dtype=np.float32),
                color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                intensity=75000.0)

            # Predicted pose view (before dialogue)
            eye = pred_pos.astype(np.float64)
            centre = (pred_pos + pred_dir).astype(np.float64)
            r.setup_camera(60.0, centre, eye, up_vec)
            pred_persp = np.asarray(r.render_to_image())
            if pred_persp.shape[2] == 4:
                pred_persp = pred_persp[:, :, :3]
            pred_persp = pred_persp.astype(np.uint8)
            pred_persp_titled = _add_title(pred_persp, "Predicted Pose View", dpi=args.dpi)
            pred_persp_path = out / f"{args.scan_id}_predicted_perspective.png"
            Image.fromarray(pred_persp_titled).save(pred_persp_path)
            print(f"  Saved: {pred_persp_path}")

            # Dialogue-refined pose view (separate file)
            if dialogue_results is not None and dialogue_results[0] is not None:
                dlg_pos = dialogue_results[0].astype(np.float64)
                dlg_dir_v = dialogue_results[1].astype(np.float64)
                r.setup_camera(60.0, dlg_pos + dlg_dir_v, dlg_pos, up_vec)
                dlg_persp = np.asarray(r.render_to_image())
                if dlg_persp.shape[2] == 4:
                    dlg_persp = dlg_persp[:, :, :3]
                dlg_persp = dlg_persp.astype(np.uint8)
                dlg_persp_titled = _add_title(dlg_persp, "Dialogue-Refined Pose View", dpi=args.dpi)
                dlg_persp_path = out / f"{args.scan_id}_dialogue_perspective.png"
                Image.fromarray(dlg_persp_titled).save(dlg_persp_path)
                print(f"  Saved: {dlg_persp_path}")

    # Heatmap overlay
    if loc_results is not None:
        cams, probs, obj_ids, pred_pos, pred_dir, cam_dirs = loc_results
        if len(cams) > 0 and probs.sum() > 0:
            _dlg_pos = dialogue_results[0] if dialogue_results is not None else None
            _dlg_dir = dialogue_results[1] if dialogue_results is not None else None

            # Original heatmap with GT + predicted pose only
            poses_img = overlay_heatmap(
                topdown_img, intrinsic, extrinsic,
                cams, probs, pred_pos, pred_dir,
                query=query_text or "", alpha=args.heatmap_alpha,
                cmap_name=args.heatmap_cmap, sigma=args.heatmap_sigma,
                h_fov_deg=args.h_fov, up_axis=up_axis,
                dpi=args.dpi,
                gt_pos=gt_pos, gt_dir=gt_dir)
            poses_path = out / f"{args.scan_id}_poses_overlay.png"
            Image.fromarray(poses_img).save(poses_path)
            print(f"  Saved: {poses_path}")

            # Main heatmap overlay — use dialogue-updated probs when available
            if dialogue_results is not None:
                dlg_grid_posterior = dialogue_results[4]
                # Dialogue heatmap with GT + dialogue-refined pose
                heatmap_img = overlay_heatmap(
                    topdown_img, intrinsic, extrinsic,
                    cams, dlg_grid_posterior,
                    pred_pos=pred_pos, pred_dir=pred_dir,
                    query=query_text or "", alpha=args.heatmap_alpha,
                    cmap_name=args.heatmap_cmap, sigma=args.heatmap_sigma,
                    h_fov_deg=args.h_fov, up_axis=up_axis,
                    dpi=args.dpi,
                    gt_pos=gt_pos, gt_dir=gt_dir,
                    dialogue_pos=_dlg_pos, dialogue_dir=_dlg_dir,
                    anchor_probs=probs)
            else:
                heatmap_img = overlay_heatmap(
                    topdown_img, intrinsic, extrinsic,
                    cams, probs, pred_pos, pred_dir,
                    query=query_text or "", alpha=args.heatmap_alpha,
                    cmap_name=args.heatmap_cmap, sigma=args.heatmap_sigma,
                    h_fov_deg=args.h_fov, up_axis=up_axis,
                    dpi=args.dpi,
                    gt_pos=gt_pos, gt_dir=gt_dir)
            heatmap_path = out / f"{args.scan_id}_heatmap_overlay.png"
            Image.fromarray(heatmap_img).save(heatmap_path)
            print(f"  Saved: {heatmap_path}")

            # Dialogue-only heatmap (same updated probs, only GT + dialogue marker)
            if dialogue_results is not None:
                dlg_only_img = overlay_heatmap(
                    topdown_img, intrinsic, extrinsic,
                    cams, dlg_grid_posterior,
                    pred_pos=None, pred_dir=None,
                    query=query_text or "", alpha=args.heatmap_alpha,
                    cmap_name=args.heatmap_cmap, sigma=args.heatmap_sigma,
                    h_fov_deg=args.h_fov, up_axis=up_axis,
                    dpi=args.dpi,
                    gt_pos=gt_pos, gt_dir=gt_dir,
                    dialogue_pos=_dlg_pos, dialogue_dir=_dlg_dir,
                    anchor_probs=probs)
                dlg_only_path = out / f"{args.scan_id}_heatmap_dialogue.png"
                Image.fromarray(dlg_only_img).save(dlg_only_path)
                print(f"  Saved: {dlg_only_path}")

            # Direction field overlay
            dirfield_img = None
            if args.direction_field:
                dirfield_img = overlay_direction_field(
                    topdown_img, intrinsic, extrinsic,
                    cams, probs, cam_dirs,
                    pred_pos=pred_pos, pred_dir=pred_dir,
                    h_fov_deg=args.h_fov,
                    up_axis=up_axis, stride=args.dir_stride,
                    dpi=args.dpi,
                    gt_pos=gt_pos, gt_dir=gt_dir)
                dirfield_path = out / f"{args.scan_id}_direction_field.png"
                Image.fromarray(dirfield_img).save(dirfield_path)
                print(f"  Saved: {dirfield_path}")

            # Scene graph visualization (render before combined so it's available)
            sg_img = None
            if args.scene_graph:
                if not args.graphs_3dssg:
                    print("  Warning: --graphs-3dssg required for --scene-graph, skipping.")
                else:
                    print("Generating scene graph visualization...")
                    matched_ids = loc_results[2] if loc_results is not None else []
                    if args.dataset == "scannet":
                        labels_ply = str(scan_dir / f"{args.scan_id}_vh_clean_2.labels.ply")
                    else:
                        labels_ply = None
                    sg_img = render_scene_graph(
                        mesh, sg_all_data, args.scan_id,
                        matched_obj_ids=matched_ids,
                        obj2faces=obj2faces,
                        up_axis=up_axis,
                        width=args.width, height=args.height,
                        interactive=False,
                        gt_pos=gt_pos, gt_dir=gt_dir,
                        labels_ply=labels_ply)
                    sg_path = out / f"{args.scan_id}_scene_graph.png"
                    Image.fromarray(sg_img).save(sg_path)
                    print(f"  Saved: {sg_path}")

            if args.combined:
                teaser_img = compose_teaser(
                    persp_img, heatmap_img,
                    query_text or "", dpi=args.dpi,
                    dirfield_img=dirfield_img,
                    scene_graph_img=sg_img)
                teaser_path = out / f"{args.scan_id}_teaser.png"
                Image.fromarray(teaser_img).save(teaser_path)
                print(f"  Saved: {teaser_path}")
        else:
            print("  Skipping heatmap — no visible matched objects.")

    # Scene graph fallback: render even without localization results
    if args.scene_graph and loc_results is None:
        if not args.graphs_3dssg:
            print("  Warning: --graphs-3dssg required for --scene-graph, skipping.")
        else:
            print("Generating scene graph visualization...")
            if args.dataset == "scannet":
                labels_ply = str(scan_dir / f"{args.scan_id}_vh_clean_2.labels.ply")
            else:
                labels_ply = None
            sg_img = render_scene_graph(
                mesh, sg_all_data, args.scan_id,
                matched_obj_ids=[],
                obj2faces=obj2faces,
                up_axis=up_axis,
                width=args.width, height=args.height,
                interactive=False,
                gt_pos=gt_pos, gt_dir=gt_dir,
                labels_ply=labels_ply)
            sg_path = out / f"{args.scan_id}_scene_graph.png"
            Image.fromarray(sg_img).save(sg_path)
            print(f"  Saved: {sg_path}")

    # --- Scene retrieval: render scene graphs for pre-computed matches ---
    if retrieval_matches is not None:
        query_labels = retrieval_matches["query_labels"]
        query_relations = retrieval_matches["query_relations"]
        matches = retrieval_matches["matches"]

        print(f"\n[Retrieval] Finding {args.num_retrieval_scenes} matching scenes...")
        print(f"  Query objects: {sorted(query_labels)}")
        print(f"  Query relations: {len(query_relations)} spatial edges")

        retrieval_dir = out / f"retrieval_{args.scan_id}"
        retrieval_dir.mkdir(parents=True, exist_ok=True)

        for rank, m in enumerate(matches, 1):
            m_scan = m["scan_id"]
            m_score = m["score"]
            print(f"  #{rank} {m_scan}  score={m_score:.2f}  "
                  f"objs={sorted(m['matched_labels'])}")

            # Find best frame in that scene
            m_scan_dir = args.root / m_scan
            if not m_scan_dir.exists():
                print(f"    Skipping — scan dir not found: {m_scan_dir}")
                continue

            best_frame = find_best_frame(m_scan_dir, query_labels)
            if best_frame is None:
                print(f"    Skipping — no frame descriptions found")
                continue

            best_fid = best_frame.get("image_index", "?")
            best_desc = best_frame.get("description", "")[:80]
            print(f"    Best frame: {best_fid} — \"{best_desc}...\"")

            # Load mesh for this scene
            if args.dataset == "scannet":
                m_mesh, m_tri2obj, m_obj2faces = load_scene_scannet(
                    m_scan_dir, m_scan)
                m_labels_ply = str(m_scan_dir / f"{m_scan}_vh_clean_2.labels.ply")
            else:
                m_mesh, m_tri2obj, m_obj2faces = load_scene_3rscan(m_scan_dir)
                m_labels_ply = None

            # Get matched object IDs from the scene graph
            m_g = sg_all_data.get(m_scan, {})
            m_objects = m_g.get("objects", {})
            m_matched_ids = []
            for oid_str, obj in m_objects.items():
                if obj["label"] in m["matched_labels"]:
                    m_matched_ids.append(int(oid_str))

            # Get GT pose from best frame for camera placement
            m_gt_pos, m_gt_dir = None, None
            m_pose = best_frame.get("scene_pose")
            if m_pose is not None:
                m_pose_mat = np.array(m_pose, dtype=np.float64)
                m_gt_pos = m_pose_mat[:3, 3]
                m_dir = m_pose_mat[:3, 2]
                m_dir_norm = np.linalg.norm(m_dir)
                if m_dir_norm > 1e-6:
                    m_gt_dir = m_dir / m_dir_norm

            m_up = detect_up_axis(m_mesh)

            # Render scene graph — keep original mesh colors for retrievals
            sg_img = render_scene_graph(
                m_mesh, sg_all_data, m_scan,
                matched_obj_ids=m_matched_ids,
                obj2faces=m_obj2faces,
                up_axis=m_up,
                width=args.width, height=args.height,
                gt_pos=m_gt_pos, gt_dir=m_gt_dir,
                desat=0.0,
                labels_ply=m_labels_ply)
            sg_path = retrieval_dir / f"{rank}_{m_scan}_scene_graph.png"
            Image.fromarray(sg_img).save(sg_path)
            print(f"    Saved: {sg_path}")

    # --- Pose error computation ---
    if loc_results is not None and gt_pos is not None:
        cams, probs, obj_ids, pred_pos, pred_dir, cam_dirs = loc_results
        from langloc.eval.metrics import (
            compute_metrics_standard, compute_view_iou_error)

        up_idx = {"x_up": 0, "y_up": 1, "z_up": 2}.get(up_axis, 2)
        floor_axes = [i for i in range(3) if i != up_idx]

        # 2D floor position error (like the paper)
        gt_2d = gt_pos[floor_axes]
        pred_2d = pred_pos[floor_axes] if pred_pos is not None else gt_2d
        pos_err = float(np.linalg.norm(pred_2d - gt_2d))

        # Top-10 position error
        dists_2d = np.linalg.norm(cams[:, floor_axes] - gt_2d[None, :], axis=1)
        k = min(10, len(probs))
        top_idx = np.argpartition(probs, -k)[-k:]
        top10_err = float(dists_2d[top_idx].min())

        # Angular error
        ang_err = None
        if pred_dir is not None and gt_dir is not None:
            pred_2d_dir = pred_dir[floor_axes]
            gt_2d_dir = gt_dir[floor_axes]
            pn = np.linalg.norm(pred_2d_dir)
            gn = np.linalg.norm(gt_2d_dir)
            if pn > 1e-6 and gn > 1e-6:
                cos_a = np.clip(np.dot(pred_2d_dir / pn, gt_2d_dir / gn), -1, 1)
                ang_err = float(np.degrees(np.arccos(cos_a)))

        # 3D IoU
        iou_val = None
        try:
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            tris = np.asarray(mesh.triangles, dtype=np.int32)
            tri_pts = verts[tris]  # (T, 3, 3)
            tri_centroids = tri_pts.mean(axis=1)
            e1 = tri_pts[:, 1] - tri_pts[:, 0]
            e2 = tri_pts[:, 2] - tri_pts[:, 0]
            tri_areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            rc_scene = o3d.t.geometry.RaycastingScene()
            geom_id = rc_scene.add_triangles(mesh_t)

            hfov = math.radians(args.h_fov) if args.h_fov > 0 else math.radians(60)
            vfov = hfov * args.height / args.width

            iou, iou_err, _, _ = compute_view_iou_error(
                gt_pos, gt_dir, pred_pos, pred_dir,
                hfov, vfov, rc_scene, geom_id,
                tri_pts, tri_centroids, tri_areas)
            iou_val = iou
        except Exception as e:
            print(f"  Warning: IoU computation failed: {e}")

        # Dialogue-refined metrics
        dlg_pos_err = None
        dlg_ang_err = None
        dlg_iou_val = None
        dlg_n_questions = None
        if dialogue_results is not None:
            refined_pos, refined_dir, n_q, dlg_log = dialogue_results[:4]
            dlg_n_questions = n_q
            ref_2d = refined_pos[floor_axes] if refined_pos is not None else gt_2d
            dlg_pos_err = float(np.linalg.norm(ref_2d - gt_2d))
            if refined_dir is not None and gt_dir is not None:
                ref_2d_dir = refined_dir[floor_axes]
                rn = np.linalg.norm(ref_2d_dir)
                gn2 = np.linalg.norm(gt_2d_dir) if 'gt_2d_dir' in dir() else np.linalg.norm(gt_dir[floor_axes])
                if rn > 1e-6 and gn2 > 1e-6:
                    cos_d = np.clip(np.dot(ref_2d_dir / rn, gt_dir[floor_axes] / gn2), -1, 1)
                    dlg_ang_err = float(np.degrees(np.arccos(cos_d)))
            try:
                dlg_iou, _, _, _ = compute_view_iou_error(
                    gt_pos, gt_dir, refined_pos, refined_dir,
                    hfov, vfov, rc_scene, geom_id,
                    tri_pts, tri_centroids, tri_areas)
                dlg_iou_val = dlg_iou
            except Exception:
                pass

        # Print summary
        print(f"\n  === Pose Error Summary ===")
        print(f"  Position error (2D):    {pos_err:.3f} m")
        print(f"  Top-10 position error:  {top10_err:.3f} m")
        if ang_err is not None:
            print(f"  Angular error:          {ang_err:.1f} deg")
        if iou_val is not None:
            print(f"  3D View IoU:            {iou_val:.3f}")

        if dlg_pos_err is not None:
            print(f"\n  === After Dialogue ({dlg_n_questions} questions) ===")
            print(f"  Position error (2D):    {dlg_pos_err:.3f} m  (was {pos_err:.3f} m)")
            if dlg_ang_err is not None:
                print(f"  Angular error:          {dlg_ang_err:.1f} deg  (was {f'{ang_err:.1f}' if ang_err is not None else '—'} deg)")
            if dlg_iou_val is not None:
                print(f"  3D View IoU:            {dlg_iou_val:.3f}  (was {f'{iou_val:.3f}' if iou_val is not None else '—'})")

        # Write metrics file (Markdown with LaTeX table)
        metrics_path = out / f"{args.scan_id}_metrics.md"
        frame_id_str = frame_data.get("image_index", "?") if frame_data else "?"

        # 3D errors for the .md (complement the 2D errors already computed)
        from langloc.dialogue.math_utils import pose_errors as _pe
        pred_3d_pos_err, pred_3d_rot_err = _pe(pred_pos, pred_dir, gt_pos, gt_dir)

        lines = [
            f"# Pose Error — {args.scan_id} / frame {frame_id_str}",
            "",
            f"**Query:** {query_text or '(none)'}",
            "",
            "## Metrics (Fine Localization)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Position error (2D) | {pos_err:.3f} m |",
            f"| Position error (3D) | {pred_3d_pos_err:.3f} m |",
            f"| Top-10 position error | {top10_err:.3f} m |",
            f"| Angular error (2D) | {f'{ang_err:.1f} deg' if ang_err is not None else '—'} |",
            f"| Angular error (3D) | {f'{pred_3d_rot_err:.1f} deg' if not np.isnan(pred_3d_rot_err) else '—'} |",
            f"| 3D View IoU | {f'{iou_val:.3f}' if iou_val is not None else '—'} |",
            f"| Matched objects | {len(obj_ids)} |",
            f"| Grid points | {len(cams)} |",
        ]
        if dlg_pos_err is not None:
            refined_pos, refined_dir, n_q, dlg_log = dialogue_results[:4]
            dlg_3d_pos_err, dlg_3d_rot_err = _pe(refined_pos, refined_dir, gt_pos, gt_dir)
            lines += [
                "",
                f"## Metrics (After Dialogue — {dlg_n_questions} questions, backend: {args.dialogue_backend.upper()})",
                "",
                "| Metric | Before | After | Δ |",
                "|--------|--------|-------|---|",
                f"| Position error (2D) | {pos_err:.3f} m | {dlg_pos_err:.3f} m | {dlg_pos_err - pos_err:+.3f} m |",
                f"| Position error (3D) | {pred_3d_pos_err:.3f} m | {dlg_3d_pos_err:.3f} m | {dlg_3d_pos_err - pred_3d_pos_err:+.3f} m |",
                f"| Angular error (2D) | {f'{ang_err:.1f}' if ang_err is not None else '—'} deg | {f'{dlg_ang_err:.1f}' if dlg_ang_err is not None else '—'} deg | {f'{dlg_ang_err - ang_err:+.1f}' if ang_err is not None and dlg_ang_err is not None else '—'} deg |",
                f"| Angular error (3D) | {f'{pred_3d_rot_err:.1f}' if not np.isnan(pred_3d_rot_err) else '—'} deg | {f'{dlg_3d_rot_err:.1f}' if not np.isnan(dlg_3d_rot_err) else '—'} deg | {f'{dlg_3d_rot_err - pred_3d_rot_err:+.1f}' if not (np.isnan(pred_3d_rot_err) or np.isnan(dlg_3d_rot_err)) else '—'} deg |",
                f"| 3D View IoU | {f'{iou_val:.3f}' if iou_val is not None else '—'} | {f'{dlg_iou_val:.3f}' if dlg_iou_val is not None else '—'} | {f'{dlg_iou_val - iou_val:+.3f}' if iou_val is not None and dlg_iou_val is not None else '—'} |",
                "",
                "### Dialogue Log",
                "",
                "| Round | Question | Answer | Top P |",
                "|-------|----------|--------|-------|",
            ]
            for i, (q_text, ans, tp_str) in enumerate(dlg_log, 1):
                lines.append(f"| {i} | {q_text} | {ans} | {tp_str} |")

        # LaTeX table
        lines += [
            "",
            "## LaTeX",
            "",
            "```latex",
            "\\begin{table}[h]",
            "\\centering",
            f"\\caption{{Pose error for {args.scan_id}, frame {frame_id_str}.}}",
        ]
        if dlg_pos_err is not None:
            lines += [
                "\\begin{tabular}{lcc}",
                "\\toprule",
                "Metric & Before & After \\\\",
                "\\midrule",
                f"Position error (2D) & {pos_err:.3f}\\,m & {dlg_pos_err:.3f}\\,m \\\\",
                f"Position error (3D) & {pred_3d_pos_err:.3f}\\,m & {dlg_3d_pos_err:.3f}\\,m \\\\",
                f"Angular error (2D) & {f'{ang_err:.1f}' if ang_err is not None else '--'}$^\\circ$ & {f'{dlg_ang_err:.1f}' if dlg_ang_err is not None else '--'}$^\\circ$ \\\\",
                f"Angular error (3D) & {f'{pred_3d_rot_err:.1f}' if not np.isnan(pred_3d_rot_err) else '--'}$^\\circ$ & {f'{dlg_3d_rot_err:.1f}' if not np.isnan(dlg_3d_rot_err) else '--'}$^\\circ$ \\\\",
                f"3D View IoU & {f'{iou_val:.3f}' if iou_val is not None else '--'} & {f'{dlg_iou_val:.3f}' if dlg_iou_val is not None else '--'} \\\\",
                f"Questions asked & \\multicolumn{{2}}{{c}}{{{dlg_n_questions}}} \\\\",
                "\\bottomrule",
                "\\end{tabular}",
            ]
        else:
            lines += [
                "\\begin{tabular}{lc}",
                "\\toprule",
                "Metric & Value \\\\",
                "\\midrule",
                f"Position error (2D) & {pos_err:.3f}\\,m \\\\",
                f"Position error (3D) & {pred_3d_pos_err:.3f}\\,m \\\\",
                f"Angular error (2D) & {f'{ang_err:.1f}' if ang_err is not None else '--'}$^\\circ$ \\\\",
                f"Angular error (3D) & {f'{pred_3d_rot_err:.1f}' if not np.isnan(pred_3d_rot_err) else '--'}$^\\circ$ \\\\",
                f"3D View IoU & {f'{iou_val:.3f}' if iou_val is not None else '--'} \\\\",
                f"Matched objects & {len(obj_ids)} \\\\",
                f"Grid points & {len(cams)} \\\\",
                "\\bottomrule",
                "\\end{tabular}",
            ]
        lines += [
            "\\end{table}",
            "```",
            "",
        ]
        metrics_path.write_text("\n".join(lines))
        print(f"  Saved: {metrics_path}")

    print("Done.")


if __name__ == "__main__":
    main()
    # Force immediate exit to avoid segfault from Open3D Filament cleanup
    # during normal Python interpreter shutdown.
    os._exit(0)
