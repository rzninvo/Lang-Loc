#!/usr/bin/env python3
"""
3RScan-style NBV selection + mask export (instance & semantic) + debug viz.

Pipeline:
1) Load 3RScan scene: mesh (+ vertex colors from OBJ+MTL+PNG), segmentation,
   semseg.json groups, color frames, poses, and intrinsics.
2) Filter frames by BRISQUE quality & subsample (with fallback thresholds).
3) Rasterize visibility for candidate frames; compute per-object pixel counts.
4) Greedy NBV selection + adaptive K-means clustering for diversity.
5) Save representative frames + poses + masks (instance/semantic).
6) (Optional) Auto-clean raw frame files.

Author: Roham Zendehdel Nobari (rzendehdel@ethz.ch)
"""

import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from tqdm import tqdm
import plyfile
from sklearn.neighbors import NearestNeighbors

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesUV
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, HardPhongShader,
    PointLights, RasterizationSettings
)
from pytorch3d.utils import cameras_from_opencv_projection

# Project imports - modular frame selection components
from src.frame_selection.iqa import filter_quality_images
from src.frame_selection.visibility import (
    VOID_ID,
    make_p3d_camera_from_opencv,
    make_rasterizer,
    per_face_object_ids,
    precompute_object_geometry,
    rasterize_visibility,
    compute_image_visibility,
    compute_visible_objects,
    compute_spatial_relations,
)
from src.frame_selection.dpp import (
    compute_face_normals,
    compute_clip_embeddings,
    dpp_select_views,
)
from src.frame_selection.legacy import greedy_next_best_views, cluster_camera_poses
from src.frame_selection.masks import pix_to_instance_mask, pix_to_semantic_mask, save_png16
from src.utils.camera_utils import (
    invert_se3_to_opencv,
    load_cam2world,
    load_intrinsics_info,
)
from src.utils.config_loader import load_config
from src.utils.nbv_config import NBVConfig, extract_nbv_config
import matplotlib.pyplot as plt


# ----------------------------- Loaders ---------------------------------

def load_mesh_with_textures(scene_path, device):
    """
    Load the refined 3RScan mesh together with its texture map and return a
    PyTorch3D `Meshes` instance ready for rendering.

    The OBJ uses shared vertices for UV seams; to keep texture coordinates
    consistent with PyTorch3D we duplicate vertices per triangle and return the
    index mapping so auxiliary per-vertex data (e.g., segmentation) can be
    expanded in the same way.

    Args:
        scene_path (Path | str): Directory containing the 3RScan assets.
        device (torch.device): Device on which the mesh tensors should live.

    Returns:
        tuple[Meshes, np.ndarray, np.ndarray]:
            - meshes:  Single-item `Meshes` with duplicated vertices and UVs.
            - orig_vertex_idx: 1D array mapping each expanded vertex back to the
              original OBJ vertex index (length = 3 * num_faces).
            - verts: Original vertex positions from the OBJ (N, 3), kept so the
              caller can build per-vertex annotations without duplication.
    """

    obj_path = scene_path / "mesh.refined.v2.obj"
    tex_path = scene_path / "mesh.refined_0.png"
    mesh = o3d.io.read_triangle_mesh(str(obj_path), True)

    uvs = np.asarray(mesh.triangle_uvs)

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    verts_uvs = uvs
    faces_uvs = np.arange(len(verts_uvs)).reshape(-1, 3)

    tex_img = np.array(Image.open(tex_path).convert("RGB")).astype(np.float32) / 255.0
    tex_img = tex_img ** 2.2

    # Expand verts to match UVs (the key step)
    orig_vertex_idx = faces.reshape(-1)
    expanded_verts = verts[orig_vertex_idx]
    expanded_faces = np.arange(len(expanded_verts)).reshape(-1, 3)

    meshes = Meshes(
        verts=[torch.tensor(expanded_verts, dtype=torch.float32, device=device)],
        faces=[torch.tensor(expanded_faces, dtype=torch.int64, device=device)],
        textures=TexturesUV(
            maps=torch.tensor(tex_img, dtype=torch.float32, device=device).unsqueeze(0),
            faces_uvs=[torch.tensor(faces_uvs, dtype=torch.int64, device=device)],
            verts_uvs=[torch.tensor(verts_uvs, dtype=torch.float32, device=device)],
        ),
    )
    return meshes, orig_vertex_idx, verts


def load_segments_and_instances(scene_path: Path, base_vertices: np.ndarray):
    """
    Transfer per-vertex instance IDs and semantic labels onto the mesh.

    The official `mesh.refined.0.010000.segs.v2.json` is aligned with a decimated
    mesh, so we instead use the annotated point cloud
    `labels.instances.annotated.v2.ply` and find the nearest annotated point for
    every vertex of the textured mesh.

    Args:
        scene_path (Path): 3RScan scene directory.
        base_vertices (np.ndarray): (N, 3) array of the original (non-duplicated)
            OBJ vertices.

    Returns:
        tuple[np.ndarray, dict[int, int], dict[int, str]]:
            - vert_obj: per-vertex instance IDs aligned with `base_vertices`.
            - seg_to_obj: identity mapping retained for API compatibility
              (`segment` -> `objectId`).
            - obj_to_label: objectId -> semantic label (lowercased).
    """
    semseg_json = scene_path / "semseg.v2.json"
    ply_path = scene_path / "labels.instances.annotated.v2.ply"
    if not semseg_json.exists() or not ply_path.exists():
        raise FileNotFoundError("Missing semantic annotations or annotated point cloud.")

    groups = json.loads(semseg_json.read_text())["segGroups"]
    obj_to_label = {int(g["objectId"]): g.get("label", "").strip().lower() for g in groups}

    ply = plyfile.PlyData.read(ply_path)
    pts = np.vstack([ply["vertex"][axis] for axis in ("x", "y", "z")]).T.astype(np.float32)
    obj_ids = np.asarray(ply["vertex"]["objectId"], dtype=np.int32)

    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
    nn.fit(pts)
    _, idx = nn.kneighbors(base_vertices.astype(np.float32), return_distance=True)
    vert_obj = obj_ids[idx[:, 0]]

    seg_to_obj = {int(oid): int(oid) for oid in np.unique(vert_obj) if oid >= 0}
    return vert_obj.astype(np.int32), seg_to_obj, obj_to_label


def load_object_geometry_from_semseg(scene_path: Path):
    """
    Load per-object geometry from 3RScan `semseg.v2.json` OBB fields.

    The file provides, per object:
      - centroid
      - axis lengths
      - normalized axes (3x3)

    We convert this into the geometry structure expected by
    `compute_visible_objects(...)`:
      centroid_world, bbox_min/max, obb_center/axes/extents.
    """
    semseg_json = scene_path / "semseg.v2.json"
    if not semseg_json.exists():
        return {}

    groups = json.loads(semseg_json.read_text()).get("segGroups", [])
    geometry = {}

    # 8 corners of a box in local OBB coordinates.
    signs = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )

    for g in groups:
        oid = int(g.get("objectId", -1))
        if oid < 0:
            continue

        obb = g.get("obb") or {}
        try:
            center = np.asarray(obb["centroid"], dtype=np.float64).reshape(3)
            axes_lengths = np.asarray(obb["axesLengths"], dtype=np.float64).reshape(3)
            axes = np.asarray(obb["normalizedAxes"], dtype=np.float64).reshape(3, 3)
        except Exception:
            continue

        # Normalize row-wise to ensure unit axis vectors.
        norms = np.linalg.norm(axes, axis=1, keepdims=True)
        axes = axes / np.maximum(norms, 1e-12)
        extents = np.maximum(axes_lengths / 2.0, 1e-8)

        # rows in `axes` are interpreted as world-space axis vectors.
        local_corners = signs * extents[None, :]
        corners_world = center[None, :] + local_corners @ axes
        bbox_min = corners_world.min(axis=0)
        bbox_max = corners_world.max(axis=0)

        geometry[oid] = {
            "centroid_world": center,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "obb_center": center,
            "obb_axes": axes,
            "obb_extents": extents,
        }

    return geometry


# -------------------------- Debug Visualization Helpers -----------------------

@torch.no_grad()
def _render_textured_rgb(meshes, cameras, H, W, device):
    """
    Render a textured 3RScan mesh using PyTorch3D at full resolution.

    Lighting is kept ambient-only to match the baked texture as closely as
    possible. The function is primarily used for debug side-by-side comparisons.

    Args:
        meshes (Meshes): PyTorch3D mesh batch (single mesh expected).
        cameras (PerspectiveCameras): Camera aligned with the RGB frame.
        H, W (int): Output resolution.
        device (torch.device): Device to run the renderer on.

    Returns:
        np.ndarray: (H, W, 3) float image in [0, 1].
    """
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        faces_per_pixel=1,
        blur_radius=0.0,
        perspective_correct=True,
        cull_backfaces=True,
    )

    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    lights = PointLights(
        device=device,
        ambient_color=((1.0, 1.0, 1.0),),
        diffuse_color=((0.0, 0.0, 0.0),),
        specular_color=((0.0, 0.0, 0.0),),
        location=[[0.0, 0.0, 0.0]],
    )
    shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    img = renderer(meshes, cameras=cameras)[0, ..., :3].clamp(0, 1).cpu().numpy()
    return img

def _save_side_by_side(rgb_path: Path, rendered_rgb: np.ndarray, title_left: str, title_right: str, out_path: Path):
    """
    Save a diagnostic figure comparing dataset RGB and PyTorch3D render.

    Args:
        rgb_path (Path): Path to the original RGB frame.
        rendered_rgb (np.ndarray): Rendered view in [0, 1].
        title_left (str): Title for the dataset image.
        title_right (str): Title for the rendered image.
        out_path (Path): Destination PNG path.
    """
    rgb = np.array(Image.open(rgb_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(rgb); plt.title(title_left); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(rendered_rgb); plt.title(title_right); plt.axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=100); plt.close()

def _save_overlay(rgb_path: Path, pix_to_face: torch.Tensor, out_path: Path):
    """
    Overlay the rasterized silhouette on top of the RGB frame for alignment checks.

    Args:
        rgb_path (Path): Path to the RGB frame to use as the background.
        pix_to_face (torch.Tensor): Rasterizer output with face indices.
        out_path (Path): Destination PNG path for the overlay.
    """
    rgb = np.array(Image.open(rgb_path))
    vis = pix_to_face.detach().cpu().numpy()

    # ---- normalize dimensions ----
    if vis.ndim == 4:          # (N,H,W,K)
        vis = vis[0, ..., 0]
    elif vis.ndim == 3:        # (N,H,W)
        vis = vis[0]
    elif vis.ndim == 2 and vis.shape[0] == 1:
        vis = vis.reshape(rgb.shape[0], rgb.shape[1])
    elif vis.ndim == 1:
        vis = vis.reshape(rgb.shape[0], rgb.shape[1])

    # ---- orientation correction ----
    if vis.shape != rgb.shape[:2]:
        if vis.shape[::-1] == rgb.shape[:2]:
            vis = vis.T
        else:
            from cv2 import resize, INTER_NEAREST
            vis = resize(vis, (rgb.shape[1], rgb.shape[0]), interpolation=INTER_NEAREST)

    vis = (vis >= 0).astype(np.uint8)

    overlay = rgb.copy()
    overlay[vis > 0] = (0.3 * overlay[vis > 0] + 0.7 * np.array([0, 255, 0])).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_path)

# ------------------------------- Main ----------------------------------

def main(scene_id: str, config_path: str, device_str=None,
         debug=False, auto_clean=False,
         save_semantic_masks=False, save_instance_masks=False):
    """
    Entry point for selecting representative 3RScan frames and exporting assets.

    Steps performed:
        1. Load mesh, annotations, and camera intrinsics/poses.
        2. Filter frames by BRISQUE quality and optional subsampling.
        3. Rasterize downsampled visibility to compute per-object coverage.
        4. Run greedy next-best-view selection and K-means clustering.
        5. Copy RGB/Depth/Pose files for the selected frames.
        6. Optionally export semantic/instance masks and render debug visualizations.

    Args:
        scene_id (str): UUID of the 3RScan scene.
        config_path (str): Path to the YAML config with dataset settings.
        device_str (str | None): Optional override for the compute device
            (e.g., "cuda:0" or "cpu"). Defaults to CUDA if available.
        debug (bool): If True, generate PyTorch3D vs RGB comparisons.
        auto_clean (bool): If True, removes raw frames once outputs are saved.
        save_semantic_masks (bool): Export semantic 16-bit PNG masks if True.
        save_instance_masks (bool): Export instance 16-bit PNG masks if True.
    """

    cfg = load_config(config_path)
    nbv_cfg = extract_nbv_config(cfg, dataset="3rscan")

    dataset_path = Path(cfg["paths"]["3rscan_dataset_path"])
    scan_path = dataset_path / scene_id

    output_dir = scan_path / nbv_cfg.output_folder
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------- Device Setup ----------------------------
    device = torch.device(device_str or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    # ---------------------- Load Mesh + Labels --------------------------
    meshes, orig_vertex_idx, original_verts = load_mesh_with_textures(scan_path, device)

    faces = meshes.faces_packed().cpu().numpy()
    verts = meshes.verts_packed().cpu().numpy()

    vert_seg_raw, seg_to_obj, obj_to_label = load_segments_and_instances(scan_path, original_verts)
    max_required_idx = int(orig_vertex_idx.max())
    if max_required_idx >= len(vert_seg_raw):
        pad = max_required_idx + 1 - len(vert_seg_raw)
        print(f"[WARN] Padding vert_seg with {pad} void entries to cover expanded vertices.")
        vert_seg_raw = np.pad(vert_seg_raw, (0, pad), constant_values=-1)
    vert_seg = vert_seg_raw[orig_vertex_idx]

    faces_np = meshes.faces_packed().cpu().numpy()
    verts_np = meshes.verts_packed().cpu().numpy()

    max_idx = vert_seg.shape[0]
    if (faces_np >= max_idx).any():
        raise ValueError("Vertex segmentation array misaligned with expanded mesh.")

    faces = faces_np
    face_obj_ids = per_face_object_ids(faces, vert_seg, seg_to_obj)

    obj_to_sem_id = {oid: idx + 1 for idx, oid in enumerate(obj_to_label.keys())}

    fx, fy, cx, cy = load_intrinsics_info(scan_path / "_info.txt")

    # ------------------------- Collect Frames ---------------------------
    frame_ids = [p.name.replace(".color.jpg", "") for p in sorted(scan_path.glob("frame-*.color.jpg"))]
    if not frame_ids:
        raise RuntimeError("No frames found")
    sample_img = np.array(Image.open(scan_path / f"{frame_ids[0]}.color.jpg"))
    H0, W0 = map(int, sample_img.shape[:2])

    # IQA filtering with 3RScan file pattern (*.color.jpg)
    iqa_metric = nbv_cfg.iqa_metric
    iqa_threshold = nbv_cfg.iqa_threshold
    iqa_device = nbv_cfg.iqa_device

    frame_ids = filter_quality_images(
        scan_path,
        metric_name=iqa_metric,
        threshold=iqa_threshold,
        file_pattern="*.color.jpg",
        device=iqa_device,
    )

    if not frame_ids:
        raise RuntimeError("No quality images found.")

    frame_ids = [fid.replace(".color", "") for fid in frame_ids]
    if nbv_cfg.subsample_factor > 1:
        frame_ids = frame_ids[::nbv_cfg.subsample_factor]
    if nbv_cfg.limit_images is not None:
        frame_ids = frame_ids[:int(nbv_cfg.limit_images)]

    print(f"[INFO] Using {len(frame_ids)} candidate frames after filtering and subsampling.")

    # --------------------------- Debug Visualization ---------------------------
    if debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        viz_n = min(5, len(frame_ids))  # visualize first few frames
        print(f"[DEBUG] Rendering {viz_n} debug comparisons to {debug_dir}")

        for fid in frame_ids[:viz_n]:
            rgb_path = scan_path / f"{fid}.color.jpg"
            pose = load_cam2world(scan_path / f"{fid}.pose.txt")
            R_cv, t_cv = invert_se3_to_opencv(pose)

            cams_full = make_p3d_camera_from_opencv(R_cv, t_cv, fx, fy, cx, cy, H0, W0, device)

            rendered = _render_textured_rgb(meshes, cams_full, H0, W0, device)

            sxs_path = debug_dir / f"{fid}_side_by_side.png"
            _save_side_by_side(rgb_path, rendered, f"RGB {fid}", "PyTorch3D render", sxs_path)

            # optional silhouette overlay to check pixel alignment
            rasterizer_full = make_rasterizer(H0, W0, faces_per_pixel=1)
            pix_to_face_full, _ = rasterize_visibility(meshes, cams_full, rasterizer_full)
            ov_path = debug_dir / f"{fid}_overlay.png"
            _save_overlay(rgb_path, pix_to_face_full, ov_path)

        print(f"[DEBUG] Saved debug renders to {debug_dir}")


    # ---------------------- Visibility Pass (Cached) ---------------------
    downsample = nbv_cfg.image_downsample_factor
    H_vis = max(1, H0 // max(1, downsample))
    W_vis = max(1, W0 // max(1, downsample))
    fx_vis, fy_vis, cx_vis, cy_vis = fx / downsample, fy / downsample, cx / downsample, cy / downsample

    cache_json = cache_dir / f"{scene_id}.json"
    if cache_json.exists():
        print(f"[INFO] Loading cached visibility stats: {cache_json}")
        image_stats = json.loads(cache_json.read_text())
        # Check for DPP-required fields
        if nbv_cfg.dpp_enabled and image_stats and "visible_face_ids" not in image_stats[0]:
            print("[WARN] Cache missing visible_face_ids (needed for DPP). Recomputing...")
            cache_json.unlink()
            image_stats = None
        elif image_stats:
            sample_obj_meta = None
            for entry in image_stats[:10]:
                vo = entry.get("visible_objects", {})
                if vo:
                    sample_obj_meta = next(iter(vo.values()))
                    break
            if sample_obj_meta is not None and "obb_world" not in sample_obj_meta:
                print("[WARN] Cache missing obb_world/full-object geometry. Recomputing...")
                cache_json.unlink()
                image_stats = None
    else:
        image_stats = None

    if image_stats is None:
        print(f"[INFO] Computing visibility for {len(frame_ids)} frames at {H_vis}x{W_vis}...")
        rasterizer_vis = make_rasterizer(H_vis, W_vis,
                                         faces_per_pixel=nbv_cfg.faces_per_pixel,
                                         bin_size=nbv_cfg.bin_size,
                                         max_faces_per_bin=nbv_cfg.max_faces_per_bin,
                                         blur_radius=nbv_cfg.blur_radius,
                                         )
        image_stats = []
        object_geometry_cache = load_object_geometry_from_semseg(scan_path)
        if not object_geometry_cache:
            print("[WARN] semseg.v2.json OBB geometry missing/unusable. Falling back to mesh-derived geometry.")
            object_geometry_cache = precompute_object_geometry(verts, faces, face_obj_ids)
        else:
            mesh_object_ids = set(int(oid) for oid in np.unique(face_obj_ids[face_obj_ids >= 0]))
            missing_geom = sorted(mesh_object_ids - set(object_geometry_cache.keys()))
            if missing_geom:
                print(
                    f"[WARN] semseg geometry missing {len(missing_geom)} object(s). "
                    "Filling missing entries from mesh geometry."
                )
                fallback_geom = precompute_object_geometry(verts, faces, face_obj_ids)
                for oid in missing_geom:
                    if oid in fallback_geom:
                        object_geometry_cache[oid] = fallback_geom[oid]

        for fid in tqdm(frame_ids, desc="Visibility", dynamic_ncols=True):
            pose = load_cam2world(scan_path / f"{fid}.pose.txt")
            R_cv, t_cv = invert_se3_to_opencv(pose)

            cams = make_p3d_camera_from_opencv(
                R_cv, t_cv, fx_vis, fy_vis, cx_vis, cy_vis, H_vis, W_vis, device
            )
            pix_to_face, _ = rasterize_visibility(meshes, cams, rasterizer_vis)
            obj_px, total_px = compute_image_visibility(pix_to_face, face_obj_ids)

            visible_objects = compute_visible_objects(
                verts, faces, vert_seg, seg_to_obj, obj_to_label,
                obj_px, face_obj_ids, pix_to_face,
                pose, R_cv, t_cv,
                fx_vis, fy_vis, cx_vis, cy_vis,
                W_vis, H_vis,
                fov_depth_clip=nbv_cfg.fov_depth_clip,
                coverage_threshold=nbv_cfg.coverage_threshold,
                min_pixel_count=nbv_cfg.min_pixel_count,
                full_object_geometry=object_geometry_cache,
            )
            spatial_relations = compute_spatial_relations(
                visible_objects,
                max_distance=nbv_cfg.spatial_max_distance,
                size_ratio_threshold=nbv_cfg.spatial_size_ratio_threshold,
                eps=nbv_cfg.spatial_eps,
            )

            # Extract unique visible face indices for DPP normal/novelty computation
            p2f_np = pix_to_face.cpu().numpy()
            visible_face_ids = np.unique(p2f_np[p2f_np >= 0]).tolist()

            image_entry = {
                "fid": fid,
                "obj_pixels": obj_px,
                "total_labeled_px": int(total_px),
                "visible_objects": visible_objects,
                "spatial_relations": spatial_relations,
                "visible_face_ids": visible_face_ids,
            }
            image_stats.append(image_entry)


        cache_json.write_text(json.dumps(image_stats, indent=2))
        print(f"[INFO] Saved visibility cache: {cache_json}")

    # ---------------------- VIEW SELECTION --------------------------------
    if nbv_cfg.dpp_enabled:
        # ======================== DPP VIEW SELECTION ========================
        print("[INFO] Using 3-stage DPP view selection...")

        # 1. Precompute face normals (once per scene)
        face_normals = compute_face_normals(verts_np, faces)

        # 2. Compute CLIP embeddings for all candidate frames
        clip_image_paths = [scan_path / f"{stat['fid']}.color.jpg" for stat in image_stats]
        print(f"[INFO] Computing CLIP embeddings for {len(clip_image_paths)} frames...")
        clip_embeddings = compute_clip_embeddings(
            clip_image_paths,
            model_name=nbv_cfg.dpp_clip_model,
            device=nbv_cfg.dpp_clip_device,
        )

        # 3. Load camera poses for Stage 2/3 spatial constraints
        print("[INFO] Loading camera poses for spatial DPP stages...")
        camera_poses_dict = {}
        for stat in image_stats:
            fid = stat["fid"]
            pose_path = scan_path / f"{fid}.pose.txt"
            if pose_path.exists():
                camera_poses_dict[fid] = load_cam2world(pose_path)
        print(f"[INFO] Loaded {len(camera_poses_dict)} camera poses.")

        # 4. Run 3-stage DPP selection
        cluster_reps = dpp_select_views(
            image_stats,
            verts_np, faces, face_normals, face_obj_ids, clip_embeddings,
            total_views=nbv_cfg.dpp_total_views,
            seed_size=nbv_cfg.dpp_seed_size,
            camera_poses=camera_poses_dict,
            stage1_total_views=nbv_cfg.dpp_stage1_total_views,
            stage2_total_views=nbv_cfg.dpp_stage2_total_views,
            stage2_sigma_position=nbv_cfg.dpp_stage2_sigma_position,
            stage2_sigma_angle=nbv_cfg.dpp_stage2_sigma_angle,
            stage2_sigma_overlap=nbv_cfg.dpp_stage2_sigma_overlap,
            hard_max_overlap=nbv_cfg.dpp_hard_max_overlap,
            hard_min_position_distance=nbv_cfg.dpp_hard_min_position_distance,
            hard_min_angle_distance=nbv_cfg.dpp_hard_min_angle_distance,
        )
        print(f"[INFO] DPP selected {len(cluster_reps)} views.")

        if not cluster_reps:
            raise RuntimeError("No views selected by DPP — check config or input data.")

    else:
        # ======================== LEGACY: GREEDY + K-MEANS ========================
        # Load camera poses for spatial filtering (if enabled)
        camera_poses_dict = None
        if nbv_cfg.nbv_enable_pose_filtering:
            print("[INFO] Loading camera poses for spatial diversity filtering...")
            camera_poses_dict = {}
            for stat in image_stats:
                fid = stat["fid"]
                pose_path = scan_path / f"{fid}.pose.txt"
                if pose_path.exists():
                    camera_poses_dict[fid] = load_cam2world(pose_path)
            print(f"[INFO] Loaded {len(camera_poses_dict)} camera poses.")

        order_cache = cache_dir / f"{scene_id}.pth"
        if order_cache.exists():
            print(f"[INFO] Loading cached NBV order: {order_cache}")
            best_views = torch.load(order_cache)
        else:
            print("[INFO] Computing greedy next-best views...")
            best_views = greedy_next_best_views(
                image_stats,
                max_images=nbv_cfg.max_best,
                min_gain_pixels=nbv_cfg.min_gain_pixels,
                alpha=nbv_cfg.nbv_alpha,
                min_obj_pixels_for_presence=nbv_cfg.min_obj_pixels_for_presence,
                camera_poses=camera_poses_dict,
                min_position_distance=nbv_cfg.nbv_min_position_distance,
                min_angle_distance=nbv_cfg.nbv_min_angle_distance,
                enable_pose_filtering=nbv_cfg.nbv_enable_pose_filtering,
            )
            torch.save(best_views, order_cache)
            print(f"[INFO] Saved NBV order: {order_cache}")

        if not best_views:
            raise RuntimeError("No best views selected — check BRISQUE threshold or config.")

        # Adaptive K-means Clustering
        print("[INFO] Applying adaptive K-means clustering to selected camera poses...")
        camera_poses = [load_cam2world(scan_path / f"{fid}.pose.txt") for fid in best_views]
        n_candidates = len(best_views)

        if isinstance(nbv_cfg.kmeans_n_clusters, int) and nbv_cfg.kmeans_n_clusters > 0:
            n_clusters = min(n_candidates, nbv_cfg.kmeans_n_clusters)
        else:
            n_clusters = max(6, min(40, int(round(n_candidates / 12)))) if n_candidates >= 6 else n_candidates

        cluster_reps, cluster_labels = cluster_camera_poses(
            camera_poses, best_views, n_clusters
        )

        print(f"[INFO] Selected {len(cluster_reps)} cluster representatives.")

    # -------------------------- Save Outputs ----------------------------
    color_out = output_dir / "color"
    depth_out = output_dir / "depth"
    pose_out = output_dir / "pose"
    for d in (color_out, depth_out, pose_out): d.mkdir(parents=True, exist_ok=True)

    cam_json = {}
    for fid in cluster_reps:
        cam_json[fid] = load_cam2world(scan_path / f"{fid}.pose.txt").tolist()
        shutil.copy(scan_path / f"{fid}.color.jpg", color_out / f"{fid}.jpg")
        shutil.copy(scan_path / f"{fid}.depth.pgm", depth_out / f"{fid}.pgm")
        shutil.copy(scan_path / f"{fid}.pose.txt", pose_out / f"{fid}.txt")
        print(f"[INFO] Saved selected frame {fid}.")

    with open(output_dir / "camera_pose.json", "w") as f:
        json.dump(cam_json, f, indent=2)

    # ------------------------- Mask Rendering ----------------------------
    if save_instance_masks or save_semantic_masks:
        print("[INFO] Rendering masks for selected frames...")
        inst_dir = output_dir / "instance"
        sem_dir = output_dir / "semantic"
        if save_instance_masks: inst_dir.mkdir(parents=True, exist_ok=True)
        if save_semantic_masks: sem_dir.mkdir(parents=True, exist_ok=True)

        mask_ds = nbv_cfg.mask_downsample_factor
        Hm = max(1, H0 // max(1, mask_ds))
        Wm = max(1, W0 // max(1, mask_ds))
        fx_m, fy_m, cx_m, cy_m = fx / mask_ds, fy / mask_ds, cx / mask_ds, cy / mask_ds
        rasterizer_mask = make_rasterizer(Hm, Wm)
        face_obj_ids_np = np.asarray(face_obj_ids, dtype=np.int32)

        for fid in tqdm(cluster_reps, desc="Masks", dynamic_ncols=True):
            pose = load_cam2world(scan_path / f"{fid}.pose.txt")
            R_cv, t_cv = invert_se3_to_opencv(pose)
            cams = make_p3d_camera_from_opencv(
                R_cv, t_cv, fx_m, fy_m, cx_m, cy_m, Hm, Wm, device
            )
            pix_to_face, _ = rasterize_visibility(meshes, cams, rasterizer_mask)
            p2f_np = pix_to_face.cpu().numpy()

            if save_instance_masks:
                inst = pix_to_instance_mask(p2f_np, face_obj_ids_np, VOID_ID)
                save_png16(inst_dir / f"{fid}.png", inst)
            if save_semantic_masks:
                sem = pix_to_semantic_mask(p2f_np, face_obj_ids_np, obj_to_sem_id, VOID_ID)
                save_png16(sem_dir / f"{fid}.png", sem)

    # --------------------------- Auto Clean ------------------------------
    if auto_clean:
        print("[INFO] Auto-clean enabled, deleting raw frames...")
        for ext in ("*.color.jpg", "*.depth.pgm", "*.pose.txt"):
            for f in scan_path.glob(ext):
                f.unlink()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scene_id")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--save_semantic_masks", action="store_true")
    p.add_argument("--save_instance_masks", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--auto_clean", action="store_true")
    args = p.parse_args()
    main(args.scene_id, args.config,
         device_str=args.device,
         save_semantic_masks=args.save_semantic_masks,
         save_instance_masks=args.save_instance_masks,
         debug=args.debug,
         auto_clean=args.auto_clean)
