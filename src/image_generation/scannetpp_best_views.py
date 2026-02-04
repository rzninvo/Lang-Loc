"""
ScanNet++-style NBV selection + mask export (instance & semantic) + debug viz.

This script implements a compact pipeline for ScanNet scenes:

Pipeline
--------
1) Load one ScanNet scene: mesh (+ vertex colors), segmentation, aggregation,
   color frames, per-frame poses, and intrinsics.
2) Convert vertex-level segment ids to per-face object ids via majority vote.
3) Rasterize a *downsampled* visibility pass over a subset of frames and count,
   per image, how many pixels (front-most face only) belong to each object id.
4) Greedy "next-best-view" selection that balances coverage across instances.
5) Cache the per-image stats and the selected list; write a summary CSV; copy
   selected RGB/poses to a folder.
6) For the selected frames, re-rasterize (usually at full resolution) and save:
   - Instance mask as 16-bit PNG (pixel value = ScanNet objectId)
   - Semantic mask as 16-bit PNG (pixel value = semanticId from TSV)

Debug Visualization
-------------------
If --debug is enabled, for a few frames we show the dataset RGB (left) and
a PyTorch3D render from the exact same camera (right), using the vertex colors
stored in the *_vh_clean_2.ply mesh. This is a good sanity check that the
camera model and pose parsing line up in pixel space.

Key Design Choice
-----------------
We build the PyTorch3D camera using the official OpenCV bridge:
    cameras_from_opencv_projection(R, tvec, K, image_size=(H,W))
This removes the need for manual axis flips; you pass in OpenCV world->cam
(R, t) and the pixel intrinsics, and the bridge returns a PerspectiveCameras
that renders in *the same* pixel coordinates as your RGBs.

Author: Roham Zendehdel Nobari (rzendehdel@ethz.ch)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import argparse

# PyTorch3D
from pytorch3d.renderer import (
    MeshRenderer, HardPhongShader, PointLights
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.renderer import Textures

# Project imports - shared NBV pipeline components
from src.image_generation.nbv_pipeline import (
    VOID_ID,
    filter_quality_images,
    make_p3d_camera_from_opencv,
    make_rasterizer,
    per_face_object_ids,
    rasterize_visibility,
    compute_image_visibility,
    compute_visible_objects,
    compute_spatial_relations,
    greedy_next_best_views,
    pix_to_instance_mask,
    pix_to_semantic_mask,
    save_png16,
    cluster_camera_poses,
)
from src.utils.camera_utils import (
    load_cam2world,
    invert_se3_to_opencv,
    load_intrinsics_txt,
)
from src.utils.config_loader import load_config
from src.utils.nbv_config import NBVConfig, extract_nbv_config


# ----------------------------- I/O & Utilities --------------------------------


def load_mesh_with_vertex_colors(scene_path: Path, scene_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the *_vh_clean_2.ply mesh (vertices, faces, vertex colors).

    Args:
        scene_path: Root path of the scene (e.g., .../scene0000_00).
        scene_id: Scene id string ('scene0000_00').

    Returns:
        V: (Nv, 3) float32 vertices in world units.
        F: (Nf, 3) int64 face indices.
        VC: (Nv, 3) float32 vertex colors; in [0,1] if stored that way in PLY,
            else we normalize later.
    """
    ply = scene_path / f"{scene_id}_vh_clean_2.ply"
    if not ply.exists():
        raise FileNotFoundError(f"Mesh not found: {ply}")
    mesh = o3d.io.read_triangle_mesh(str(ply))
    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.triangles, dtype=np.int64)
    VC = np.asarray(mesh.vertex_colors, dtype=np.float32) if mesh.has_vertex_colors() else np.ones_like(V, dtype=np.float32) * 0.7
    return V, F, VC


def load_segments_and_instances(scene_path: Path, scene_id: str) -> Tuple[np.ndarray, Dict[int, int], Dict[int, str]]:
    """
    Load segmentation + aggregation JSON and build mappings.

    - segs["segIndices"] provides a per-vertex segment id (same order as PLY).
    - aggregation groups segments into objects (instances) and provides labels.

    Args:
        scene_path: Root path of the scene.
        scene_id: Scene id string.

    Returns:
        vert_seg:   (Nv,) int32 segment id per vertex.
        seg_to_obj: dict mapping segment id -> object id (instance).
        obj_to_label: dict mapping object id -> raw label string (lowercased).
    """
    segs_json = scene_path / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    if not segs_json.exists():
        segs_json = scene_path / f"{scene_id}_vh_clean_2.segs.json"
    if not segs_json.exists():
        raise FileNotFoundError(f"Segs JSON not found: {segs_json}")

    agg_json = scene_path / f"{scene_id}.aggregation.json"
    if not agg_json.exists():
        raise FileNotFoundError(f"Aggregation JSON not found: {agg_json}")

    segs = json.loads(segs_json.read_text())
    agg = json.loads(agg_json.read_text())

    vert_seg = np.array(segs["segIndices"], dtype=np.int32)

    seg_to_obj: Dict[int, int] = {}
    obj_to_label: Dict[int, str] = {}
    for g in agg["segGroups"]:
        oid = int(g["objectId"])
        obj_to_label[oid] = g.get("label", "").strip().lower()
        for s in g["segments"]:
            seg_to_obj[int(s)] = oid

    return vert_seg, seg_to_obj, obj_to_label


# ---------------------------- Semantic label map ------------------------------

def load_scannet_labelmap_tsv(tsv_path: Path, semantic_id_key: str = "nyu40id") -> Dict[str, int]:
    """
    Parse 'scannetv2-labels.combined.tsv' and map raw labels to semantic ids.

    Args:
        tsv_path: Path to the combined TSV file.
        semantic_id_key: Which column to use for semantic ids (e.g., 'nyu40id').

    Returns:
        labelmap: dict raw_label_lowercase -> semantic_id (int).
    """
    if not tsv_path.exists():
        raise FileNotFoundError(f"Label map TSV not found: {tsv_path}")

    lines = tsv_path.read_text().splitlines()
    header = [h.strip() for h in lines[0].split("\t")]
    rows = [dict(zip(header, [c.strip() for c in L.split("\t")])) for L in lines[1:]]

    # Heuristically pick a column that holds the "raw" label string (robust to TSV variants).
    name_candidates = ["raw_category", "raw_name", "raw_label", "name", "raw_category_0"]
    raw_col = next((k for k in name_candidates if k in header), None)
    if raw_col is None:
        for h in header:
            hlow = h.lower()
            if ("raw" in hlow) and ("name" in hlow or "category" in hlow):
                raw_col = h
                break
    if raw_col is None:
        raise ValueError(f"Could not find a raw label column in TSV header: {header}")

    if semantic_id_key not in header:
        raise ValueError(f"semantic_id_key='{semantic_id_key}' not in TSV header: {header}")

    labelmap: Dict[str, int] = {}
    for r in rows:
        raw = (r.get(raw_col) or "").strip().lower()
        sid = (r.get(semantic_id_key) or "").strip()
        if not raw or not sid:
            continue
        try:
            labelmap[raw] = int(float(sid))
        except ValueError:
            continue
    return labelmap


# -------------------------- Debug visualization helper ------------------------

@torch.no_grad()
def render_rgb_with_camera(V, F, VC, cameras, H: int, W: int, device):
    """
    Render the mesh with vertex colors from the given camera.

    This is used for quick side-by-side debugging (RGB vs render).

    Args:
        V, F, VC: mesh vertices, faces, vertex colors.
        cameras: PerspectiveCameras built from the OpenCV bridge.
        H, W: image size in pixels.
        device: CUDA/CPU device.

    Returns:
        A float32 RGB image in [0,1] as np.ndarray with shape (H, W, 3).
    """
    verts = torch.from_numpy(V).float().to(device)
    faces = torch.from_numpy(F).long().to(device)

    vc = VC.astype(np.float32)
    if vc.max() > 1.0:
        vc = vc / 255.0  # normalize if colors were stored 0..255
    verts_rgb = torch.from_numpy(vc).float().unsqueeze(0).to(device)

    mesh = Meshes(verts=[verts], faces=[faces], textures=TexturesVertex(verts_features=verts_rgb))
    rasterizer = make_rasterizer(H, W, faces_per_pixel=1)

    # Ambient-only lighting: color == vertex color (diffuse/specular off).
    lights = PointLights(
        device=device,
        ambient_color=((1.0, 1.0, 1.0),),
        diffuse_color=((0.0, 0.0, 0.0),),
        specular_color=((0.0, 0.0, 0.0),),
        location=[[0.0, 0.0, 0.0]],
    )
    shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    img = renderer(mesh, cameras=cameras)[0, ..., :3].clamp(0, 1).cpu().numpy()
    return img


def show_side_by_side(fid: str, rgb_path: Path, rendered: np.ndarray):
    """
    Utility to display dataset RGB (left) vs. PyTorch3D render (right).

    Args:
        fid: frame id string for the title.
        rgb_path: path to the RGB image file.
        rendered: rendered RGB image (H,W,3) in [0,1].
    """
    rgb = np.array(Image.open(rgb_path))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(rgb); plt.title(f"RGB {fid}"); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(rendered); plt.title("PyTorch3D render"); plt.axis("off")
    plt.tight_layout(); plt.show()

def plot_clusters(camera_poses: List[np.ndarray], cluster_labels: List[int]):
    """
    Plot the camera poses in 3D with color mapping for each cluster.
    """
    # Extract positions (translation vectors) from the camera poses
    positions = np.stack([pose[:3, 3] for pose in camera_poses], axis=0)  # (N,3)

    # Plot the clusters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=cluster_labels, cmap='jet', s=50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Clustered Camera Poses')

    plt.colorbar(scatter)
    plt.show()

# ----------------------------------- Main -------------------------------------

def main(scene_id: str, config_path: str, device_str: str | None = None,
         debug: bool = False, viz_first: int = 3, viz_selected: int = 0, auto_clean: bool = False, save_semantic_masks: bool = False, save_instance_masks: bool = False) -> None:
    """
    End-to-end pipeline orchestrator.

    It loads all assets, builds caches, performs NBV selection, copies selected
    frames, optionally visualizes, and finally exports instance/semantic masks.

    Args:
        scene_id: ScanNet scene id, e.g., 'scene0000_00'.
        config_path: path to a YAML with:
            paths.dataset_path: root folder containing ScanNet scenes.
            scannetpp.*: pipeline knobs (see code).
        device_str: override torch device string ('cuda:0', 'cpu', ...).
        debug: show RGB vs render for a few frames (sanity check).
        viz_first: how many initial frames to visualize.
        viz_selected: how many top selected frames to visualize.
    """
    cfg = load_config(config_path)
    nbv_cfg = extract_nbv_config(cfg, dataset="scannetpp")

    # Paths
    dataset_path = Path(cfg["paths"]["scannet_dataset_path"])
    scan_path = dataset_path / scene_id
    intrinsic_path = scan_path / "intrinsic" / "intrinsic_color.txt"
    color_dir = scan_path / "color"
    pose_dir = scan_path / "pose"
    depth_dir = scan_path / "depth"
    label_dir = scan_path / "label"
    output_dir = scan_path / "output"
    sens_file = scan_path / f"{scene_id}.sens"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output / cache directories
    cache_dir = output_dir / nbv_cfg.cache_dir
    out_dir = output_dir / nbv_cfg.raster_out_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device(device_str) if device_str else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Collect frames using IQA filtering
    iqa_metric = nbv_cfg.iqa_metric
    iqa_threshold = nbv_cfg.iqa_threshold
    iqa_device = nbv_cfg.iqa_device

    frame_ids = filter_quality_images(
        color_dir,
        metric_name=iqa_metric,
        threshold=iqa_threshold,
        device=iqa_device,
    )

    if nbv_cfg.subsample_factor > 1:
        frame_ids = frame_ids[::nbv_cfg.subsample_factor]
    if nbv_cfg.limit_images is not None:
        frame_ids = frame_ids[:int(nbv_cfg.limit_images)]

    if not frame_ids:
        print(f"[WARN] No frames passed quality threshold {iqa_threshold} for {iqa_metric}")
        # For QualiCLIP: try relaxing by decreasing threshold (more permissive)
        # Adjust these values based on your quality requirements
        relax_seq = [iqa_threshold - 0.05, iqa_threshold - 0.10, iqa_threshold - 0.15, max(0.0, iqa_threshold - 0.20)]

        for thr in relax_seq:
            frame_ids = filter_quality_images(
                color_dir,
                metric_name=iqa_metric,
                threshold=thr,
                device=iqa_device,
            )
            if frame_ids:
                print(f"[WARN] Relaxed {iqa_metric} threshold to {thr:.4f}; kept {len(frame_ids)} frames.")
                break

    if not frame_ids:
        raise RuntimeError(f"No quality images found in {color_dir} even after relaxing threshold.")
    if debug:
        print(f"[DEBUG] Using {len(frame_ids)} sharp frames after subsample/limit.")

    # Intrinsics
    fx, fy, cx, cy = load_intrinsics_txt(intrinsic_path)
    if debug:
        print(f"[DEBUG] Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # Mesh (+ vertex colors) and labels
    V, F, VC = load_mesh_with_vertex_colors(scan_path, scene_id)
    if debug:
        print(f"[DEBUG] Mesh: {V.shape[0]} vertices, {F.shape[0]} faces; colors: {VC.shape}")
    vert_seg, seg_to_obj, obj_to_label = load_segments_and_instances(scan_path, scene_id)
    face_obj_ids = per_face_object_ids(F, vert_seg, seg_to_obj)
    unique_face_objs = np.unique(face_obj_ids[face_obj_ids >= 0])
    if debug:
        print(f"[DEBUG] face_obj_ids: {len(unique_face_objs)} unique objects (non-void).")

    # Semantic label map (objectId -> semanticId)
    labelmap = load_scannet_labelmap_tsv(nbv_cfg.labelmap_tsv, semantic_id_key=nbv_cfg.semantic_id_key)
    obj_to_sem_id: Dict[int, int] = {oid: int(labelmap.get(lbl, VOID_ID)) for oid, lbl in obj_to_label.items()}

    # PyTorch3D mesh for visibility
    verts = torch.from_numpy(V).float().to(device)
    faces = torch.from_numpy(F).long().to(device)
    textures_white = Textures(verts_rgb=torch.ones_like(verts)[None, ...])  # visibility only
    meshes = Meshes(verts=[verts], faces=[faces], textures=textures_white)

    # Base image sizes
    sample_img = np.array(Image.open(color_dir / f"{frame_ids[0]}.jpg"))
    H0, W0 = map(int, sample_img.shape[:2])

    # ------------------------------- DEBUG VIZ --------------------------------
    if debug and viz_first > 0:
        print(f"[DEBUG] Visualizing first {viz_first} frame(s) (RGB vs PyTorch3D render)...")
        for fid in frame_ids[:viz_first]:
            pose = load_cam2world(pose_dir / f"{fid}.txt")
            R_cv, t_cv = invert_se3_to_opencv(pose)  # world->cam (OpenCV)

            # Full-res camera matches the RGB size.
            cams_dbg = make_p3d_camera_from_opencv(R_cv, t_cv, fx, fy, cx, cy, H0, W0, device)

            rendered = render_rgb_with_camera(V, F, VC, cams_dbg, H0, W0, device)
            show_side_by_side(fid, color_dir / f"{fid}.jpg", rendered)

    # ----------------------- VISIBILITY (DOWNSAMPLED) -------------------------
    downsample = nbv_cfg.image_downsample_factor
    H_vis = max(1, H0 // max(1, downsample))
    W_vis = max(1, W0 // max(1, downsample))
    fx_vis, fy_vis, cx_vis, cy_vis = fx / downsample, fy / downsample, cx / downsample, cy / downsample

    rasterizer_vis = make_rasterizer(
        H_vis, W_vis,
        faces_per_pixel=nbv_cfg.faces_per_pixel,
        bin_size=nbv_cfg.bin_size,
        max_faces_per_bin=nbv_cfg.max_faces_per_bin,
        blur_radius=nbv_cfg.blur_radius,
    )

    cache_json = cache_dir / f"{scene_id}.json"
    if cache_json.exists():
        print(f"[INFO] Loading visibility cache: {cache_json}")
        image_stats = json.loads(cache_json.read_text())
    else:
        print(f"[INFO] Computing visibility for {len(frame_ids)} frames at {H_vis}x{W_vis}...")
        image_stats: List[Dict] = []

        for idx, fid in enumerate(tqdm(frame_ids)):
            pose = load_cam2world(pose_dir / f"{fid}.txt")
            R_cv, t_cv = invert_se3_to_opencv(pose)

            cams_vis = make_p3d_camera_from_opencv(
                R_cv, t_cv, fx_vis, fy_vis, cx_vis, cy_vis, H_vis, W_vis, device
            )
            pix_to_face, _ = rasterize_visibility(meshes, cams_vis, rasterizer_vis)
            obj_px, total_px = compute_image_visibility(pix_to_face, face_obj_ids)

            visible_objects = compute_visible_objects(
                V, F, vert_seg, seg_to_obj, obj_to_label,
                obj_px, face_obj_ids, pix_to_face,
                pose, R_cv, t_cv,
                fx_vis, fy_vis, cx_vis, cy_vis,
                W_vis, H_vis,
                fov_depth_clip=nbv_cfg.fov_depth_clip,
                coverage_threshold=nbv_cfg.coverage_threshold,
                min_pixel_count=nbv_cfg.min_pixel_count,
            )
            spatial_relations = compute_spatial_relations(
                visible_objects,
                max_distance=nbv_cfg.spatial_max_distance,
                size_ratio_threshold=nbv_cfg.spatial_size_ratio_threshold,
                eps=nbv_cfg.spatial_eps,
            )

            image_entry = {
                "fid": fid,
                "obj_pixels": obj_px,
                "total_labeled_px": int(total_px),
                "visible_objects": visible_objects,
                "spatial_relations": spatial_relations,
            }
            image_stats.append(image_entry)

            if idx < 5 and debug:
                print(f"[DEBUG] Frame {fid}: {len(visible_objects)} objs, {len(spatial_relations)} relations")

        cache_json.write_text(json.dumps(image_stats, indent=2))
        print(f"[INFO] Saved visibility cache: {cache_json}")

    # ------------------------ GREEDY NEXT-BEST-VIEWS --------------------------
    # Load camera poses for spatial filtering (if enabled)
    camera_poses_dict = None
    if nbv_cfg.nbv_enable_pose_filtering:
        print("[INFO] Loading camera poses for spatial diversity filtering...")
        camera_poses_dict = {}
        for stat in image_stats:
            fid = stat["fid"]
            pose_path = pose_dir / f"{fid}.txt"
            if pose_path.exists():
                camera_poses_dict[fid] = load_cam2world(pose_path)
        print(f"[INFO] Loaded {len(camera_poses_dict)} camera poses.")

    order_cache = cache_dir / f"{scene_id}.pth"
    if order_cache.exists():
        print(f"[INFO] Loading best-views order: {order_cache}")
        best_views: List[str] = torch.load(order_cache)
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
        print(f"[INFO] Saved best-views order: {order_cache}")

   # --------------------------- K-means Clustering -----------------------------
    print("[INFO] Applying K-means clustering to selected camera poses...")

    # Build pose list for only the NBV-ordered frames
    camera_poses = []
    for fid in best_views:
        pose = load_cam2world(pose_dir / f"{fid}.txt")
        camera_poses.append(pose)

    # Choose number of clusters robustly
    n_candidates = len(best_views)
    if isinstance(nbv_cfg.kmeans_n_clusters, int) and nbv_cfg.kmeans_n_clusters > 0:
        n_clusters = min(n_candidates, nbv_cfg.kmeans_n_clusters)
    else:
        # fallback: about one per ~12 views, clamped
        n_clusters = max(6, min(40, int(round(n_candidates / 12)))) if n_candidates >= 6 else n_candidates

    cluster_representatives, cluster_labels = cluster_camera_poses(
        camera_poses, best_views, n_clusters
    )

    print(f"[INFO] Selected {len(cluster_representatives)} cluster representatives.")

    # --------------------------- Save selected frames ----------------------------
    color_output_dir = output_dir / "color"
    pose_output_dir = output_dir / "pose"
    depth_output_dir = output_dir / "depth"
    label_output_dir = output_dir / "label"
    for d in (color_output_dir, pose_output_dir, depth_output_dir, label_output_dir):
        d.mkdir(parents=True, exist_ok=True)

    camera_pose_json = {}
    for fid in cluster_representatives:
        pose = load_cam2world(pose_dir / f"{fid}.txt")
        camera_pose_json[fid] = pose.tolist()

        shutil.copy(color_dir / f"{fid}.jpg",  color_output_dir / f"{fid}.jpg")
        shutil.copy(depth_dir / f"{fid}.png",  depth_output_dir / f"{fid}.png")
        shutil.copy(pose_dir  / f"{fid}.txt",  pose_output_dir / f"{fid}.txt")
        lp = label_dir / f"{fid}.png"
        if lp.exists():
            shutil.copy(lp, label_output_dir / f"{fid}.png")
        print(f"[INFO] Saved cluster-representative frame {fid} to {output_dir}")

    with open(output_dir / "camera_pose.json", "w") as f:
        json.dump(camera_pose_json, f, indent=2)

    # --------------------------- (Optional) Debug plot ---------------------------
    if debug:
        positions = np.stack([pose[:3, 3] for pose in camera_poses], axis=0)
        rank = {fid: i for i, fid in enumerate(best_views)}

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all candidates, colored by cluster
        for lbl in np.unique(cluster_labels):
            idx = np.where(cluster_labels == lbl)[0]
            ax.scatter(positions[idx, 0], positions[idx, 1], positions[idx, 2], label=f"C{int(lbl)}", s=30)

        # Highlight chosen representatives (larger markers)
        rep_idx = [best_views.index(fid) for fid in cluster_representatives]
        ax.scatter(positions[rep_idx, 0], positions[rep_idx, 1], positions[rep_idx, 2], s=120, marker='*')

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title('Clustered Camera Poses (+ reps)')
        ax.legend()
        plt.show()

    # --------------------------- Export instance/semantic masks -------------------
    if save_instance_masks or save_semantic_masks:
        print("[INFO] Rendering masks for selected frames...")
        # Build image size / intrinsics for mask resolution
        mask_ds = nbv_cfg.mask_downsample_factor
        Hm = max(1, H0 // max(1, mask_ds))
        Wm = max(1, W0 // max(1, mask_ds))
        fx_m, fy_m, cx_m, cy_m = fx / mask_ds, fy / mask_ds, cx / mask_ds, cy / mask_ds

        rasterizer_mask = make_rasterizer(
            Hm, Wm,
            faces_per_pixel=1,
            bin_size=nbv_cfg.bin_size,
            max_faces_per_bin=nbv_cfg.max_faces_per_bin,
            blur_radius=0.0,
        )

        inst_dir = output_dir / "instance"
        sem_dir  = output_dir / "semantic"
        if save_instance_masks: inst_dir.mkdir(parents=True, exist_ok=True)
        if save_semantic_masks: sem_dir.mkdir(parents=True, exist_ok=True)

        # numpy copy of face->object ids for fast indexing
        face_obj_ids_np = np.asarray(face_obj_ids, dtype=np.int32)

        for fid in tqdm(cluster_representatives, desc="Masks", dynamic_ncols=True):
            pose = load_cam2world(pose_dir / f"{fid}.txt")
            R_cv, t_cv = invert_se3_to_opencv(pose)

            cams_mask = make_p3d_camera_from_opencv(R_cv, t_cv, fx_m, fy_m, cx_m, cy_m, Hm, Wm, device)
            pix_to_face, _ = rasterize_visibility(meshes, cams_mask, rasterizer_mask)
            p2f_np = pix_to_face.cpu().numpy()

            if save_instance_masks:
                inst = pix_to_instance_mask(p2f_np, face_obj_ids_np, void_val=VOID_ID)
                save_png16(inst_dir / f"{fid}.png", inst)

            if save_semantic_masks:
                sem = pix_to_semantic_mask(p2f_np, face_obj_ids_np, obj_to_sem_id, void_val=VOID_ID)
                save_png16(sem_dir / f"{fid}.png", sem)

    # Auto-clean if enabled
    if auto_clean:
        print("\n[INFO] Auto-clean enabled. Deleting intermediate files...")
        if sens_file.exists():
            sens_file.unlink()
            print(f"Deleted: {sens_file}")
        for folder in [color_dir, depth_dir, pose_dir, label_dir]:
            if folder.exists():
                shutil.rmtree(folder)
                print(f"Deleted folder: {folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScanNet++ NBV + masks + debug viz")
    parser.add_argument("scene_id", type=str, help="Scene ID (e.g., scene0000_00)")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="torch device (e.g., cuda:0 or cpu)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: Print extra info and visualize some preprocessing steps")
    parser.add_argument("--viz_first", type=int, default=3, help="Visualize this many initial frames")
    parser.add_argument("--viz_selected", type=int, default=0, help="Visualize this many top selected frames")
    parser.add_argument("--auto_clean", action="store_true", help="Auto-clean intermediate files (color/depth/pose/label dirs and .sens file)")
    parser.add_argument("--save_semantic_masks", action="store_true", help="Save semantic masks (16-bit PNGs)")
    parser.add_argument("--save_instance_masks", action="store_true", help="Save instance masks (16-bit PNGs)")
    args = parser.parse_args()

    main(
        args.scene_id,
        args.config,
        device_str=args.device,
        debug=args.debug,
        viz_first=args.viz_first,
        viz_selected=args.viz_selected,
        auto_clean=args.auto_clean,
        save_semantic_masks=args.save_semantic_masks,
        save_instance_masks=args.save_instance_masks,
    )
