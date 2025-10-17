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
"""

import json, shutil, argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesUV
from pytorch3d.utils import cameras_from_opencv_projection

from src.utils.config_loader import load_config
from src.image_generation.scannetpp_best_views import (   # reuse ScanNet utilities
    invert_se3_to_opencv, per_face_object_ids, make_rasterizer,
    filter_sharp_images, rasterize_visibility, compute_image_visibility,
    greedy_next_best_views, pix_to_instance_mask, pix_to_semantic_mask, save_png16,
    compute_visible_objects, compute_spatial_relations,
)

from pytorch3d.utils import cameras_from_opencv_projection
import matplotlib.pyplot as plt

# -------------------------------- Constants -----------------------------------
# Semantic void ID (unlabeled pixels)
VOID_ID = 0

# ----------------------------- Loaders ---------------------------------

def load_mesh_with_textures(scene_path, device):
    import open3d as o3d
    import numpy as np, torch
    from PIL import Image
    from pytorch3d.renderer import TexturesUV
    from pytorch3d.structures import Meshes

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
    expanded_verts = verts[faces.reshape(-1)]
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
    return meshes


def load_segments_and_instances(scene_path: Path):
    segs_json = scene_path / "mesh.refined.0.010000.segs.v2.json"
    semseg_json = scene_path / "semseg.v2.json"
    if not segs_json.exists() or not semseg_json.exists():
        raise FileNotFoundError("Missing segs or semseg JSON")
    segs = json.loads(segs_json.read_text())
    groups = json.loads(semseg_json.read_text())["segGroups"]
    vert_seg = np.array(segs["segIndices"], dtype=np.int32)
    seg_to_obj, obj_to_label = {}, {}
    for g in groups:
        oid = int(g["objectId"])
        obj_to_label[oid] = g.get("label", "").strip().lower()
        for s in g["segments"]:
            seg_to_obj[int(s)] = oid
    return vert_seg, seg_to_obj, obj_to_label


def load_intrinsics_info(info_path: Path):
    lines = info_path.read_text().splitlines()
    K = None
    for L in lines:
        if L.startswith("m_calibrationColorIntrinsic"):
            vals = [float(x) for x in L.split("=")[1].split()]
            K = np.array(vals).reshape(4, 4)
    if K is None:
        raise RuntimeError("Could not parse intrinsics from _info.txt")
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def load_cam2world(pose_path: Path):
    return np.loadtxt(pose_path, dtype=np.float64).reshape(4, 4)


@torch.no_grad()
def _build_camera(R_cv, t_cv, fx, fy, cx, cy, H, W, device):
    return cameras_from_opencv_projection(
        R=torch.from_numpy(R_cv)[None].float().to(device),
        tvec=torch.from_numpy(t_cv)[None].float().to(device),
        camera_matrix=torch.tensor([[[fx, 0, cx],[0, fy, cy],[0,0,1]]], device=device).float(),
        image_size=torch.tensor([[H, W]], device=device).float()
    )

@torch.no_grad()
def _render_textured_rgb(meshes, cameras, H, W, device):
    """
    Render a textured 3RScan mesh using PyTorch3D (ambient-only lighting, no shading).
    Used for debug visualization.
    """
    from pytorch3d.renderer import (
        MeshRenderer, MeshRasterizer, HardPhongShader,
        PointLights, RasterizationSettings
    )

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
    rgb = np.array(Image.open(rgb_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(rgb); plt.title(title_left); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(rendered_rgb); plt.title(title_right); plt.axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=100); plt.close()

def _save_overlay(rgb_path: Path, pix_to_face: torch.Tensor, out_path: Path):
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
    # --- no more raise! ---

    # ---- orientation correction ----
    if vis.shape != rgb.shape[:2]:
        if vis.shape[::-1] == rgb.shape[:2]:
            vis = vis.T                      # ← your case (540,960) → (960,540)
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

    cfg = load_config(config_path)
    dataset_path = Path(cfg["paths"]["3rscan_dataset_path"])
    scan_path = dataset_path / scene_id

    rpp = cfg.get("3rscan", {})
    downsample = int(rpp.get("image_downsample_factor", 2))
    subsample_factor = int(rpp.get("subsample_factor", 5))
    faces_per_pixel = int(rpp.get("faces_per_pixel", 1))
    bin_size = rpp.get("bin_size", None)
    max_faces_per_bin = rpp.get("max_faces_per_bin", None)
    blur_radius = float(rpp.get("blur_radius", 0.0))
    limit_images = rpp.get("limit_images", None)
    max_best = rpp.get("max_best", None)
    min_gain_pixels = int(rpp.get("min_gain_pixels", 0))
    kmeans_n_clusters = int(rpp.get("kmeans_n_clusters", 10))
    imq_threshold = float(rpp.get("imq_threshold", 40.0))
    mask_ds = int(rpp.get("mask_downsample_factor", 1))

    output_dir = scan_path / rpp.get("output_folder", "output")
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------- Device Setup ----------------------------
    device = torch.device(device_str or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    # ---------------------- Load Mesh + Labels --------------------------
    meshes = load_mesh_with_textures(scan_path, device)

    faces = meshes.faces_packed().cpu().numpy()
    verts = meshes.verts_packed().cpu().numpy()

    vert_seg, seg_to_obj, obj_to_label = load_segments_and_instances(scan_path)

    faces_np = meshes.faces_packed().cpu().numpy()
    verts_np = meshes.verts_packed().cpu().numpy()

    max_idx = vert_seg.shape[0]
    valid_mask = (faces_np < max_idx).all(axis=1)
    if not valid_mask.all():
        print(f"[WARN] Removing {np.count_nonzero(~valid_mask)} invalid faces.")
    faces_np = faces_np[valid_mask]

    # rebuild filtered mesh
    meshes = Meshes(
        verts=[torch.tensor(verts_np, dtype=torch.float32, device=device)],
        faces=[torch.tensor(faces_np, dtype=torch.int64, device=device)],
        textures=meshes.textures,
    )

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

    frame_ids = filter_sharp_images(scan_path, threshold=imq_threshold)
    if not frame_ids:
        relax_seq = [imq_threshold + 10, 45, 50, 60]
        for thr in relax_seq:
            frame_ids = filter_sharp_images(scan_path, threshold=thr)
            if frame_ids:
                print(f"[WARN] Relaxed BRISQUE threshold to {thr}; kept {len(frame_ids)} frames.")
                break
    if not frame_ids:
        raise RuntimeError(f"No sharp images found even after relaxing threshold.")

    frame_ids = [fid.replace(".color", "") for fid in frame_ids]
    if subsample_factor > 1:
        frame_ids = frame_ids[::subsample_factor]
    if limit_images is not None:
        frame_ids = frame_ids[:int(limit_images)]

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

            cams_full = _build_camera(R_cv, t_cv, fx, fy, cx, cy, H0, W0, device)

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
    H_vis = max(1, H0 // max(1, downsample))
    W_vis = max(1, W0 // max(1, downsample))
    fx_vis, fy_vis, cx_vis, cy_vis = fx / downsample, fy / downsample, cx / downsample, cy / downsample

    cache_json = cache_dir / f"{scene_id}.json"
    if cache_json.exists():
        print(f"[INFO] Loading cached visibility stats: {cache_json}")
        image_stats = json.loads(cache_json.read_text())
    else:
        print(f"[INFO] Computing visibility for {len(frame_ids)} frames at {H_vis}x{W_vis}...")
        rasterizer_vis = make_rasterizer(H_vis, W_vis,
                                         faces_per_pixel=faces_per_pixel,
                                         bin_size=bin_size,
                                         max_faces_per_bin=max_faces_per_bin,
                                         blur_radius=blur_radius,
                                         )
        image_stats = []
        for fid in tqdm(frame_ids, desc="Visibility", dynamic_ncols=True):
            pose = load_cam2world(scan_path / f"{fid}.pose.txt")
            R_cv, t_cv = invert_se3_to_opencv(pose)

            cams = cameras_from_opencv_projection(
                R=torch.from_numpy(R_cv)[None].float().to(device),
                tvec=torch.from_numpy(t_cv)[None].float().to(device),
                camera_matrix=torch.tensor(
                    [[[fx_vis, 0, cx_vis],
                    [0, fy_vis, cy_vis],
                    [0, 0, 1]]], device=device
                ),
                image_size=torch.tensor([[H_vis, W_vis]], device=device),
            )
            pix_to_face, _ = rasterize_visibility(meshes, cams, rasterizer_vis)
            obj_px, total_px = compute_image_visibility(pix_to_face, face_obj_ids)

            visible_objects = compute_visible_objects(
                verts, faces, vert_seg, seg_to_obj, obj_to_label, 
                obj_px, face_obj_ids, pix_to_face,
                pose, R_cv, t_cv
            )
            spatial_relations = compute_spatial_relations(visible_objects)

            image_entry = {
                "fid": fid,
                "obj_pixels": obj_px,
                "total_labeled_px": int(total_px),
                "visible_objects": visible_objects,
                "spatial_relations": spatial_relations,
            }
            image_stats.append(image_entry)


        cache_json.write_text(json.dumps(image_stats, indent=2))
        print(f"[INFO] Saved visibility cache: {cache_json}")

    # ---------------------- Greedy NBV (Cached) --------------------------
    order_cache = cache_dir / f"{scene_id}.pth"
    if order_cache.exists():
        print(f"[INFO] Loading cached NBV order: {order_cache}")
        best_views = torch.load(order_cache)
    else:
        print("[INFO] Computing greedy next-best views...")
        best_views = greedy_next_best_views(
            image_stats, max_images=max_best, min_gain_pixels=min_gain_pixels
        )
        torch.save(best_views, order_cache)
        print(f"[INFO] Saved NBV order: {order_cache}")

    if not best_views:
        raise RuntimeError("No best views selected — check BRISQUE threshold or config.")

    # -------------------- Adaptive K-means Clustering --------------------
    print("[INFO] Applying adaptive K-means clustering to selected camera poses...")
    camera_poses = [load_cam2world(scan_path / f"{fid}.pose.txt") for fid in best_views]
    positions = np.stack([pose[:3, 3] for pose in camera_poses], axis=0)
    n_candidates = len(best_views)

    if isinstance(kmeans_n_clusters, int) and kmeans_n_clusters > 0:
        n_clusters = min(n_candidates, kmeans_n_clusters)
    else:
        n_clusters = max(6, min(40, int(round(n_candidates / 12)))) if n_candidates >= 6 else n_candidates

    if n_candidates <= 1 or n_clusters <= 1:
        print("[WARN] Not enough frames for meaningful clustering. Skipping clustering.")
        cluster_labels = np.zeros(n_candidates, dtype=int)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(positions)
        print(f"[INFO] K-means clustering into {n_clusters} clusters done.")

    rank = {fid: i for i, fid in enumerate(best_views)}
    clusters = {}
    for lbl, fid in zip(cluster_labels, best_views):
        clusters.setdefault(int(lbl), []).append(fid)

    cluster_reps = []
    for lbl, fids in clusters.items():
        best_fid = min(fids, key=lambda x: rank[x])
        cluster_reps.append(best_fid)
    cluster_reps.sort(key=lambda x: rank[x])

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
        print(f"[INFO] Saved cluster-representative frame {fid}.")

    with open(output_dir / "camera_pose.json", "w") as f:
        json.dump(cam_json, f, indent=2)

    # ------------------------- Mask Rendering ----------------------------
    if save_instance_masks or save_semantic_masks:
        print("[INFO] Rendering masks for selected frames...")
        inst_dir = output_dir / "instance"
        sem_dir = output_dir / "semantic"
        if save_instance_masks: inst_dir.mkdir(parents=True, exist_ok=True)
        if save_semantic_masks: sem_dir.mkdir(parents=True, exist_ok=True)

        Hm = max(1, H0 // max(1, mask_ds))
        Wm = max(1, W0 // max(1, mask_ds))
        fx_m, fy_m, cx_m, cy_m = fx / mask_ds, fy / mask_ds, cx / mask_ds, cy / mask_ds
        rasterizer_mask = make_rasterizer(Hm, Wm)
        face_obj_ids_np = np.asarray(face_obj_ids, dtype=np.int32)

        for fid in tqdm(cluster_reps, desc="Masks", dynamic_ncols=True):
            pose = load_cam2world(scan_path / f"{fid}.pose.txt")
            R_cv, t_cv = invert_se3_to_opencv(pose)
            cams = cameras_from_opencv_projection(
                R=torch.from_numpy(R_cv)[None].float().to(device),
                tvec=torch.from_numpy(t_cv)[None].float().to(device),
                camera_matrix=torch.tensor(
                    [[[fx_m, 0, cx_m],
                    [0, fy_m, cy_m],
                    [0, 0, 1]]], device=device
                ),
                image_size=torch.tensor([[Hm, Wm]], device=device),
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
