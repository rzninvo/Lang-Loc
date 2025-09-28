#!/usr/bin/env python3
"""
3RScan-style NBV selection + mask export (instance & semantic) + debug viz.

Pipeline:
1) Load 3RScan scene: mesh (+ vertex colors from OBJ+MTL+PNG), segmentation,
   semseg.json groups, color frames, poses, and intrinsics.
2) Filter frames by BRISQUE quality & subsample.
3) Rasterize visibility for candidate frames; compute per-object pixel counts.
4) Greedy NBV selection + K-means clustering for diversity.
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
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.utils import cameras_from_opencv_projection

from src.utils.config_loader import load_config
from src.image_generation.scannetpp_best_views import (   # reuse ScanNet utilities
    invert_se3_to_opencv, per_face_object_ids, make_rasterizer,
    filter_sharp_images, rasterize_visibility, compute_image_visibility,
    greedy_next_best_views, pix_to_instance_mask, pix_to_semantic_mask, save_png16
)

VOID_ID = 0

# ----------------------------- Loaders ---------------------------------

def load_mesh_with_vertex_colors(scene_path: Path):
    obj = scene_path / "mesh.refined.v2.obj"
    if not obj.exists():
        raise FileNotFoundError(obj)
    mesh = o3d.io.read_triangle_mesh(str(obj), enable_post_processing=True)
    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.triangles, dtype=np.int64)
    VC = np.asarray(mesh.vertex_colors, dtype=np.float32) if mesh.has_vertex_colors() \
         else np.ones_like(V, dtype=np.float32) * 0.7
    return V, F, VC

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
    return float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])

def load_cam2world(pose_path: Path):
    return np.loadtxt(pose_path, dtype=np.float64).reshape(4,4)

# ------------------------------- Main ----------------------------------

def main(scene_id: str, config_path: str, device_str=None,
         debug=False, auto_clean=False,
         save_semantic_masks=False, save_instance_masks=False):

    cfg = load_config(config_path)
    dataset_path = Path(cfg["paths"]["base_data_dir"]) / "3RScan"
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
    imq_threshold = float(rpp.get("imq_threshold", 40.0))  # BRISQUE
    mask_ds = int(rpp.get("mask_downsample_factor", 1))

    output_dir = scan_path / rpp.get("output_folder", "output")
    output_dir.mkdir(parents=True, exist_ok=True)


    # Load mesh + labels
    V, F, VC = load_mesh_with_vertex_colors(scan_path)
    vert_seg, seg_to_obj, obj_to_label = load_segments_and_instances(scan_path)
    max_idx = vert_seg.shape[0]
    if F.max() >= max_idx:
        print(f"[WARN] Face indices go up to {F.max()}, but vert_seg has only {max_idx} entries.")
        # Drop invalid faces
        valid_mask = (F < max_idx).all(axis=1)
        F = F[valid_mask]
        print(f"[INFO] Filtered faces: kept {len(F)} / {len(valid_mask)}")

    face_obj_ids = per_face_object_ids(F, vert_seg, seg_to_obj)
    obj_to_sem_id = {oid: idx+1 for idx, oid in enumerate(obj_to_label.keys())}

    # Build mesh for visibility
    device = torch.device(device_str or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    verts = torch.from_numpy(V).float().to(device)
    faces = torch.from_numpy(F).long().to(device)
    meshes = Meshes(
        verts=[verts],
        faces=[faces],
        textures=TexturesVertex(
            verts_features=torch.from_numpy(VC).float().unsqueeze(0).to(device)
        )
    )

    # Intrinsics
    fx, fy, cx, cy = load_intrinsics_info(scan_path / "_info.txt")

    # Collect & filter frames
    frame_ids = [
        p.name.replace(".color.jpg", "")
        for p in sorted(scan_path.glob("frame-*.color.jpg"))
    ]
    if not frame_ids:
        raise RuntimeError("No frames found")
    H0,W0 = np.array(Image.open(scan_path / f"{frame_ids[0]}.color.jpg")).shape[:2]

    frame_ids = filter_sharp_images(scan_path, threshold=imq_threshold)  # BRISQUE
    frame_ids = [fid.replace(".color", "") for fid in frame_ids]
    if subsample_factor > 1:
        frame_ids = frame_ids[::subsample_factor]

    # Visibility pass
    H_vis = max(1, H0 // max(1, downsample))
    W_vis = max(1, W0 // max(1, downsample))

    rasterizer_vis = make_rasterizer(
        H_vis, W_vis,
        faces_per_pixel=faces_per_pixel,
        bin_size=bin_size,
        max_faces_per_bin=max_faces_per_bin,
        blur_radius=blur_radius,
    )
    image_stats = []
    for fid in tqdm(frame_ids, desc="Visibility"):
        pose = load_cam2world(scan_path / f"{fid}.pose.txt")
        R_cv, t_cv = invert_se3_to_opencv(pose)
        cams = cameras_from_opencv_projection(
            R=torch.from_numpy(R_cv)[None].float().to(device),
            tvec=torch.from_numpy(t_cv)[None].float().to(device),
            camera_matrix=torch.tensor([[[fx,0,cx],[0,fy,cy],[0,0,1]]],device=device),
            image_size=torch.tensor([[H_vis,W_vis]],device=device)
        )
        pix_to_face,_ = rasterize_visibility(meshes,cams,rasterizer_vis)
        obj_px, total_px = compute_image_visibility(pix_to_face, face_obj_ids)
        image_stats.append({"fid": fid, "obj_pixels": obj_px, "total_labeled_px": total_px})

    best_views = greedy_next_best_views(
        image_stats,
        max_images=max_best,
        min_gain_pixels=min_gain_pixels
    )

    if not best_views:
        raise RuntimeError("No best views selected — check BRISQUE threshold or config.")

    # ------------------- K-means clustering -------------------
    poses = [load_cam2world(scan_path / f"{fid}.pose.txt") for fid in best_views]
    positions = np.stack([p[:3,3] for p in poses], axis=0)
    n_clusters = min(len(best_views), kmeans_n_clusters)
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(positions)
    else:
        cluster_labels = np.zeros(len(best_views), dtype=int)

    rank = {fid:i for i,fid in enumerate(best_views)}
    clusters = {}
    for lbl,fid in zip(cluster_labels,best_views):
        clusters.setdefault(int(lbl), []).append(fid)

    cluster_reps = []
    for lbl,fids in clusters.items():
        best_fid = min(fids, key=lambda x: rank[x])
        cluster_reps.append(best_fid)
    cluster_reps.sort(key=lambda x: rank[x])

    # Save selected frames
    color_out = output_dir/"color"; depth_out=output_dir/"depth"; pose_out=output_dir/"pose"
    for d in (color_out, depth_out, pose_out): d.mkdir(parents=True, exist_ok=True)
    cam_json={}
    for fid in cluster_reps:
        cam_json[fid] = load_cam2world(scan_path/f"{fid}.pose.txt").tolist()
        shutil.copy(scan_path/f"{fid}.color.jpg", color_out/f"{fid}.jpg")
        shutil.copy(scan_path/f"{fid}.depth.pgm", depth_out/f"{fid}.pgm")
        shutil.copy(scan_path/f"{fid}.pose.txt",  pose_out/f"{fid}.txt")
    with open(output_dir/"camera_pose.json","w") as f: json.dump(cam_json,f,indent=2)

    # Masks
    if save_instance_masks or save_semantic_masks:
        inst_dir=output_dir/"instance"; sem_dir=output_dir/"semantic"
        if save_instance_masks: inst_dir.mkdir(exist_ok=True)
        if save_semantic_masks: sem_dir.mkdir(exist_ok=True)
        Hm = max(1, H0 // max(1, mask_ds))
        Wm = max(1, W0 // max(1, mask_ds))
        rasterizer_mask = make_rasterizer(Hm, Wm)
        for fid in tqdm(cluster_reps, desc="Masks"):
            pose=load_cam2world(scan_path/f"{fid}.pose.txt")
            R_cv,t_cv=invert_se3_to_opencv(pose)
            cams=cameras_from_opencv_projection(
                R=torch.from_numpy(R_cv)[None].float().to(device),
                tvec=torch.from_numpy(t_cv)[None].float().to(device),
                camera_matrix=torch.tensor([[[fx,0,cx],[0,fy,cy],[0,0,1]]],device=device),
                image_size=torch.tensor([[Hm,Wm]],device=device)
            )
            pix_to_face,_=rasterize_visibility(meshes,cams,rasterizer_mask)
            p2f_np=pix_to_face.cpu().numpy()
            if save_instance_masks:
                inst=pix_to_instance_mask(p2f_np, face_obj_ids, VOID_ID)
                save_png16(inst_dir/f"{fid}.png", inst)
            if save_semantic_masks:
                sem=pix_to_semantic_mask(p2f_np, face_obj_ids, obj_to_sem_id, VOID_ID)
                save_png16(sem_dir/f"{fid}.png", sem)

    # Auto-clean
    if auto_clean:
        print("[INFO] Auto-clean enabled, deleting raw frames...")
        for ext in ("*.color.jpg","*.depth.pgm","*.pose.txt"):
            for f in scan_path.glob(ext):
                f.unlink()

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("scene_id")
    p.add_argument("--config",default="config/default.yaml")
    p.add_argument("--device",default=None)
    p.add_argument("--save_semantic_masks",action="store_true")
    p.add_argument("--save_instance_masks",action="store_true")
    p.add_argument("--debug",action="store_true")
    p.add_argument("--auto_clean",action="store_true")
    args=p.parse_args()
    main(args.scene_id,args.config,
         device_str=args.device,
         save_semantic_masks=args.save_semantic_masks,
         save_instance_masks=args.save_instance_masks,
         debug=args.debug,
         auto_clean=args.auto_clean)
