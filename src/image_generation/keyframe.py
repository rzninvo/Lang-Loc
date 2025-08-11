import os
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
import shutil
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import KMeans
from src.utils.config_loader import load_config

# ---------- Utility Functions ----------
def load_intrinsics(path):
    return np.loadtxt(path).reshape(4, 4)

def load_pose(pose_path):
    return np.loadtxt(pose_path).reshape(4, 4)

def load_depth(depth_path):
    return np.array(Image.open(depth_path)).astype(np.float32) / 1000.0  # mm to meters

def load_color(color_path):
    return np.array(Image.open(color_path))

def load_semantic_labels(label_path):
    return np.array(Image.open(label_path)).astype(np.uint16)

def compute_semantic_score(label_img, ignore_ids={0, 1, 3}):  # 0: unannotated, 1: wall, 3: floor
    unique_labels = set(np.unique(label_img))
    useful_labels = unique_labels - ignore_ids
    return len(useful_labels)

def compute_blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def backproject(depth, intrinsics):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    height, width = depth.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    z = depth
    x3 = (x - cx) * z / fx
    y3 = (y - cy) * z / fy
    points = np.stack((x3, y3, z), axis=-1).reshape(-1, 3)
    mask = (z.flatten() > 0) & ~np.isnan(points).any(axis=1)
    return points[mask]

def transform_points(points, pose):
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    return (pose @ points_h.T).T[:, :3]

def voxelize(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd.voxel_down_sample(voxel_size=voxel_size)

def evaluate_frame(fid, color_dir, depth_dir, pose_dir, label_dir, intrinsics, voxel_size, blur_threshold):
    depth = load_depth(depth_dir / f"{fid}.png")
    color = load_color(color_dir / f"{fid}.jpg")
    pose = load_pose(pose_dir / f"{fid}.txt")
    label_path = label_dir / f"{fid}.png"

    semantic_score = 0
    if label_path.exists():
        labels = load_semantic_labels(label_path)
        semantic_score = compute_semantic_score(labels)

    blur = compute_blur_score(color)
    if blur <= blur_threshold:
        return None  # discard blurry frames early

    points_cam = backproject(depth, intrinsics)
    points_world = transform_points(points_cam, pose)
    vox = voxelize(points_world, voxel_size)

    # Pose descriptor (position + forward direction)
    position = pose[:3, 3]
    forward = pose[:3, 2]
    cam_descriptor = np.concatenate([position, forward])

    return {
        "fid": fid,
        "pose": pose.tolist(),  # Save for camera_pose.json
        "voxel_count": len(vox.points),
        "blur": blur,
        "semantic_score": semantic_score,
        "pose_vector": cam_descriptor
    }

# ----------------- Main Pipeline -----------------
def main(scene_id, config_path, auto_clean=False):
    config = load_config(config_path)

    # Keyframe parameters from config
    voxel_size = config["keyframe"]["voxel_size"]
    num_frames = config["keyframe"]["num_frames"]
    blur_threshold = config["keyframe"]["blur_threshold"]
    kmeans_n_init = config["keyframe"]["kmeans_n_init"]
    kmeans_max_iter = config["keyframe"]["kmeans_max_iter"]
    kmeans_tol = config["keyframe"]["kmeans_tol"]
    max_retries = config["keyframe"].get("max_retries", 3)

    # Paths
    dataset_path = config["paths"]["dataset_path"]
    scan_path = Path(dataset_path) / scene_id
    output_dir = scan_path / config["render"]["output_folder"]

    color_dir = scan_path / "color"
    depth_dir = scan_path / "depth"
    pose_dir = scan_path / "pose"
    label_dir = scan_path / "2d-label-filt"
    intrinsic_path = scan_path / "intrinsic" / "intrinsic_color.txt"
    sens_file = scan_path / f"{scene_id}.sens"

    frame_ids = sorted([f.stem for f in color_dir.glob("*.jpg")])
    intrinsics = load_intrinsics(intrinsic_path)

    # Retry loop: relax thresholds if too few frames
    retry_count = 0
    frame_scores = []

    while retry_count <= max_retries:
        print(f"[INFO] Attempt {retry_count+1}: blur_threshold={blur_threshold}, voxel_size={voxel_size}")

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(
                evaluate_frame, fid, color_dir, depth_dir, pose_dir, label_dir, intrinsics,
                voxel_size, blur_threshold
            ) for fid in frame_ids]

            frame_scores = [res for f in tqdm(futures, desc="Processing frames") if (res := f.result())]

        if len(frame_scores) >= num_frames or retry_count == max_retries:
            break  # we have enough frames or hit max tries

        # Relax thresholds
        blur_threshold *= 0.8      # allow slightly blurrier images
        voxel_size *= 1.2          # accept sparser geometry
        retry_count += 1

    # Normalize scores
    semantic_vals = np.array([f["semantic_score"] for f in frame_scores])
    voxel_vals = np.array([f["voxel_count"] for f in frame_scores])
    blur_vals = np.array([f["blur"] for f in frame_scores])

    semantic_norm = (semantic_vals - semantic_vals.min()) / (semantic_vals.max() - semantic_vals.min() + 1e-8)
    voxel_norm = (voxel_vals - voxel_vals.min()) / (voxel_vals.max() - voxel_vals.min() + 1e-8)
    blur_norm = (blur_vals - blur_vals.min()) / (blur_vals.max() - blur_vals.min() + 1e-8)

    for f, s, v, b in zip(frame_scores, semantic_norm, voxel_norm, blur_norm):
        f["score"] = s + 0.5 * v + 1.0 * b

    # K-Means on pose vectors
    pose_matrix = np.stack([f["pose_vector"] for f in frame_scores])
    mask = np.isfinite(pose_matrix).all(axis=1)
    pose_matrix = pose_matrix[mask]
    kmeans = KMeans(
        n_clusters=min(num_frames, len(frame_scores)),
        random_state=42,
        n_init=kmeans_n_init,
        max_iter=kmeans_max_iter,
        tol=kmeans_tol
    ).fit(pose_matrix)
    labels = kmeans.labels_

    selected = []
    for cluster_id in range(kmeans.n_clusters):
        cluster_frames = [f for f, label in zip(frame_scores, labels) if label == cluster_id]
        if cluster_frames:
            best = max(cluster_frames, key=lambda f: f["score"])
            selected.append(best)

    print("\nSelected Keyframes:")
    for f in selected:
        print(f"{f['fid']}: semantic={f['semantic_score']}, voxels={f['voxel_count']}, blur={f['blur']:.2f}")

    # Save output
    for sub in ["color", "depth", "pose", "label"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    camera_pose_json = {}
    for f in selected:
        fid = f["fid"]
        shutil.copy(color_dir / f"{fid}.jpg", output_dir / "color" / f"{fid}.jpg")
        shutil.copy(depth_dir / f"{fid}.png", output_dir / "depth" / f"{fid}.png")
        shutil.copy(pose_dir / f"{fid}.txt", output_dir / "pose" / f"{fid}.txt")
        label_path = label_dir / f"{fid}.png"
        if label_path.exists():
            shutil.copy(label_path, output_dir / "label" / f"{fid}.png")

        camera_pose_json[fid] = f["pose"]

    with open(output_dir / config["paths"]["camera_pose_file"], "w") as f:
        json.dump(camera_pose_json, f, indent=2)

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

# ---------- Entry Point ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate keyframes and camera poses from ScanNet scenes")
    parser.add_argument("scene_id", type=str, help="Scene ID (e.g., scene0000_00)")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML file")
    parser.add_argument("--auto_clean", action="store_true", help="Automatically delete .sens file and extracted folders after processing")
    args = parser.parse_args()

    main(scene_id=args.scene_id, config_path=args.config, auto_clean=args.auto_clean)
