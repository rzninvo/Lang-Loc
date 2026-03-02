"""Legacy greedy next-best-view selection and K-means clustering."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def greedy_next_best_views(
    image_stats: List[Dict],
    max_images: int | None = None,
    min_gain_pixels: int = 0,
    alpha: float = 0.5,  # 1.0 = old behavior (coverage only)
    min_obj_pixels_for_presence: int = 100,  # pixels to count an object as "visible" (diversity)
    camera_poses: Dict[str, np.ndarray] | None = None,  # fid -> (4,4) cam2world pose
    min_position_distance: float = 0.0,  # Min position distance between views (meters)
    min_angle_distance: float = 0.0,  # Min angle distance between views (degrees)
    enable_pose_filtering: bool = False,  # Enable spatial diversity filtering
) -> List[str]:
    """
    Balanced greedy NBV with optional spatial diversity filtering:
      score = alpha * normalized_coverage_gain + (1 - alpha) * normalized_diversity

    - Coverage gain: how many *new* pixels we add toward each object's cap.
    - Diversity: how many distinct objects in this image are clearly visible
      (>= min_obj_pixels_for_presence). This reflects "descriptiveness".
    - Spatial filtering: Optionally reject views too close to already-selected ones.

    Both terms are normalized (0..1) per-iteration across remaining candidates
    so alpha meaningfully trades off the two.

    Args:
        image_stats: Per-frame visibility stats (must contain "fid" and "obj_pixels").
        max_images: Maximum number of views to select.
        min_gain_pixels: Minimum pixel gain to continue selection.
        alpha: Balance between coverage (1.0) and diversity (0.0).
        min_obj_pixels_for_presence: Min pixels to count object as present.
        camera_poses: Optional dict mapping frame id -> camera pose matrix.
        min_position_distance: Min distance (m) between selected camera positions.
        min_angle_distance: Min angle (deg) between selected viewing directions.
        enable_pose_filtering: If True, apply spatial diversity filtering.

    Returns:
        List of selected frame ids in selection order.
    """
    from langloc.utils.camera_utils import is_pose_too_similar

    covered: Dict[int, int] = defaultdict(int)  # covered pixels toward each object's cap
    remaining = set(range(len(image_stats)))
    selected: List[str] = []
    selected_poses: List[np.ndarray] = []  # Track poses of selected views for spatial filtering

    # Per-object caps: max contribution any single image can provide for that object.
    obj_caps: Dict[int, int] = defaultdict(int)
    for s in image_stats:
        for oid, c in s["obj_pixels"].items():
            obj_caps[oid] = max(obj_caps[oid], c)

    while remaining:
        # Compute raw terms for all remaining candidates this round
        cov_gains = {}
        diversities = {}

        for i in remaining:
            s = image_stats[i]

            # --- Coverage gain (with caps, only counts remaining headroom) ---
            gain_cov = 0
            for oid, c in s["obj_pixels"].items():
                cap = obj_caps[oid]
                if cap > covered[oid]:
                    gain_cov += min(c, cap - covered[oid])

            # --- Diversity (descriptiveness): # of objects clearly visible ---
            # Count objects with enough pixels in this image (regardless of novelty).
            gain_div = sum(1 for _, c in s["obj_pixels"].items() if c >= min_obj_pixels_for_presence)

            cov_gains[i] = gain_cov
            diversities[i] = gain_div

        if not cov_gains:
            break

        # Stop early if even the best raw coverage gain is below the floor
        best_raw_gain = max(cov_gains.values()) if cov_gains else 0
        if best_raw_gain < min_gain_pixels:
            break

        # --- Normalize terms to 0..1 so alpha is meaningful ---
        max_cov = max(cov_gains.values()) or 1
        max_div = max(diversities.values()) or 1

        best_idx = None

        # Sort candidates by score and try them in order (with pose filtering)
        scored_candidates = []
        for i in remaining:
            norm_cov = cov_gains[i] / max_cov
            norm_div = diversities[i] / max_div
            score = alpha * norm_cov + (1.0 - alpha) * norm_div
            scored_candidates.append((i, score))

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Try candidates in score order, applying pose filter if enabled
        for candidate_idx, _ in scored_candidates:
            fid = image_stats[candidate_idx]["fid"]

            # Check spatial diversity constraint if enabled
            if enable_pose_filtering and camera_poses is not None and fid in camera_poses:
                candidate_pose = camera_poses[fid]
                if is_pose_too_similar(
                    candidate_pose,
                    selected_poses,
                    min_position_distance,
                    min_angle_distance,
                ):
                    continue  # Skip this candidate, try next one

            # Accept this candidate
            best_idx = candidate_idx
            break

        if best_idx is None:
            break

        # Commit selection
        fid_selected = image_stats[best_idx]["fid"]
        selected.append(fid_selected)

        # Track pose for spatial filtering
        if enable_pose_filtering and camera_poses is not None and fid_selected in camera_poses:
            selected_poses.append(camera_poses[fid_selected])

        # Update covered toward caps using the selected frame's contributions
        for oid, c in image_stats[best_idx]["obj_pixels"].items():
            cap = obj_caps[oid]
            if covered[oid] < cap:
                covered[oid] = min(cap, covered[oid] + c)

        remaining.remove(best_idx)
        if max_images is not None and len(selected) >= max_images:
            break

    return selected


def cluster_camera_poses(
    camera_poses: List[np.ndarray],
    frame_ids: List[str],
    n_clusters: int,
    random_state: int = 42,
) -> Tuple[List[str], np.ndarray]:
    """
    Cluster camera poses using K-means and return representative frame IDs.

    Args:
        camera_poses: List of (4,4) camera-to-world pose matrices.
        frame_ids: List of frame IDs corresponding to each pose.
        n_clusters: Number of clusters to form.
        random_state: Random seed for reproducibility.

    Returns:
        cluster_representatives: List of frame IDs (one per cluster, sorted by original rank).
        cluster_labels: Array of cluster labels for each input pose.
    """
    from sklearn.cluster import KMeans

    n_candidates = len(camera_poses)

    if n_candidates <= 1 or n_clusters <= 1:
        print("[WARN] Not enough frames for meaningful clustering. Skipping clustering.")
        return frame_ids[:], np.zeros(n_candidates, dtype=int)

    # Use effective cluster count
    n_clusters = min(n_candidates, n_clusters)

    # Extract positions from poses
    positions = np.stack([pose[:3, 3] for pose in camera_poses], axis=0)

    # Run K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(positions)
    print(f"[INFO] K-means clustering into {n_clusters} clusters done.")

    # Map frame id -> original rank (lower is better, preserves NBV order)
    rank = {fid: i for i, fid in enumerate(frame_ids)}

    # Group frames by cluster id
    clusters = {}
    for lbl, fid in zip(cluster_labels, frame_ids):
        clusters.setdefault(int(lbl), []).append(fid)

    # Pick the top-ranked (earliest in original order) frame per cluster
    cluster_representatives = []
    for lbl, fids in clusters.items():
        best_fid = min(fids, key=lambda x: rank[x])
        cluster_representatives.append(best_fid)

    # Sort chosen reps by their original rank for a nice, stable order
    cluster_representatives.sort(key=lambda x: rank[x])

    return cluster_representatives, cluster_labels
