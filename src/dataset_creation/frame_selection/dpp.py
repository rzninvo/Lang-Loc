"""
Determinantal Point Process (DPP) based view selection (2-stage pipeline).

Stage 1: Semantic 2-phase DPP (quality + CLIP similarity).
Stage 2: Spatial one-shot DPP (pose + pixel-IoU diversity) to final count.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


# ======================= Quality Metrics ======================================


def compute_face_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute unit face normals for the mesh.

    Args:
        V: (Nv, 3) vertex positions.
        F: (Nf, 3) face indices.

    Returns:
        normals: (Nf, 3) unit normal vectors per face.
    """
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return cross / norms


def compute_instance_entropy(obj_px: Dict[int, int]) -> float:
    """
    Compute normalized Shannon entropy over visible object pixel distribution.

    A view with many equally-sized objects gets high entropy (informative).
    A view dominated by one object gets low entropy (less informative).

    Args:
        obj_px: {objectId: pixel_count} from compute_image_visibility().

    Returns:
        Normalized entropy in [0, 1]. Returns 0 if fewer than 2 objects.
    """
    if len(obj_px) < 2:
        return 0.0

    counts = np.array(list(obj_px.values()), dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0

    probs = counts / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(probs))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


def compute_relation_density(
    visible_objects: Dict[int, Dict], eps: float = 1e-6
) -> float:
    """
    Compute pairwise proximity-based relation density among visible objects.

    S_rel = (1 / C(n,2)) * sum_{a<b} 1 / (|x_a - x_b| + eps)

    Higher density means objects are close together and relations are rich.

    Args:
        visible_objects: from compute_visible_objects(), keyed by objectId.
            Each entry must have "centroid_world" key.
        eps: small constant to avoid division by zero.

    Returns:
        S_rel score (non-negative float). Returns 0 if fewer than 2 objects.
    """
    centroids = []
    for meta in visible_objects.values():
        c = meta.get("centroid_world")
        if c is not None:
            centroids.append(np.array(c, dtype=np.float64))

    n = len(centroids)
    if n < 2:
        return 0.0

    n_pairs = n * (n - 1) / 2
    total_inv_dist = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            total_inv_dist += 1.0 / (dist + eps)

    return float(total_inv_dist / n_pairs)


def compute_normal_variance(
    face_normals: np.ndarray, visible_face_ids: np.ndarray
) -> float:
    """
    Compute geometric diversity of visible surface orientations.

    Uses the resultant length approach from directional statistics:
        V_norm = 1 - ||mean(unit_normals)||

    0 = all normals perfectly aligned (flat wall).
    1 = maximally dispersed orientations.

    Args:
        face_normals: (Nf, 3) precomputed unit face normals.
        visible_face_ids: array of face indices visible in this frame.

    Returns:
        V_norm in [0, 1].
    """
    if len(visible_face_ids) == 0:
        return 0.0

    normals = face_normals[visible_face_ids]
    mean_normal = normals.mean(axis=0)
    resultant_length = np.linalg.norm(mean_normal)
    return float(1.0 - min(resultant_length, 1.0))


def compute_vertex_novelty(
    F: np.ndarray, visible_face_ids: np.ndarray, vertex_counts: np.ndarray
) -> float:
    """
    Compute how novel a frame's visible geometry is relative to prior selections.

    V_count = 1 / (1 + mean(c(v) for v in visible_vertices))

    1 = completely novel (no prior visits), tends to 0 = heavily revisited.

    Args:
        F: (Nf, 3) face indices.
        visible_face_ids: face indices visible in this frame.
        vertex_counts: (Nv,) array of visit counts per vertex.

    Returns:
        V_count in (0, 1].
    """
    if len(visible_face_ids) == 0:
        return 0.0

    vert_ids = np.unique(F[visible_face_ids].ravel())
    mean_count = vertex_counts[vert_ids].mean()
    return float(1.0 / (1.0 + mean_count))


def update_vertex_counts(
    F: np.ndarray, visible_face_ids: np.ndarray, vertex_counts: np.ndarray
) -> np.ndarray:
    """
    Increment vertex visit counts for vertices of the selected frame's
    visible faces. Modifies vertex_counts in-place.

    Args:
        F: (Nf, 3) face indices.
        visible_face_ids: face indices visible in the selected frame.
        vertex_counts: (Nv,) int array, modified in-place.

    Returns:
        The same vertex_counts array.
    """
    if len(visible_face_ids) == 0:
        return vertex_counts
    vert_ids = np.unique(F[visible_face_ids].ravel())
    vertex_counts[vert_ids] += 1
    return vertex_counts


# ======================= CLIP Embeddings ======================================


def compute_clip_embeddings(
    image_paths: List[Path],
    model_name: str = "ViT-B/32",
    device: str = "cuda",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Compute L2-normalized CLIP image embeddings for a list of images.

    Args:
        image_paths: list of paths to RGB images.
        model_name: CLIP model architecture (e.g., "ViT-B/32").
        device: compute device.
        batch_size: inference batch size.

    Returns:
        embeddings: (N, d) float32 array, L2-normalized per row.
    """
    try:
        import clip
    except ImportError:
        raise ImportError(
            "OpenAI CLIP is not installed. Install it with:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading CLIP model {model_name} on {dev}...")
    model, preprocess = clip.load(model_name, device=dev)
    model.eval()

    all_embeddings = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            images.append(preprocess(img))

        batch_tensor = torch.stack(images).to(dev)
        with torch.no_grad():
            features = model.encode_image(batch_tensor)
        # L2 normalize
        features = features / features.norm(dim=-1, keepdim=True)
        all_embeddings.append(features.cpu().float().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"[INFO] CLIP embeddings computed: {embeddings.shape}")
    return embeddings.astype(np.float32)


# ======================= DPP Solvers ==========================================


def greedy_map_dpp(L: np.ndarray, k: int) -> List[int]:
    """
    Greedy MAP DPP selection via incremental Cholesky updates.

    At each step, selects the item that maximizes log det(L_{S+{i}}).
    Uses the efficient O(k^2 * N) algorithm.

    Args:
        L: (N, N) positive semidefinite DPP L-kernel.
        k: number of items to select.

    Returns:
        List of selected indices (up to k).
    """
    N = L.shape[0]
    k = min(k, N)

    selected: List[int] = []
    cis = np.zeros((k, N), dtype=np.float64)  # Cholesky rows
    di2 = np.diag(L).copy().astype(np.float64)  # conditional variance

    for t in range(k):
        scores = di2.copy()
        for s in selected:
            scores[s] = -np.inf

        best = int(np.argmax(scores))
        if scores[best] <= 1e-12:
            break

        selected.append(best)

        # Update Cholesky factor
        if t == 0:
            sqrt_d = np.sqrt(di2[best])
            cis[0, :] = L[:, best] / sqrt_d
        else:
            ei = L[:, best].copy()
            for s in range(t):
                ei -= cis[s, best] * cis[s, :]
            sqrt_d = np.sqrt(di2[best])
            cis[t, :] = ei / sqrt_d

        # Update conditional variance
        di2 -= cis[t, :] ** 2
        di2 = np.maximum(di2, 0.0)

    return selected


def greedy_map_dpp_conditioned(
    L: np.ndarray, k: int, preselected: List[int]
) -> List[int]:
    """
    Greedy MAP DPP conditioned on a pre-selected set S.

    First processes the preselected items through Cholesky updates
    (consuming their conditional variance) without counting them toward k.
    Then greedily picks k NEW items that maximize log det(L_{S∪T}) for
    growing T, ensuring CLIP-space repulsion from both S and earlier T items.

    Args:
        L: (N, N) positive semidefinite DPP L-kernel over ALL items.
        k: number of NEW items to select (beyond preselected).
        preselected: indices of already-selected items in L.

    Returns:
        List of k newly selected indices (not including preselected).
    """
    N = L.shape[0]
    k = min(k, N - len(preselected))

    total_steps = len(preselected) + k
    cis = np.zeros((total_steps, N), dtype=np.float64)
    di2 = np.diag(L).copy().astype(np.float64)

    # --- Pre-load Cholesky state with preselected items ---
    blocked = set(preselected)
    for t, idx in enumerate(preselected):
        if t == 0:
            sqrt_d = np.sqrt(max(di2[idx], 1e-12))
            cis[0, :] = L[:, idx] / sqrt_d
        else:
            ei = L[:, idx].copy()
            for s in range(t):
                ei -= cis[s, idx] * cis[s, :]
            sqrt_d = np.sqrt(max(di2[idx], 1e-12))
            cis[t, :] = ei / sqrt_d

        di2 -= cis[t, :] ** 2
        di2 = np.maximum(di2, 0.0)

    # --- Greedily pick k new items conditioned on S ---
    new_selected: List[int] = []
    t_offset = len(preselected)

    for step in range(k):
        t = t_offset + step
        scores = di2.copy()
        for b in blocked:
            scores[b] = -np.inf

        best = int(np.argmax(scores))
        if scores[best] <= 1e-12:
            break

        new_selected.append(best)
        blocked.add(best)

        # Cholesky update
        ei = L[:, best].copy()
        for s in range(t):
            ei -= cis[s, best] * cis[s, :]
        sqrt_d = np.sqrt(max(di2[best], 1e-12))
        cis[t, :] = ei / sqrt_d

        di2 -= cis[t, :] ** 2
        di2 = np.maximum(di2, 0.0)

    return new_selected


# ======================= Spatial Similarity ===================================


def _gaussian_similarity(distance: float, sigma: float) -> float:
    """
    Convert a distance into a similarity in (0, 1] using an RBF kernel.
    """
    sigma = max(float(sigma), 1e-6)
    return float(np.exp(-(distance ** 2) / (2.0 * sigma ** 2)))


def _compute_pairwise_iou(
    visibility_masks: List[np.ndarray],
    candidate_indices: List[int],
) -> np.ndarray:
    """
    Compute symmetric pairwise pixel IoU for candidate frames.

    Args:
        visibility_masks: per-frame (H, W) uint8 labeled-object masks.
        candidate_indices: indices into visibility_masks.

    Returns:
        (M, M) float64 symmetric IoU matrix with 1s on diagonal.
    """
    M = len(candidate_indices)
    iou = np.zeros((M, M), dtype=np.float64)

    for i_local in range(M):
        mi = visibility_masks[candidate_indices[i_local]].ravel().astype(bool)
        for j_local in range(i_local + 1, M):
            mj = visibility_masks[candidate_indices[j_local]].ravel().astype(bool)
            intersection = np.count_nonzero(mi & mj)
            union = np.count_nonzero(mi | mj)
            if union > 0:
                iou_val = intersection / union
                iou[i_local, j_local] = iou_val
                iou[j_local, i_local] = iou_val

    np.fill_diagonal(iou, 1.0)
    return iou


def _build_spatial_similarity_matrix(
    candidate_indices: List[int],
    visibility_masks: List[np.ndarray],
    poses_by_index: Dict[int, np.ndarray],
    sigma_position: float,
    sigma_angle: float,
    iou_gamma: float = 1.0,
) -> np.ndarray:
    """
    Build spatial similarity S = S_pose * IoU^gamma for Stage 2 DPP.

    High similarity = frames are similar (close pose AND overlapping content),
    so the DPP will repel them.

    Args:
        candidate_indices: indices into the full image_stats list.
        visibility_masks: per-frame (H, W) uint8 labeled-object masks.
        poses_by_index: mapping from frame index to (4,4) cam2world pose.
        sigma_position: RBF bandwidth for position distance (meters).
        sigma_angle: RBF bandwidth for angular distance (degrees).
        iou_gamma: exponent on IoU (< 1 softens, 1 = direct IoU).

    Returns:
        (M, M) similarity matrix.
    """
    from src.utils.camera_utils import compute_pose_distance

    M = len(candidate_indices)
    sim = np.ones((M, M), dtype=np.float64)
    if M <= 1:
        return sim

    iou_matrix = _compute_pairwise_iou(visibility_masks, candidate_indices)

    for i_local in range(M):
        i = candidate_indices[i_local]
        pose_i = poses_by_index.get(i)

        for j_local in range(i_local + 1, M):
            j = candidate_indices[j_local]
            pose_j = poses_by_index.get(j)

            pose_sim = 1.0
            if pose_i is not None and pose_j is not None:
                pos_dist, ang_dist = compute_pose_distance(pose_i, pose_j)
                pose_sim = _gaussian_similarity(pos_dist, sigma_position) * _gaussian_similarity(
                    ang_dist, sigma_angle
                )

            overlap_sim = iou_matrix[i_local, j_local] ** iou_gamma

            sim_ij = pose_sim * overlap_sim
            sim[i_local, j_local] = sim_ij
            sim[j_local, i_local] = sim_ij

    np.fill_diagonal(sim, 1.0)
    return sim


# ======================= 2-Stage DPP Pipeline =================================


def dpp_select_views(
    image_stats: List[Dict],
    V: np.ndarray,
    F: np.ndarray,
    face_normals: np.ndarray,
    face_obj_ids: np.ndarray,
    clip_embeddings: np.ndarray,
    visibility_masks: List[np.ndarray],
    total_views: int = 10,
    seed_size: int = 4,
    camera_poses: Optional[Dict[str, np.ndarray]] = None,
    stage1_total_views: int = 25,
    stage2_sigma_position: float = 0.75,
    stage2_sigma_angle: float = 20.0,
    stage2_iou_gamma: float = 1.0,
) -> List[str]:
    """
    Two-stage DPP view selection.

    Stage 1: semantic 2-phase DPP on all candidate views.
        Similarity: CLIP cosine in [0, 1].
        Quality: q_i = H_v * S_rel * V_norm (plus novelty in sequential phase).
        No backfill — DPP may naturally produce fewer than stage1_total_views.

    Stage 2: spatial one-shot DPP on Stage-1 subset to final count.
        Similarity: S_pose(i,j) * IoU(i,j)^gamma.
        Quality: uniform (q_i = 1), i.e. pure spatial diversity.
        No backfill — may return fewer than total_views.

    Args:
        image_stats: per-frame stats, each must have keys:
            "fid", "obj_pixels", "visible_objects", "visible_face_ids".
        V: (Nv, 3) mesh vertices.
        F: (Nf, 3) face indices.
        face_normals: (Nf, 3) precomputed unit face normals.
        face_obj_ids: (Nf,) object id per face.
        clip_embeddings: (N, d) L2-normalized CLIP embeddings.
        visibility_masks: (N,) list of (H, W) uint8 labeled-object masks.
        total_views: final number of views (Stage 2 target).
        seed_size: Stage-1 seed count for semantic DPP.
        camera_poses: optional dict mapping fid -> (4,4) cam2world pose.
        stage1_total_views: max candidates after Stage 1.
        stage2_sigma_position: RBF bandwidth for position distance (meters).
        stage2_sigma_angle: RBF bandwidth for angular distance (degrees).
        stage2_iou_gamma: exponent on IoU for Stage 2 (< 1 softens).

    Returns:
        List of selected frame ID strings (length <= total_views).
    """
    _ = face_obj_ids

    N = len(image_stats)
    if N == 0:
        return []
    if clip_embeddings.shape[0] != N:
        raise ValueError(
            f"clip_embeddings has {clip_embeddings.shape[0]} rows but image_stats has {N} items."
        )
    if len(visibility_masks) != N:
        raise ValueError(
            f"visibility_masks has {len(visibility_masks)} items but image_stats has {N} items."
        )

    final_target = min(max(int(total_views), 0), N)
    if final_target <= 0:
        return []

    stage1_target = min(max(int(stage1_total_views), final_target), N)
    seed_size = min(max(int(seed_size), 0), stage1_target)
    phase2_size = stage1_target - seed_size

    print(
        "[DPP] 2-stage config: "
        f"stage1={stage1_target}, final={final_target}"
    )

    Nv = V.shape[0]
    vertex_counts = np.zeros(Nv, dtype=np.int32)

    # ------ Precompute per-frame quality components ------
    entropies = np.zeros(N, dtype=np.float64)
    rel_densities = np.zeros(N, dtype=np.float64)
    norm_variances = np.zeros(N, dtype=np.float64)
    visible_face_ids_list: List[np.ndarray] = []

    for i, stat in enumerate(image_stats):
        entropies[i] = compute_instance_entropy(stat.get("obj_pixels", {}))
        rel_densities[i] = compute_relation_density(stat.get("visible_objects", {}))

        vfids = stat.get("visible_face_ids", [])
        if isinstance(vfids, np.ndarray):
            vfids_arr = vfids.astype(np.int64, copy=False)
        else:
            vfids_arr = np.array(list(vfids), dtype=np.int64)
        visible_face_ids_list.append(vfids_arr)
        norm_variances[i] = compute_normal_variance(face_normals, vfids_arr)

    # Normalize relation density to [0, 1] across candidates.
    rd_min, rd_max = rel_densities.min(), rel_densities.max()
    if rd_max > rd_min:
        rel_densities = (rel_densities - rd_min) / (rd_max - rd_min)
    else:
        rel_densities[:] = 1.0

    q_base = entropies * rel_densities * norm_variances
    q_base = np.maximum(q_base, 1e-8)

    # Stage-1 semantic similarity.
    cos_sim = clip_embeddings @ clip_embeddings.T
    sim_sem = np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0)
    np.fill_diagonal(sim_sem, 1.0)

    # ======================== STAGE 1: Semantic DPP =========================
    print(f"[DPP] Stage 1: selecting up to {stage1_target} semantic candidates...")
    L_seed = np.outer(q_base, q_base) * sim_sem
    L_seed += np.eye(N) * 1e-10

    seed_indices = greedy_map_dpp(L_seed, seed_size)
    for idx in seed_indices:
        update_vertex_counts(F, visible_face_ids_list[idx], vertex_counts)

    selected_indices = list(seed_indices)
    selected_set = set(selected_indices)
    remaining = set(range(N)) - selected_set

    for _ in range(phase2_size):
        if not remaining:
            break

        novelty = np.ones(N, dtype=np.float64)
        for idx in remaining:
            novelty[idx] = compute_vertex_novelty(
                F, visible_face_ids_list[idx], vertex_counts
            )

        q_full = np.maximum(q_base * novelty, 1e-8)
        L_full = np.outer(q_full, q_full) * sim_sem
        L_full += np.eye(N) * 1e-10

        new_picks = greedy_map_dpp_conditioned(L_full, 1, selected_indices)
        if not new_picks:
            break

        chosen = new_picks[0]
        selected_indices.append(chosen)
        selected_set.add(chosen)
        remaining.remove(chosen)
        update_vertex_counts(F, visible_face_ids_list[chosen], vertex_counts)

    stage1_indices = selected_indices[:stage1_target]
    stage1_fids = [image_stats[i]["fid"] for i in stage1_indices]
    print(f"[DPP] Stage 1 output ({len(stage1_fids)}): {stage1_fids}")

    # ======================== STAGE 2: Spatial DPP ==========================
    stage2_target = min(final_target, len(stage1_indices))

    poses_by_index: Dict[int, np.ndarray] = {}
    if camera_poses is not None:
        for idx in stage1_indices:
            fid = image_stats[idx]["fid"]
            pose = camera_poses.get(fid)
            if pose is not None:
                poses_by_index[idx] = pose

    missing_poses = len(stage1_indices) - len(poses_by_index)
    if camera_poses is None:
        print("[DPP] Stage 2: no camera poses provided, using IoU-only similarity.")
    elif missing_poses > 0:
        print(
            f"[DPP] Stage 2: {missing_poses} candidates missing poses, "
            "falling back to IoU-only for those pairs."
        )

    print(f"[DPP] Stage 2: selecting up to {stage2_target} spatially diverse views...")
    spatial_sim = _build_spatial_similarity_matrix(
        stage1_indices,
        visibility_masks,
        poses_by_index,
        sigma_position=stage2_sigma_position,
        sigma_angle=stage2_sigma_angle,
        iou_gamma=stage2_iou_gamma,
    )
    L_spatial = spatial_sim + np.eye(len(stage1_indices)) * 1e-10
    stage2_local = greedy_map_dpp(L_spatial, stage2_target)
    stage2_indices = [stage1_indices[i] for i in stage2_local]

    result = [image_stats[i]["fid"] for i in stage2_indices]
    print(f"[DPP] Final selection ({len(result)} views): {result}")
    return result
