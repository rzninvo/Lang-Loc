#!/usr/bin/env python3
"""
visualize_eval_loc.py
---------------------
Evaluate localisation quality for ScanScribe style captions that come with
ground-truth camera poses. For every 3RScan scene this script:

1.  Loads processed 3D-SSG graphs, per-scene meshes, and selects a frame JSON
    from `output/descriptions` according to the requested policy.
2.  Builds a caption SceneGraph from visible_objects + spatial_relations in the
    frame (word2vec embeddings only).
3.  Matches caption nodes to 3D objects via cosine similarity and keeps the top
    K candidates.
4.  Loads the coloured mesh, samples an XY grid at eye height, and computes
    centroids for each matched object.
5.  Casts rays from every grid camera to those centroids, counts first hits, and
    derives visibility probabilities.
6.  Extracts the ground-truth camera centre/direction from `scene_pose` and
    reports Hit@r curve, mass-radius, Top-K min distance, angular error, and
    distance error at the ground-truth.
7.  Optionally aggregates viewing directions into a FOV-weighted arrow field
    and softmaxes those counts into probabilities.
8.  Chooses camera predictions for both the grid and arrow candidates using
    the same strategy (argmax/random/cluster-weighted), optionally averaging
    directions for the arrow field.
9.  Visualises heatmap scatter, arrow quiver, and an Open3D scene with matched
    objects, probability spheres, and GT/predicted cameras.
10. Logs a per-scene table, aggregate metrics, and optionally saves a JSON dump.

The script reuses helper functions from visualize_loc_prob.py and constructs
caption graphs directly from the structured per-frame JSON to avoid any LLM
dependency during evaluation.


### Direct scene graph to scene graph object matching via word2vec similarity
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

# --------------------------------------------------------------------------- #
# Repository imports                                                         #
# --------------------------------------------------------------------------- #

from langloc.graphs.scene_graph import SceneGraph
from langloc.utils.embedding import _embed_word2vec
from langloc.utils.mesh_segmentation import build_segmented_mesh

# --------------------------------------------------------------------------- #
# Import helpers from langloc.localization                                   #
# --------------------------------------------------------------------------- #

from langloc.localization.grid import load_scene, first_hit_is_object
from langloc.localization.matching import topk_matched_objects
from langloc.localization.visualization import (
    colour_objects, colormap, dir_to_yaw_pitch,
    best_fov_window, average_direction,
    add_heatmap_markers, add_arrow_markers,
    create_camera_frustum,
)
from langloc.localization.frame_io import (
    camera_center_from_pose,
    ensure_query_root,
    format_args_section,
    load_frame_jsons,
    select_frame,
    frame_to_scenegraph,
    load_scene_graphs,
)
from langloc.localization.prediction import (
    select_prediction_point,
    top_n_fov_poses,
)
from langloc.eval.metrics import compute_view_iou_error
from langloc.eval.view_iou import (
    _camera_axes_from_forward,
    _visible_triangles_from_view,
)

# These functions do not exist in langloc.localization yet.
apply_exact_label_score = None
label_embedding_for_matching = None
canonical_label_for_matching = None


# --------------------------------------------------------------------------- #
# Data containers                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class SceneMetrics:
    scene_id: str
    frame_id: str
    hit_masses: Dict[float, float]
    mass_radii: Dict[float, float]
    topk_min_dist: float
    distance_error: float
    angular_error_deg: Optional[float]
    grid_points: int
    matched_objects: int
    iou_error: Optional[float] = None


def build_metrics_table(metrics_list: List[SceneMetrics],
                        hit_radii: List[float],
                        mass_percentiles: List[float],
                        topk_k: int) -> str:
    hit_radii = sorted(set(float(r) for r in hit_radii))
    mass_percentiles = sorted(set(float(p) for p in mass_percentiles))
    headers = [
        "Scene",
        "Frame",
        *[f"Hit@{r:.2f}m" for r in hit_radii],
        *[f"R{p:.0f}%" for p in mass_percentiles],
        f"TopK{topk_k} (m)",
        "Err (m)",
        "Ang err (deg)",
        "IoU err",
        "Matches",
        "Grid pts",
    ]
    rows: List[List[str]] = []
    for m in metrics_list:
        hit_vals = [m.hit_masses.get(r, 0.0) for r in hit_radii]
        rad_vals = [m.mass_radii.get(p, float("nan")) for p in mass_percentiles]
        ang_err = "-" if m.angular_error_deg is None else f"{m.angular_error_deg:.2f}"
        rows.append([
            m.scene_id,
            m.frame_id,
            *[f"{v:.3f}" for v in hit_vals],
            *[f"{v:.3f}" if np.isfinite(v) else "-" for v in rad_vals],
            f"{m.topk_min_dist:.3f}",
            f"{m.distance_error:.3f}",
            ang_err,
            "-" if m.iou_error is None else f"{m.iou_error:.3f}",
            str(m.matched_objects),
            str(m.grid_points),
        ])

    if not rows:
        return ""

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

    separator = "-+-".join("-" * w for w in col_widths)
    lines = [fmt_row(headers), separator]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Camera pose + metric helpers                                               #
# --------------------------------------------------------------------------- #

def compute_metrics(cams: np.ndarray,
                    probs: np.ndarray,
                    gt_cam: np.ndarray,
                    hit_radii: List[float],
                    mass_percentiles: List[float],
                    topk_k: int) -> Tuple[int, SceneMetrics]:
    pred_idx = int(np.argmax(probs))
    distances = np.linalg.norm(cams - gt_cam[None, :], axis=1)

    hit_masses: Dict[float, float] = {}
    for r in sorted(set(float(r) for r in hit_radii)):
        hit_masses[r] = float(probs[distances <= r].sum())

    mass_radii: Dict[float, float] = {}
    order = np.argsort(distances)
    cum = np.cumsum(probs[order])
    for p in sorted(set(float(p) for p in mass_percentiles)):
        target = max(0.0, min(p / 100.0, 1.0))
        idx = int(np.searchsorted(cum, target, side="left"))
        if idx >= len(order):
            mass_radii[p] = float(distances[order[-1]])
        else:
            mass_radii[p] = float(distances[order[idx]])

    topk_k = max(1, int(topk_k))
    k = min(topk_k, len(probs))
    top_idx = np.argpartition(probs, -k)[-k:]
    topk_min_dist = float(distances[top_idx].min()) if len(top_idx) else float("nan")

    dist_err = float(np.linalg.norm(cams[pred_idx] - gt_cam))

    return pred_idx, SceneMetrics(
        scene_id="",
        frame_id="",
        hit_masses=hit_masses,
        mass_radii=mass_radii,
        topk_min_dist=topk_min_dist,
        distance_error=dist_err,
        angular_error_deg=None,
        grid_points=len(cams),
        matched_objects=0,
    )


def softmax_probs(scores: np.ndarray, tau: float) -> np.ndarray:
    """Return softmax probabilities over scores with temperature tau."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores
    tau = max(float(tau), 1e-6)
    scores = scores / tau
    scores -= scores.max()
    exp_scores = np.exp(scores)
    denom = exp_scores.sum()
    if denom <= 0:
        return np.full_like(scores, 1.0 / len(scores))
    return exp_scores / denom


def proximity_bonus(distances: np.ndarray, decay: float) -> float:
    """Distance bonus where closer objects contribute more."""
    if distances.size == 0:
        return 0.0
    decay = max(float(decay), 1e-6)
    return float(np.exp(-distances / decay).sum())


# --------------------------------------------------------------------------- #
# Main evaluation pipeline                                                   #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate localisation probabilities against ground-truth camera poses."
    )
    parser.add_argument("--root", required=True,
                        help="Root directory containing <scene_id>/ meshes.")
    parser.add_argument("--graphs", required=True, type=Path,
                        help="processed_data directory holding 3dssg/*.pt files.")
    parser.add_argument("--query_root", type=Path,
                        help="Root containing per-scene output/descriptions/frame-*.json")

    parser.add_argument("--scene_ids", nargs="+",
                        help="Subset of scene IDs to evaluate. Defaults to intersection of graphs and query_root.")
    parser.add_argument("--max_scenes", type=int,
                        help="Limit number of scenes processed (after filtering).")
    parser.add_argument("--visualize_scene",
                        help="Scene ID to focus on for visualisation. Overrides --scene_ids when set.")

    parser.add_argument("--frame_policy",
                        choices=["first", "index", "random", "max_visible", "max_pixels"],
                        default="max_visible",
                        help="Strategy to pick which frame JSON to evaluate per scene.")
    parser.add_argument("--frame_index", type=int, default=0,
                        help="Frame index used when --frame_policy=index.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for random frame selection.")

    parser.add_argument("--top_k", type=int, default=25,
                        help="How many object matches to keep per caption.")
    parser.add_argument("--dynamic_top_k", action="store_true",
                        help="Use number of detected query objects as Top-K.")
    parser.add_argument("--score_threshold", type=float, default=-1.0,
                        help="Minimum cosine match score to keep (e.g. 0.1).")
    parser.add_argument("--exact_label_score", type=float, default=-1e9,
                        help=("Force canonical exact-label query/scene pairs to at least this "
                              "cosine score before ranking. Set <= -1e9 to disable."))
    parser.add_argument("--homogenize_label_embeddings", dest="homogenize_label_embeddings",
                        action="store_true", default=True,
                        help=("Encode both query and scene node labels with the same loader "
                              "word2vec function for object matching. Enabled by default."))
    parser.add_argument("--no_homogenize_label_embeddings", dest="homogenize_label_embeddings",
                        action="store_false",
                        help="Disable label-embedding homogenization and use stored node features.")
    parser.add_argument("--ensure_query_coverage", dest="ensure_query_coverage",
                        action="store_true", default=True,
                        help=("Ensure Top-K covers query nodes first: pick one best unique scene "
                              "match per query node before global fill. Enabled by default."))
    parser.add_argument("--no_ensure_query_coverage", dest="ensure_query_coverage",
                        action="store_false",
                        help="Disable per-query coverage and use pure global Top-K ranking.")
    parser.add_argument("--use_subgraph", action="store_true",
                        help="Apply DBSCAN-based subgraph matching before cosine Top-K.")
    parser.add_argument("--query_embedding_mode",
                        choices=["token", "doc"],
                        default="token",
                        help=("Embedding mode for query labels/relations: "
                              "'token' matches graph_loader_utils.get_word2vec (first token), "
                              "'doc' uses spaCy doc.vector."))
    parser.add_argument("--scene_use_attributes", action="store_true",
                        help=("Use 3D scene object attributes in node features during matching. "
                              "Default is False for cleaner object-label matching."))
    parser.add_argument("--debug_match_labels", action="store_true",
                        help="Print per-query top label matches before filtering.")
    parser.add_argument("--debug_match_all_scores", action="store_true",
                        help="Print full query-vs-scene label scores (all scene objects).")
    parser.add_argument("--debug_match_topn", type=int, default=5,
                        help="How many top scene labels to print per query node when debugging.")
    parser.add_argument("--debug_match_csv_dir", type=Path,
                        help="Optional directory to save full query-vs-scene cosine scores as CSV.")
    parser.add_argument("--grid_step", type=float, default=0.25,
                        help="XY grid spacing in metres.")
    parser.add_argument("--eye_height", type=float, default=1.6,
                        help="Eye-height offset used by the grid sampler.")
    parser.add_argument("--prob_eps", type=float, default=1e-6,
                        help="Numerical epsilon when computing log-probabilities (unused).")
    parser.add_argument("--hit_radii", nargs="+", type=float,
                        default=[0.75, 1.0, 1.5, 2.0, 2.5],
                        help="Radii (metres) for Hit@r mass curve.")
    parser.add_argument("--mass_percentiles", nargs="+", type=float,
                        default=[50.0, 90.0],
                        help="Percentiles for mass-radius metric (e.g., 90 95).")
    parser.add_argument("--top_k_min_dist", type=int, default=10,
                        help="K for Top-K min distance metric. Minimum distance among top-K probs.")
    parser.add_argument("--prediction_strategy",
                        choices=["argmax", "random", "weighted"],
                        default="weighted",
                        help="How to convert candidate positions into a final camera prediction.")
    parser.add_argument("--cluster_bandwidth", type=float, default=1.0,#0.75 #0.25
                        help="Bandwidth (metres) for the weighted cluster strategy.")
    parser.add_argument("--max_cluster_points", type=int, default=50,#512 # 20
                        help="Maximum candidates used when computing cluster-aware predictions.")

    parser.add_argument("--show_heatmap", action="store_true",
                        help="Show 2-D probability scatter heatmap.")
    parser.add_argument("--show_3d", action="store_true",
                        help="Visualise mesh with probability spheres in Open3D.")
    parser.add_argument("--show_arrows", action="store_true",
                        help="Show FOV-weighted arrow (quiver) plot.")
    parser.add_argument("--h_fov_deg", type=float, default=39.31,
                        help="Horizontal FOV (degrees) for arrow aggregation.")
    parser.add_argument("--v_fov_deg", type=float, default=64.76,
                        help="Vertical FOV (degrees) for arrow aggregation.")
    parser.add_argument("--arrow_stride", type=int, default=2,
                        help="Plot every Nth grid camera in the arrow field.")
    parser.add_argument("--arrow_len", type=float, default=0.0,
                        help="Maximum arrow length (metres). 0 → 0.9 * grid_step.")
    parser.add_argument("--score_tau", type=float, default= 1.5, #1.5, # 0.5, # works quite well
                        help="Temperature for softmax sharpening over visibility counts.")
    parser.add_argument("--distance_bonus_weight", type=float, default=0.5,
                        help="Weight of proximity bonus added to each camera score.")
    parser.add_argument("--distance_bonus_decay", type=float, default=2.0,
                        help="Distance decay (metres) for proximity bonus exp(-d/decay).")

    parser.add_argument("--save_metrics", type=Path,
                        help="Optional path to save per-scene metrics as JSON.")
    parser.add_argument("--log_file", type=Path, default=Path("eval_loc_summary.log"),
                        help="Path to write a plain-text summary log.")
    parser.add_argument("--top_pose_count", type=int, default=5,
                        help="Number of top FOV-weighted poses to list.")
    return parser.parse_args()


def debug_label_matches(query_graph: SceneGraph,
                        scene_graph: SceneGraph,
                        topn: int = 5,
                        print_all: bool = False,
                        csv_dir: Optional[Path] = None,
                        scene_id: str = "",
                        frame_id: str = "",
                        exact_label_score: float = -1e9,
                        homogenize_label_embeddings: bool = False) -> None:
    """Print top scene-label matches per query node."""
    def _canon(text: str) -> str:
        if canonical_label_for_matching is not None:
            return canonical_label_for_matching(text)
        return str(text).strip().lower()

    qids = list(query_graph.nodes)
    sids = list(scene_graph.nodes)
    if not qids or not sids:
        print("    [DEBUG] match labels: empty query/scene nodes.")
        return

    if homogenize_label_embeddings and label_embedding_for_matching is not None:
        qf = np.asarray([label_embedding_for_matching(query_graph.nodes[q].label)
                         for q in qids], dtype=np.float32)
        sf = np.asarray([label_embedding_for_matching(scene_graph.nodes[s].label)
                         for s in sids], dtype=np.float32)
    else:
        qf = np.asarray([query_graph.nodes[q].features for q in qids], dtype=np.float32)
        sf = np.asarray([scene_graph.nodes[s].features for s in sids], dtype=np.float32)

    qn = np.linalg.norm(qf, axis=1, keepdims=True)
    sn = np.linalg.norm(sf, axis=1, keepdims=True)
    qn = np.where(qn < 1e-9, 1.0, qn)
    sn = np.where(sn < 1e-9, 1.0, sn)
    qf = qf / qn
    sf = sf / sn
    sim = qf @ sf.T
    if apply_exact_label_score is not None:
        qlabels = [query_graph.nodes[qid].label for qid in qids]
        slabels = [scene_graph.nodes[sid].label for sid in sids]
        sim = apply_exact_label_score(torch.tensor(sim, dtype=torch.float32),
                                      qlabels,
                                      slabels,
                                      exact_label_score=float(exact_label_score)).numpy()

    if csv_dir is not None:
        csv_dir.mkdir(parents=True, exist_ok=True)
        sid_safe = str(scene_id).replace("/", "_")
        fid_safe = str(frame_id).replace("/", "_")
        out_csv = csv_dir / f"match_scores_{sid_safe}_{fid_safe}.csv"
        with out_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["query_id", "query_label", "scene_oid", "scene_label", "cosine_score"])
            for qi, qid in enumerate(qids):
                qlabel = query_graph.nodes[qid].label
                order = np.argsort(-sim[qi])
                for si in order:
                    sid = sids[si]
                    slabel = scene_graph.nodes[sid].label
                    sval = float(sim[qi, si])
                    writer.writerow([qid, qlabel, sid, slabel, f"{sval:.8f}"])
        print(f"    [DEBUG] saved full match scores to {out_csv}")

    if print_all:
        print("    [DEBUG] full label matches per query node:")
        for qi, qid in enumerate(qids):
            qlabel = query_graph.nodes[qid].label
            qcanon = _canon(qlabel)
            order = np.argsort(-sim[qi])
            print(f"      q[{qid}] {qlabel}:")
            for si in order:
                sid = sids[si]
                slabel = scene_graph.nodes[sid].label
                sval = float(sim[qi, si])
                print(f"        {sid}:{slabel}({sval:.3f})")
            same_scores = []
            for si, sid in enumerate(sids):
                slabel = scene_graph.nodes[sid].label
                if qcanon and qcanon == _canon(slabel):
                    same_scores.append(float(sim[qi, si]))
            if same_scores:
                print(f"        [same-label best] {max(same_scores):.3f}")
        return

    topn = max(1, int(topn))
    print("    [DEBUG] top label matches per query node:")
    for qi, qid in enumerate(qids):
        qlabel = query_graph.nodes[qid].label
        qcanon = _canon(qlabel)
        order = np.argsort(-sim[qi])[: topn]
        parts = []
        for si in order:
            sid = sids[si]
            slabel = scene_graph.nodes[sid].label
            sval = float(sim[qi, si])
            parts.append(f"{sid}:{slabel}({sval:.3f})")
        same_scores = []
        for si, sid in enumerate(sids):
            slabel = scene_graph.nodes[sid].label
            if qcanon and qcanon == _canon(slabel):
                same_scores.append(float(sim[qi, si]))
        suffix = f" [same-label best: {max(same_scores):.3f}]" if same_scores else ""
        print(f"      q[{qid}] {qlabel} -> " + " | ".join(parts) + suffix)


def _extract_floor_bbox(scene_dir: Path,
                        verts: np.ndarray,
                        tris: np.ndarray,
                        obj2faces: Dict[int, np.ndarray]) -> Optional[Dict[str, float]]:
    """Return the AABB of all 'floor'-labelled objects as {x_min, x_max,
    y_min, y_max, z_min, z_max}.  Returns None when semseg.v2.json is
    missing or contains no floor segment."""
    semseg_path = scene_dir / "semseg.v2.json"
    if not semseg_path.exists():
        return None

    groups = json.loads(semseg_path.read_text())["segGroups"]
    floor_ids = {int(g["objectId"]) for g in groups
                 if g.get("label", "").strip().lower() == "floor"}
    if not floor_ids:
        return None

    face_lists = [obj2faces[oid] for oid in floor_ids if oid in obj2faces]
    if not face_lists:
        return None

    floor_faces = np.concatenate(face_lists)
    floor_vert_idx = np.unique(tris[floor_faces].ravel())
    floor_verts = verts[floor_vert_idx]

    return {
        "x_min": float(floor_verts[:, 0].min()),
        "x_max": float(floor_verts[:, 0].max()),
        "y_min": float(floor_verts[:, 1].min()),
        "y_max": float(floor_verts[:, 1].max()),
        "z_min": float(floor_verts[:, 2].min()),
        "z_max": float(floor_verts[:, 2].max()),
    }


def evaluate_scene(scene_id: str,
                   scene_graph: SceneGraph,
                   args: argparse.Namespace,
                   rng: np.random.Generator) -> Optional[SceneMetrics]:
    mesh_root = Path(args.root)
    scene_dir = mesh_root / scene_id
    if not scene_dir.exists():
        print(f"[WARN] Scene directory missing for {scene_id} — skipped.")
        return None

    query_root = ensure_query_root(args.query_root, Path(args.root))
    desc_dir = query_root / scene_id / "output" / "descriptions"
    if not desc_dir.exists():
        # Fallback: allow descriptions alongside mesh root (already same path)
        desc_dir = scene_dir / "output" / "descriptions"
    frames = load_frame_jsons(desc_dir)
    if not frames:
        print(f"[WARN] No frame JSONs under {desc_dir} — skipped.")
        return None

    selection = select_frame(frames, args.frame_policy, args.frame_index, rng)
    if selection is None:
        print(f"[WARN] Frame selection failed for {scene_id} — skipped.")
        return None

    frame = selection.frame
    try:
        caption_graph, _ = frame_to_scenegraph(frame,
                                               query_embedding_mode=args.query_embedding_mode)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to build caption graph for {scene_id}: {exc}")
        return None

    frame_id_dbg = str(frame.get("image_index", selection.path.name))
    if args.debug_match_labels or args.debug_match_all_scores or args.debug_match_csv_dir is not None:
        debug_label_matches(caption_graph,
                            scene_graph,
                            topn=args.debug_match_topn,
                            print_all=args.debug_match_all_scores,
                            csv_dir=args.debug_match_csv_dir,
                            scene_id=scene_id,
                            frame_id=frame_id_dbg,
                            exact_label_score=args.exact_label_score,
                            homogenize_label_embeddings=args.homogenize_label_embeddings)

    gt_pose = frame.get("scene_pose")
    if gt_pose is None:
        print(f"[WARN] scene_pose missing in {selection.path} — skipped.")
        return None

    pose_mat = np.asarray(gt_pose, dtype=np.float64)
    gt_cam = camera_center_from_pose(pose_mat)
    rot_cam_world = pose_mat[:3, :3]
    forward_cv = rot_cam_world @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward_o3d = forward_cv
    norm_forward = np.linalg.norm(forward_o3d)
    gt_dir = forward_o3d / norm_forward if norm_forward > 1e-6 else None

    obj_ids, obj_scores = topk_matched_objects(
        caption_graph,
        scene_graph,
        k=args.top_k,
        return_scores=True,
        use_subgraph=args.use_subgraph,
        score_threshold=args.score_threshold,
        dynamic_k=args.dynamic_top_k,
        exact_label_score=args.exact_label_score,
        homogenize_label_embeddings=args.homogenize_label_embeddings,
        ensure_query_coverage=args.ensure_query_coverage,
    )
    if not obj_ids:
        print(f"[WARN] {scene_id}: no cosine matches — skipped.")
        return None

    mesh, tri2obj, obj2faces = load_scene(scene_dir)
    rc = o3d.t.geometry.RaycastingScene()
    mesh_id = rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    tri_pts = verts[tris]
    tri_vecs = tri_pts[:, 1] - tri_pts[:, 0]
    tri_vecs_b = tri_pts[:, 2] - tri_pts[:, 0]
    tri_cross = np.cross(tri_vecs, tri_vecs_b)
    tri_areas = 0.5 * np.linalg.norm(tri_cross, axis=1)
    tri_centroids = tri_pts.mean(axis=1)
    floor_bbox = _extract_floor_bbox(scene_dir, verts, tris, obj2faces)
    if floor_bbox is not None:
        x_min, x_max = floor_bbox["x_min"], floor_bbox["x_max"]
        y_min, y_max = floor_bbox["y_min"], floor_bbox["y_max"]
        z_eye = floor_bbox["z_max"] + args.eye_height
        print(f"    floor bbox: X=[{x_min:.2f}, {x_max:.2f}] "
              f"Y=[{y_min:.2f}, {y_max:.2f}] Z_eye={z_eye:.2f} m")
    else:
        print(f"    [WARN] No floor in semseg — sampling over full mesh bounds.")
        x_min, x_max = float(verts[:, 0].min()), float(verts[:, 0].max())
        y_min, y_max = float(verts[:, 1].min()), float(verts[:, 1].max())
        z_eye = float(verts[:, 2].min()) + args.eye_height

    gx = np.arange(x_min, x_max + 1e-4, args.grid_step)
    gy = np.arange(y_min, y_max + 1e-4, args.grid_step)
    Nx, Ny = len(gx), len(gy)
    xv, yv = np.meshgrid(gx, gy, indexing="xy")
    n = xv.size
    cams = np.stack([xv.ravel(), yv.ravel(), np.full(n, z_eye)], axis=1)

    centroids: Dict[int, np.ndarray] = {}
    for oid in obj_ids:
        faces = obj2faces.get(int(oid))
        if faces is None or not len(faces):
            continue
        centroids[int(oid)] = verts[np.unique(tris[faces].ravel())].mean(axis=0)

    if not centroids:
        print(f"[WARN] {scene_id}: matched objects missing geometry — skipped.")
        return None

    visible_dirs: List[List[np.ndarray]] = [[] for _ in range(len(cams))]
    visible_dists: List[List[float]] = [[] for _ in range(len(cams))]
    for idx, cam in enumerate(cams):
        for oid, centre in centroids.items():
            if first_hit_is_object(cam, centre, oid, rc, tri2obj):
                d = centre - cam
                l = np.linalg.norm(d)
                if l > 1e-6:
                    visible_dirs[idx].append(d / l)
                    visible_dists[idx].append(float(l))

    counts = np.array([len(v) for v in visible_dirs], dtype=np.int32)
    total = counts.sum()
    if total == 0:
        print(f"[WARN] {scene_id}: matched objects invisible from grid — skipped.")
        return None

    # Score = visibility count + proximity bonus (closer visible objects score higher)
    dist_bonus = np.array(
        [proximity_bonus(np.asarray(d, dtype=np.float64), args.distance_bonus_decay)
         for d in visible_dists],
        dtype=np.float64,
    )
    grid_scores = counts.astype(np.float64) + args.distance_bonus_weight * dist_bonus
    grid_probs = softmax_probs(grid_scores, args.score_tau)

    pred_idx, metrics = compute_metrics(cams, grid_probs, gt_cam,
                                        hit_radii=args.hit_radii,
                                        mass_percentiles=args.mass_percentiles,
                                        topk_k=args.top_k_min_dist)
    metrics.scene_id = scene_id
    metrics.frame_id = str(frame.get("image_index", selection.path.name))
    metrics.matched_objects = len(obj_ids)

    try:
        pred_cam_grid, grid_sel_idx, grid_sel_weights = select_prediction_point(
            cams,
            grid_probs,
            strategy=args.prediction_strategy,
            rng=rng,
            bandwidth=args.cluster_bandwidth,
            max_points=args.max_cluster_points,
        )
    except ValueError:
        pred_cam_grid = cams[pred_idx]
        grid_sel_idx = [int(pred_idx)]
        grid_sel_weights = np.asarray([1.0], dtype=np.float64)

    hfov_rad = math.radians(args.h_fov_deg)
    vfov_rad = math.radians(args.v_fov_deg)

    # ---- Arrow-based aggregation (computed regardless of plotting)
    arrow_positions: List[np.ndarray] = []
    arrow_dirs: List[np.ndarray] = []
    arrow_counts: List[float] = []

    have_arrow_helpers = bool(dir_to_yaw_pitch and best_fov_window and average_direction)
    if have_arrow_helpers:
        stride = max(1, int(args.arrow_stride))
        for gy_i in range(0, Ny, stride):
            for gx_i in range(0, Nx, stride):
                idx = gy_i * Nx + gx_i
                dirs = np.asarray(visible_dirs[idx], dtype=np.float32)
                if dirs.size == 0:
                    continue
                yaws = np.empty(len(dirs), dtype=np.float32)
                pits = np.empty(len(dirs), dtype=np.float32)
                for i, vec in enumerate(dirs):
                    yaw, pit = dir_to_yaw_pitch(vec)  # type: ignore[arg-type]
                    yaws[i] = yaw
                    pits[i] = pit
                sel, count = best_fov_window(yaws, pits, hfov_rad, vfov_rad)  # type: ignore[arg-type]
                if count == 0:
                    continue
                mdir = average_direction(dirs, sel)  # type: ignore[arg-type]
                if mdir is None:
                    continue
                local_dists = np.asarray(visible_dists[idx], dtype=np.float64)
                sel_bonus = proximity_bonus(local_dists[sel], args.distance_bonus_decay)
                arrow_score = float(count) + args.distance_bonus_weight * sel_bonus
                arrow_positions.append(cams[idx])
                arrow_dirs.append(mdir)
                arrow_counts.append(arrow_score)

    arrow_probs: Optional[np.ndarray] = None
    pred_cam_arrow: Optional[np.ndarray] = None
    pred_dir_arrow: Optional[np.ndarray] = None
    arrow_sel_idx: List[int] = []
    arrow_sel_weights = np.asarray([], dtype=np.float64)

    if arrow_counts:
        arrow_positions_np = np.asarray(arrow_positions, dtype=np.float64)
        arrow_dirs_np = np.asarray(arrow_dirs, dtype=np.float64)
        arrow_probs = softmax_probs(np.asarray(arrow_counts, dtype=np.float64),
                                    args.score_tau)
        top_fov_poses = top_n_fov_poses(arrow_positions_np,
                                        arrow_probs,
                                        n=args.top_pose_count,
                                        rng=rng,
                                        directions=arrow_dirs_np)
        # print(f"    top-{args.top_pose_count} FOV-weighted poses (pose x,y + dir): "
        #       f"{top_fov_poses}")
        try:
            pred_cam_arrow, arrow_sel_idx, arrow_sel_weights = select_prediction_point(
                arrow_positions_np,
                arrow_probs,
                strategy=args.prediction_strategy,
                rng=rng,
                bandwidth=args.cluster_bandwidth,
                max_points=args.max_cluster_points,
            )
        except ValueError:
            idx_fallback = int(np.argmax(arrow_probs))
            pred_cam_arrow = arrow_positions_np[idx_fallback]
            arrow_sel_idx = [idx_fallback]
            arrow_sel_weights = np.asarray([1.0], dtype=np.float64)

        if arrow_sel_idx:
            dir_vectors = arrow_dirs_np[arrow_sel_idx]
            weight_vec = arrow_sel_weights
            if weight_vec.shape[0] != len(arrow_sel_idx):
                weight_vec = np.ones(len(arrow_sel_idx), dtype=np.float64)
            weight_vec = np.clip(weight_vec, 0.0, None)
            if not np.any(weight_vec > 0):
                weight_vec = np.ones_like(weight_vec)
            weight_vec /= weight_vec.sum()
            mean_dir = np.sum(weight_vec[:, None] * dir_vectors, axis=0)
            norm_dir = float(np.linalg.norm(mean_dir))
            if norm_dir > 1e-6:
                pred_dir_arrow = mean_dir / norm_dir

    pred_source_primary = "arrow_field" if pred_cam_arrow is not None else "grid_probability"
    pred_cam_primary = pred_cam_arrow if pred_cam_arrow is not None else pred_cam_grid
    pred_dir_primary = pred_dir_arrow if pred_cam_arrow is not None else None

    pred_source = f"{pred_source_primary}:{args.prediction_strategy}"
    metrics.distance_error = float(np.linalg.norm(pred_cam_primary - gt_cam))
    if gt_dir is not None and pred_dir_primary is not None:
        dot = float(np.clip(np.dot(gt_dir, pred_dir_primary), -1.0, 1.0))
        metrics.angular_error_deg = float(math.degrees(math.acos(dot)))
    else:
        metrics.angular_error_deg = None

    grid_err = float(np.linalg.norm(pred_cam_grid - gt_cam))
    print(f"    predicted camera (grid:{args.prediction_strategy}): "
          f"{pred_cam_grid.tolist()} | err={grid_err:.3f} m")
    if pred_cam_arrow is not None:
        arrow_err = float(np.linalg.norm(pred_cam_arrow - gt_cam))
        print(f"    predicted camera (arrow:{args.prediction_strategy}): "
              f"{pred_cam_arrow.tolist()} | err={arrow_err:.3f} m")
        if pred_dir_arrow is not None:
            print(f"    approx. viewing direction (arrow): {pred_dir_arrow.tolist()}")
        else:
            print("    approx. viewing direction (arrow): n/a (no directional vote)")
    else:
        print("    predicted camera (arrow): n/a (no valid FOV windows)")
    print(f"    primary prediction used for metrics: {pred_source}")

    # --- debug: per-object visibility from GT vs predicted pose ---------------
    semseg_path = scene_dir / "semseg.v2.json"
    obj_labels: Dict[int, str] = {}
    if semseg_path.exists():
        for g in json.loads(semseg_path.read_text())["segGroups"]:
            obj_labels[int(g["objectId"])] = g.get("label", "").strip()

    gt_vis_oids: List[int] = []
    pred_vis_oids: List[int] = []
    for oid, centre in centroids.items():
        if first_hit_is_object(gt_cam, centre, oid, rc, tri2obj):
            gt_vis_oids.append(oid)
        if first_hit_is_object(pred_cam_primary, centre, oid, rc, tri2obj):
            pred_vis_oids.append(oid)

    gt_vis_oid_set = set(gt_vis_oids)
    pred_vis_oid_set = set(pred_vis_oids)
    missed = gt_vis_oid_set - pred_vis_oid_set
    score_map = {int(oid): float(score) for oid, score in zip(obj_ids, obj_scores)}
    print(f"    [DEBUG] matched-object visibility  ({len(centroids)} objects):")
    print(f"    {'oid':<8} {'label':<22} {'score':>7} {'d_GT':>7} {'GT':>4} {'d_pred':>8} {'pred':>5}")
    for oid in sorted(centroids):
        centre = centroids[oid]
        label = obj_labels.get(oid, "?")
        score = score_map.get(oid, float("nan"))
        d_gt   = float(np.linalg.norm(centre - gt_cam))
        d_pred = float(np.linalg.norm(centre - pred_cam_primary))
        v_gt   = "YES" if oid in gt_vis_oid_set   else "no"
        v_pred = "YES" if oid in pred_vis_oid_set else "no"
        print(f"    {oid:<8} {label:<22} {score:>7.3f} {d_gt:>6.2f}m {v_gt:>4} {d_pred:>7.2f}m {v_pred:>5}")
    shared = gt_vis_oid_set & pred_vis_oid_set
    missed = gt_vis_oid_set - pred_vis_oid_set


    print(f"    [DEBUG] GT sees {len(gt_vis_oids)}/{len(centroids)} | "
          f"pred sees {len(pred_vis_oids)}/{len(centroids)} | "
          f"shared {len(shared)} | "
          f"missed by pred: {len(missed)}")
    if missed:
        print(f"    {'oid':<8} {'label':<22} {'d_GT':>7} {'d_pred':>8}")
        for oid in sorted(missed):
            centre = centroids[oid]
            label  = obj_labels.get(oid, "?")
            d_gt   = float(np.linalg.norm(centre - gt_cam))
            d_pred = float(np.linalg.norm(centre - pred_cam_primary))
            print(f"    {oid:<8} {label:<22} {d_gt:>6.2f}m {d_pred:>7.2f}m")
    # ---------------------------------------------------------------------------

    iou_val, iou_err, gt_vis_set, pred_vis_set = compute_view_iou_error(
        gt_cam, gt_dir,
        pred_cam_primary, pred_dir_primary,
        hfov=hfov_rad,
        vfov=vfov_rad,
        rc=rc,
        geom_id=int(mesh_id),
        tri_pts=tri_pts,
        tri_centroids=tri_centroids,
        tri_areas=tri_areas,
        near=0.05,
        far=None,
    )
    metrics.iou_error = iou_err
    if iou_val is not None and iou_err is not None:
        print(f"    view IoU: {iou_val:.3f} | IoU error: {iou_err:.3f}\n")
    else:
        print("    view IoU: n/a (missing direction or empty visibility)\n")

    if args.show_heatmap:
        plt.figure(figsize=(6.5, 6.2))
        sc = plt.scatter(cams[:, 0], cams[:, 1], c=grid_probs,
                         cmap="viridis", s=14)
        plt.colorbar(sc, label="Probability")
        plt.axis("equal")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(f"{scene_id} · {metrics.frame_id} · grid {args.grid_step:.2f} m")
        add_heatmap_markers(gt_cam,
                            pred_grid=pred_cam_grid,
                            pred_arrow=pred_cam_arrow,
                            label_grid=f"Pred grid ({args.prediction_strategy})",
                            label_arrow=f"Pred arrow ({args.prediction_strategy})")
        plt.tight_layout()
        plt.show()

    if args.show_arrows:
        if arrow_probs is not None and arrow_positions:
            max_len = (0.9 * args.grid_step) if args.arrow_len <= 0 else args.arrow_len
            W_np = np.asarray(arrow_probs, dtype=np.float32)
            scale = np.where(W_np > 0, W_np / W_np.max(), 0.0)
            dirs_xy = np.asarray([d[:2] for d in arrow_dirs], dtype=np.float32)
            norms = np.linalg.norm(dirs_xy, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            dirs_xy /= norms
            U_np = dirs_xy[:, 0] * max_len * scale
            V_np = dirs_xy[:, 1] * max_len * scale
            Qx = [float(p[0]) for p in arrow_positions]
            Qy = [float(p[1]) for p in arrow_positions]

            plt.figure(figsize=(7, 6.5))
            plt.quiver(Qx, Qy, U_np, V_np, W_np,
                       angles="xy", scale_units="xy", scale=1.0,
                       cmap="viridis", width=0.004, minlength=0.01)
            plt.colorbar(label="FOV probability")
            plt.axis("equal")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.title(f"{scene_id} · {metrics.frame_id} · FOV arrows "
                      f"(H={math.degrees(hfov_rad):.0f}°, V={math.degrees(vfov_rad):.0f}°)")
            add_arrow_markers(gt_cam,
                              pred_grid=pred_cam_grid,
                              pred_arrow=pred_cam_arrow)
            plt.tight_layout()
            plt.show()
        else:
            print("    [info] Arrow plot skipped (no valid FOV windows).")

    if args.show_3d:
        matched_set: set[int] = {int(o) for o in obj_ids}
        frustum_scale = max(args.grid_step * 3.0, 0.6)
        try:
            mesh_vis, obj_stats = build_segmented_mesh(scene_dir, seed=42)
            colours = np.asarray(mesh_vis.vertex_colors)
            highlight = np.array([1.0, 0.3, 0.3], dtype=np.float64)
            for stats in obj_stats:
                oid = int(stats["object_id"])
                if oid in matched_set:
                    idx = stats.get("vertex_indices")
                    if idx is not None:
                        colours[idx] = np.clip(0.55 * colours[idx] + 0.45 * highlight, 0.0, 1.0)
            mesh_vis.vertex_colors = o3d.utility.Vector3dVector(colours)
            if not mesh_vis.has_vertex_normals():
                mesh_vis.compute_vertex_normals()
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] Segment mesh loading failed ({exc}) — falling back to legacy mesh.")
            mesh_vis = colour_objects(mesh, obj2faces, obj_ids)
            obj_stats = []
        if not mesh_vis.has_vertex_normals():
            mesh_vis.compute_vertex_normals()

        from open3d.visualization import gui, rendering

        global GUI_INITIALISED
        if not GUI_INITIALISED:
            gui.Application.instance.initialize()
            GUI_INITIALISED = True

        vis = o3d.visualization.O3DVisualizer(f"{scene_id} – localisation eval", 1280, 800)
        vis.show_settings = False

        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        vis.add_geometry("mesh", mesh_vis, material)

        text_added = set()
        if obj_stats:
            bbox_material = rendering.MaterialRecord()
            bbox_material.shader = "unlitLine"
            bbox_material.line_width = 1.5
            for stats in obj_stats:
                oid = int(stats["object_id"])
                label = stats.get("label") or f"id_{oid}"
                centroid = np.asarray(stats["centroid"]) if "centroid" in stats else None
                if centroid is not None and tuple(centroid) not in text_added:
                    vis.add_3d_label(centroid, f"{oid}: {label}")
                    text_added.add(tuple(centroid))
                if oid in matched_set and "bbox" in stats:
                    vis.add_geometry(f"bbox_{oid}", stats["bbox"], bbox_material)

        # Probability spheres
        prob_material = rendering.MaterialRecord()
        prob_material.shader = "defaultLit"
        prob_material.base_color = [1.0, 1.0, 1.0, 1.0]
        for idx_point, (point, colour) in enumerate(zip(cams, colormap(grid_probs))):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
            s.translate(point)
            s.paint_uniform_color(colour)
            if not s.has_vertex_normals():
                s.compute_vertex_normals()
            vis.add_geometry(f"prob_{idx_point}", s, prob_material)

        gt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        gt_sphere.translate(gt_cam)
        gt_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        if not gt_sphere.has_vertex_normals():
            gt_sphere.compute_vertex_normals()
        vis.add_geometry("gt_cam", gt_sphere, material)
        vis.add_3d_label(gt_cam, "GT")

        pred_sphere_grid = o3d.geometry.TriangleMesh.create_sphere(radius=0.085)
        pred_sphere_grid.translate(pred_cam_grid)
        pred_sphere_grid.paint_uniform_color([1.0, 0.9, 0.0])
        if not pred_sphere_grid.has_vertex_normals():
            pred_sphere_grid.compute_vertex_normals()
        vis.add_geometry("pred_cam_grid", pred_sphere_grid, material)
        vis.add_3d_label(pred_cam_grid, f"Pred grid ({args.prediction_strategy})")

        pred_sphere_arrow = None
        if pred_cam_arrow is not None:
            pred_sphere_arrow = o3d.geometry.TriangleMesh.create_sphere(radius=0.082)
            pred_sphere_arrow.translate(pred_cam_arrow)
            pred_sphere_arrow.paint_uniform_color([0.1, 0.8, 0.9])
            if not pred_sphere_arrow.has_vertex_normals():
                pred_sphere_arrow.compute_vertex_normals()
            vis.add_geometry("pred_cam_arrow", pred_sphere_arrow, material)
            vis.add_3d_label(pred_cam_arrow, f"Pred arrow ({args.prediction_strategy})")

        pred_sphere_primary = pred_sphere_arrow if pred_sphere_arrow is not None else pred_sphere_grid

        frustum_mat = None
        frustum_mat_pred = None
        frustum_gt = create_camera_frustum(gt_cam, gt_dir,
                                           colour=(1.0, 0.0, 0.0),
                                           h_fov=hfov_rad,
                                           v_fov=vfov_rad,
                                           scale=frustum_scale)
        pred_colour = (0.1, 0.8, 0.9) if pred_cam_arrow is not None else (1.0, 0.9, 0.0)
        frustum_pred = create_camera_frustum(pred_cam_primary, pred_dir_primary,
                                             colour=pred_colour,
                                             h_fov=hfov_rad,
                                             v_fov=vfov_rad,
                                             scale=frustum_scale)
        if frustum_gt is not None:
            frustum_mat = rendering.MaterialRecord()
            frustum_mat.shader = "unlitLine"
            frustum_mat.line_width = 2.0
            frustum_mat.base_color = [1.0, 0.0, 0.0, 1.0]
            vis.add_geometry("frustum_gt", frustum_gt, frustum_mat)
        if frustum_pred is not None:
            frustum_mat_pred = rendering.MaterialRecord()
            frustum_mat_pred.shader = "unlitLine"
            frustum_mat_pred.line_width = 2.0
            frustum_mat_pred.base_color = [1.0, 0.9, 0.0, 1.0]
            vis.add_geometry("frustum_pred", frustum_pred, frustum_mat_pred)

        vis.reset_camera_to_default()
        gui.Application.instance.add_window(vis)

        # Second window: IoU-only overlays (GT-only, Pred-only, intersection)
        vis_iou = o3d.visualization.O3DVisualizer(f"{scene_id} – IoU overlap", 1280, 800)
        vis_iou.show_settings = False
        base_mat = rendering.MaterialRecord()
        base_mat.shader = "defaultLitTransparency"
        base_mat.base_color = [0.8, 0.8, 0.8, 0.18]
        vis_iou.add_geometry("mesh_base", mesh, base_mat)

        def _subset_mesh(base: o3d.geometry.TriangleMesh,
                         tris_idx: set[int],
                         colour: Tuple[float, float, float],
                         alpha: float) -> Optional[o3d.geometry.TriangleMesh]:
            """Create a coloured submesh from a set of triangle indices; None if empty."""
            if not tris_idx:
                return None

            idx_arr = np.asarray(sorted(tris_idx), dtype=np.int64)
            verts = np.asarray(base.vertices)
            tris_arr = np.asarray(base.triangles)[idx_arr]

            uniq, inv = np.unique(tris_arr.reshape(-1), return_inverse=True)
            new_verts = verts[uniq]
            new_tris = inv.reshape(-1, 3)

            sub = o3d.geometry.TriangleMesh()
            sub.vertices = o3d.utility.Vector3dVector(new_verts)
            sub.triangles = o3d.utility.Vector3iVector(new_tris)
            sub.paint_uniform_color(colour)
            if not sub.has_vertex_normals():
                sub.compute_vertex_normals()
            return sub

        gt_only = gt_vis_set - pred_vis_set
        pred_only = pred_vis_set - gt_vis_set
        both = gt_vis_set & pred_vis_set
        overlays = [
            ("iou_gt_only", gt_only, (1.0, 0.0, 0.0), 0.65),         # GT-only red
            ("iou_pred_only", pred_only, (1.0, 0.85, 0.0), 0.65),    # Pred-only yellow
            ("iou_both", both, (1.0, 0.4, 0.0), 0.85),               # Intersection orange
        ]
        for name, tri_set, colour, alpha in overlays:
            mesh_subset = _subset_mesh(mesh, tri_set, colour, alpha)
            if mesh_subset is None:
                continue
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLitTransparency"
            mat.base_color = [*colour, alpha]
            vis_iou.add_geometry(name, mesh_subset, mat)

        vis_iou.add_geometry("gt_cam_iou", gt_sphere, material)
        vis_iou.add_geometry("pred_cam_iou", pred_sphere_primary, material)
        if frustum_gt is not None:
            vis_iou.add_geometry("frustum_gt_iou", frustum_gt, frustum_mat)
        if frustum_pred is not None:
            vis_iou.add_geometry("frustum_pred_iou", frustum_pred, frustum_mat_pred)
        vis_iou.reset_camera_to_default()
        gui.Application.instance.add_window(vis_iou)
        gui.Application.instance.run()

    return metrics


def main() -> None:
    args = parse_args()
    args.hit_radii = [float(r) for r in args.hit_radii]
    args.mass_percentiles = [float(p) for p in args.mass_percentiles]
    params_text = format_args_section(args)
    rng = np.random.default_rng(seed=args.seed)

    scenes = load_scene_graphs(args.graphs, use_attributes=args.scene_use_attributes)

    candidate_ids = list(scenes.keys())
    if args.visualize_scene:
        if args.scene_ids:
            print("[WARN] --visualize_scene overrides --scene_ids.")
        if args.visualize_scene not in scenes:
            print(f"[ERROR] Requested scene '{args.visualize_scene}' not found in processed graphs.")
            return
        candidate_ids = [args.visualize_scene]
    elif args.scene_ids:
        scene_set = set(args.scene_ids)
        candidate_ids = [sid for sid in candidate_ids if sid in scene_set]
    else:
        # Filter by available descriptions
        query_root = ensure_query_root(args.query_root, Path(args.root))
        candidate_ids = [
            sid for sid in candidate_ids
            if (query_root / sid / "output" / "descriptions").exists()
            or (Path(args.root) / sid / "output" / "descriptions").exists()
        ]

    candidate_ids.sort()
    if args.max_scenes is not None:
        candidate_ids = candidate_ids[: args.max_scenes]

    print(f"Evaluating {len(candidate_ids)} scene(s)...\n")

    metrics_list: List[SceneMetrics] = []
    for idx, sid in enumerate(candidate_ids, start=1):
        print(f"[{idx:03d}/{len(candidate_ids):03d}] {sid}")
        scene_metrics = evaluate_scene(sid, scenes[sid], args, rng)
        if scene_metrics is None:
            continue
        metrics_list.append(scene_metrics)
        print(f"    frame: {scene_metrics.frame_id}")
        print(f"    matches: {scene_metrics.matched_objects} | grid pts: {scene_metrics.grid_points}")
        hit_line = " | ".join(
            f"hit@{r:.2f}m: {scene_metrics.hit_masses.get(r, 0.0):.3f}"
            for r in sorted(scene_metrics.hit_masses)
        )
        print(f"    {hit_line}")
        mass_line = " | ".join(
            f"R{p:.0f}%: {scene_metrics.mass_radii.get(p, float('nan')):.3f} m"
            for p in sorted(scene_metrics.mass_radii)
        )
        if mass_line:
            print(f"    mass-radius: {mass_line}")
        ang_err = ("n/a" if scene_metrics.angular_error_deg is None
                   else f"{scene_metrics.angular_error_deg:.2f}°")
        print(f"    topK{args.top_k_min_dist} min dist: {scene_metrics.topk_min_dist:.3f} m | "
              f"dist_err: {scene_metrics.distance_error:.3f} m | ang_err: {ang_err}\n")
        if scene_metrics.iou_error is not None:
            print(f"    view IoU error: {scene_metrics.iou_error:.3f}")

    if not metrics_list:
        print("No scenes produced metrics. Nothing to report.")
        if args.log_file:
            args.log_file.parent.mkdir(parents=True, exist_ok=True)
            payload = "No scenes produced metrics.\n\n" + params_text + "\n"
            args.log_file.write_text(payload)
            print(f"Empty summary logged to {args.log_file}")
        return

    table_text = build_metrics_table(metrics_list,
                                     args.hit_radii,
                                     args.mass_percentiles,
                                     args.top_k_min_dist)
    if table_text:
        print("Scene-level summary table -------------------------------")
        print(table_text)
        print("---------------------------------------------------------\n")

    # Aggregate metrics
    def agg(values: List[float]) -> Tuple[float, float]:
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(np.median(arr))

    hit_stats: Dict[float, Tuple[float, float]] = {}
    for r in sorted(set(args.hit_radii)):
        vals = [m.hit_masses.get(r, 0.0) for m in metrics_list]
        hit_stats[r] = agg(vals)

    mass_radius_stats: Dict[float, Tuple[float, float]] = {}
    for p in sorted(set(args.mass_percentiles)):
        vals = [m.mass_radii.get(p, float("nan")) for m in metrics_list]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            mass_radius_stats[p] = agg(vals)

    mean_topk, med_topk = agg([m.topk_min_dist for m in metrics_list])
    mean_err, med_err = agg([m.distance_error for m in metrics_list])
    ang_values = [m.angular_error_deg for m in metrics_list if m.angular_error_deg is not None]
    mean_ang: Optional[float] = None
    med_ang: Optional[float] = None
    if ang_values:
        mean_ang, med_ang = agg([float(v) for v in ang_values])
    iou_err_values = [m.iou_error for m in metrics_list if m.iou_error is not None]
    mean_iou_err: Optional[float] = None
    med_iou_err: Optional[float] = None
    if iou_err_values:
        mean_iou_err, med_iou_err = agg([float(v) for v in iou_err_values])

    agg_lines = [
        "Aggregate metrics ---------------------------------------",
        f"  TopK{args.top_k_min_dist} min dist (m): mean={mean_topk:.3f} | median={med_topk:.3f}",
        f"  Distance error (m)      : mean={mean_err:.3f} | median={med_err:.3f}",
        "---------------------------------------------------------\n",
    ]
    for r in sorted(hit_stats):
        mean_hit, med_hit = hit_stats[r]
        agg_lines.insert(-1,
                         f"  Hit@{r:.2f}m              : mean={mean_hit:.3f} | median={med_hit:.3f}")
    for p in sorted(mass_radius_stats):
        mean_r, med_r = mass_radius_stats[p]
        agg_lines.insert(-1,
                         f"  Mass-radius R{p:.0f}% (m): mean={mean_r:.3f} | median={med_r:.3f}")
    if mean_ang is not None and med_ang is not None:
        agg_lines.insert(-1,
                         f"  Angular error (deg)   : mean={mean_ang:.2f} | median={med_ang:.2f}")
    if mean_iou_err is not None and med_iou_err is not None:
        agg_lines.insert(-1,
                         f"  View IoU error     : mean={mean_iou_err:.3f} | median={med_iou_err:.3f}")
    print("\n".join(agg_lines))

    log_sections: List[str] = [params_text]
    if table_text:
        log_sections.append("Scene-level summary table")
        log_sections.append(table_text)
    log_sections.append("\n".join(agg_lines))
    log_payload = "\n\n".join(log_sections).rstrip() + "\n"
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        args.log_file.write_text(log_payload)
        print(f"Metrics summary logged to {args.log_file}")

    if args.save_metrics:
        payload = [
            {
                "scene_id": m.scene_id,
                "frame_id": m.frame_id,
                "hit_masses": {str(k): v for k, v in m.hit_masses.items()},
                "mass_radii": {str(k): v for k, v in m.mass_radii.items()},
                "topk_min_dist": m.topk_min_dist,
                "distance_error": m.distance_error,
                "angular_error_deg": m.angular_error_deg,
                "iou_error": m.iou_error,
                "grid_points": m.grid_points,
                "matched_objects": m.matched_objects,
            }
            for m in metrics_list
        ]
        hit_mass_summary = {
            str(r): {"mean": mean_hit, "median": med_hit}
            for r, (mean_hit, med_hit) in hit_stats.items()
        }
        mass_radius_summary = {
            str(p): {"mean": mean_r, "median": med_r}
            for p, (mean_r, med_r) in mass_radius_stats.items()
        }
        args.save_metrics.write_text(json.dumps({
            "metrics": payload,
            "aggregate": {
                "hit_masses": hit_mass_summary,
                "mass_radii": mass_radius_summary,
                "topk_min_dist": {"mean": mean_topk, "median": med_topk},
                "distance_error": {"mean": mean_err, "median": med_err},
                "angular_error_deg": (None if mean_ang is None
                                      else {"mean": mean_ang, "median": med_ang}),
                "iou_error": None if mean_iou_err is None else {"mean": mean_iou_err, "median": med_iou_err},
                "hit_radii": args.hit_radii,
                "mass_percentiles": args.mass_percentiles,
                "top_k_min_dist": args.top_k_min_dist,
                "top_k": args.top_k,
                "grid_step": args.grid_step,
            },
        }, indent=2))
        print(f"Metrics saved to {args.save_metrics}")


GUI_INITIALISED = False

if __name__ == "__main__":
    main()
