#!/usr/bin/env python3
"""
3a_pose_level_dialogue_system_eval.py

SYSTEM-LEVEL interactive evaluation: each backend runs its OWN dialogue (its own question policy),
then you see A1/A2/A3 results side-by-side, plus baseline predicted_pose vs gt_pose.

Backends
--------
A1) Candidate posterior (discrete) using cand->frame mapping W and frame semantics
A2) Particle filter (continuous) using KNN-to-frames proxy semantics
A3) Frame posterior (discrete over frames) using exact frame semantics

Improvements included (from your analysis)
-----------------------------------------
✅ Sequential (independent) dialogues: A1 then A2 then A3 (order configurable)
✅ Baseline error: predicted_pose vs gt_pose (always printed)
✅ IG-based question selection option (--question_strategy ig) + safer fallback if thresholds over-prune
✅ IDF penalty for common labels (downweights floor/wall/etc) + ignore label list
✅ Label likelihood uses salience (pixel_percent / score) rather than pure binary (when available)
✅ Treat label answer 'u' as informative (unanswerable/unsure) instead of ignoring it
✅ Optionally include predicted_pose as an extra candidate hypothesis (helps when baseline is strong)

Requires
--------
- pose_level_dialogue_semantic_fallback.py (your dsf) in same folder or PYTHONPATH
- dataset_root/<scene_id>/output/descriptions/all_descriptions*.json readable by dsf.load_scene_data
- candidates_json with gt_pose, predicted_pose, and candidate lists

Run (example)
-------------
python /Users/abu/Desktop/LangLoc/3a_pose_level_dialogue.py \
  --candidates_json "/Users/abu/Desktop/LangLoc/abu_eval_pose_candidates.json" \
  --dataset_root "/Users/abu/Desktop/LangLoc/3RScan" \
  --only_scene_id "2e36952b-e133-204c-911e-7644cb34e8b2" \
  --limit 1 \
  --candidate_set fov \
  --eval_mode sequential \
  --question_strategy ig \
  --ask_min_p 0.01 --ask_max_p 0.99 --rel_min_answerable 0.1 \
  --max_pool_frames 30 \
  --include_predicted_pose \
  --show_gt_debug
"""

from __future__ import annotations

import argparse
import inspect
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import pose_level_dialogue_semantic_fallback as dsf


# -----------------------
# Signature-safe calls
# -----------------------
def call_with_supported_kwargs(fn, *args, **kwargs):
    sig = inspect.signature(fn)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(*args, **kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(*args, **filtered)


# -----------------------
# CandToFrameMap -> dense
# -----------------------
def c2f_to_dense(c2f_map, num_cands: int, num_frames: int, eps: float = 1e-12) -> np.ndarray:
    if isinstance(c2f_map, np.ndarray):
        M = np.asarray(c2f_map, dtype=np.float64)
    else:
        M = None

        # matrix-like
        for name in ("matrix", "mat", "M", "dense", "c2f", "weights_matrix", "to_dense", "as_matrix", "to_matrix", "toarray"):
            if hasattr(c2f_map, name):
                obj = getattr(c2f_map, name)
                try:
                    M = obj() if callable(obj) else obj
                    M = np.asarray(M, dtype=np.float64)
                    break
                except Exception:
                    M = None

        # sparse-like (idx + weights)
        if M is None:
            idx = None
            wts = None
            for iname in ("idx", "indices", "nn_idx", "knn_idx", "frame_idx", "frame_indices"):
                if hasattr(c2f_map, iname):
                    idx = getattr(c2f_map, iname)
                    break
            for wname in ("w", "weights", "nn_w", "knn_w", "vals", "values", "p"):
                if hasattr(c2f_map, wname):
                    wts = getattr(c2f_map, wname)
                    break

            if idx is not None and wts is not None:
                idx = np.asarray(idx, dtype=np.int64)
                wts = np.asarray(wts, dtype=np.float64)
                if idx.ndim == 1:
                    idx = idx[:, None]
                if wts.ndim == 1:
                    wts = wts[:, None]
                M = np.zeros((num_cands, num_frames), dtype=np.float64)
                n = min(num_cands, idx.shape[0])
                K = idx.shape[1]
                for i in range(n):
                    for k in range(K):
                        j = int(idx[i, k])
                        if 0 <= j < num_frames:
                            M[i, j] += float(wts[i, k])

    if M is None:
        raise TypeError(f"Could not convert CandToFrameMap to dense matrix: type={type(c2f_map)}")

    if M.shape == (num_frames, num_cands):
        M = M.T

    M = M[:num_cands, :num_frames]
    rs = M.sum(axis=1, keepdims=True)
    M = M / np.maximum(rs, eps)
    return M


# -----------------------
# Math helpers
# -----------------------
def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > eps else np.zeros(3, dtype=np.float64)


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = _normalize(u)
    v = _normalize(v)
    if float(np.linalg.norm(u)) < 1e-9 or float(np.linalg.norm(v)) < 1e-9:
        return float("nan")
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def pose_errors(pred_pos, pred_dir, gt_pos, gt_dir) -> Tuple[float, float]:
    pred_pos = np.asarray(pred_pos, dtype=np.float64).reshape(3)
    gt_pos = np.asarray(gt_pos, dtype=np.float64).reshape(3)
    pos_err = float(np.linalg.norm(pred_pos - gt_pos))
    if pred_dir is None or gt_dir is None:
        rot_err = float("nan")
    else:
        rot_err = angle_deg(pred_dir, gt_dir)
    return pos_err, rot_err


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / float(p.sum())
    return float(-np.sum(p * np.log(p)))


def bayes_update(p: np.ndarray, like: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    post = p * np.clip(like, eps, None)
    s = float(post.sum())
    if s <= 0:
        return np.ones_like(p) / len(p)
    return post / s


# -----------------------
# Pose getters (gt/pred)
# -----------------------
def get_pose(entry: Dict[str, Any], key: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    d = entry.get(key, None) or {}
    pos = d.get("position", None)
    direc = d.get("direction", None)
    if pos is None:
        return None, None, d
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    if direc is None:
        return pos, None, d
    direc = np.asarray(direc, dtype=np.float64).reshape(3)
    return pos, direc, d


# -----------------------
# Robust frame semantics
# -----------------------
def _to_prob01(x: float) -> float:
    x = float(x)
    if x > 1.0:
        # likely percentage
        x = x / 100.0
    return float(np.clip(x, 0.0, 1.0))


def frame_label_salience(fr) -> Dict[str, float]:
    """
    Returns dict label->salience in [0,1] if possible.
    If only binary info available, returns 1.0 for present labels.
    """
    # Case 1: visible_labels is dict label->score
    if hasattr(fr, "visible_labels"):
        v = getattr(fr, "visible_labels")
        if isinstance(v, dict):
            out = {}
            for k, val in v.items():
                lab = str(k).strip().lower()
                if isinstance(val, (int, float)):
                    out[lab] = _to_prob01(val)
                elif isinstance(val, dict):
                    # try pixel_percent or score
                    if "pixel_percent" in val:
                        out[lab] = _to_prob01(val["pixel_percent"])
                    elif "score" in val:
                        out[lab] = _to_prob01(val["score"])
                    else:
                        out[lab] = 1.0
                else:
                    out[lab] = 1.0
            return out
        if isinstance(v, (list, tuple, set)):
            return {str(x).strip().lower(): 1.0 for x in v}

    # Case 2: visible_objects might be dict id->info{label,pixel_percent}
    if hasattr(fr, "visible_objects"):
        v = getattr(fr, "visible_objects")
        if isinstance(v, dict):
            out = {}
            for _, info in v.items():
                if isinstance(info, dict) and "label" in info:
                    lab = str(info["label"]).strip().lower()
                    px = info.get("pixel_percent", info.get("score", 1.0))
                    out[lab] = _to_prob01(px)
            if out:
                return out

    # Fallback: labels list
    for name in ("labels", "label_set", "objects"):
        if hasattr(fr, name):
            v = getattr(fr, name)
            if isinstance(v, (list, tuple, set)):
                return {str(x).strip().lower(): 1.0 for x in v}
            if isinstance(v, dict):
                out = {}
                for _, info in v.items():
                    if isinstance(info, dict) and "label" in info:
                        out[str(info["label"]).strip().lower()] = 1.0
                if out:
                    return out

    return {}


def _rel_to_tuple(rel_item) -> Optional[Tuple[str, str, str]]:
    if rel_item is None:
        return None
    if hasattr(rel_item, "subj") and hasattr(rel_item, "rel") and hasattr(rel_item, "obj"):
        return (str(rel_item.subj).strip().lower(), str(rel_item.rel).strip().lower(), str(rel_item.obj).strip().lower())
    if isinstance(rel_item, dict):
        s = rel_item.get("subj") or rel_item.get("subject")
        r = rel_item.get("rel") or rel_item.get("relation") or rel_item.get("predicate")
        o = rel_item.get("obj") or rel_item.get("object")
        if s and r and o:
            return (str(s).strip().lower(), str(r).strip().lower(), str(o).strip().lower())
        return None
    if isinstance(rel_item, (tuple, list)) and len(rel_item) >= 3:
        if isinstance(rel_item[1], str):
            return (str(rel_item[0]).strip().lower(), str(rel_item[1]).strip().lower(), str(rel_item[2]).strip().lower())
        if isinstance(rel_item[2], str):
            return (str(rel_item[0]).strip().lower(), str(rel_item[2]).strip().lower(), str(rel_item[1]).strip().lower())
    return None


def frame_relations(fr) -> set:
    for name in ("rels", "relations", "spatial_relations"):
        if hasattr(fr, name):
            v = getattr(fr, name)
            if isinstance(v, dict):
                out = set()
                for _, it in v.items():
                    t = _rel_to_tuple(it)
                    if t:
                        out.add(t)
                return out
            if isinstance(v, (list, tuple, set)):
                out = set()
                for it in v:
                    t = _rel_to_tuple(it)
                    if t:
                        out.add(t)
                return out
    return set()


def relation_phrase(rel: str) -> str:
    if hasattr(dsf, "relation_to_phrase"):
        try:
            return dsf.relation_to_phrase(rel)
        except Exception:
            pass
    return rel


# -----------------------
# Candidates (optionally include predicted_pose)
# -----------------------
def extract_candidates(
    entry: Dict[str, Any],
    candidate_set: str,
    include_predicted_pose: bool,
    pred_prior: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    cs = (candidate_set or "auto").lower().strip()
    grid = entry.get("grid_point_candidates") or []
    fov = entry.get("fov_pose_candidates") or []

    if cs == "auto":
        cs = "grid" if len(grid) else "fov"

    if cs == "grid":
        cands = list(grid)
        mode = "grid"
    elif cs == "fov":
        cands = list(fov)
        mode = "fov"
    elif cs == "both":
        cands = list(fov) + list(grid)
        mode = "both"
    else:
        raise ValueError(f"Unknown candidate_set: {candidate_set}")

    # add predicted pose as extra hypothesis
    if include_predicted_pose:
        pp = entry.get("predicted_pose", None)
        if isinstance(pp, dict) and "position" in pp:
            cands.append(
                {
                    "position": pp.get("position"),
                    "direction": pp.get("direction", None),
                    "prob": float(pred_prior),
                    "visible_count": float(pred_prior),
                    "_is_predicted_pose": True,
                }
            )

    if not cands:
        raise ValueError(f"No candidates found (grid={len(grid)}, fov={len(fov)}).")

    pos_list: List[List[float]] = []
    dir_list: List[List[float]] = []
    prior_list: List[float] = []  # type: ignore

    prior_list = []

    for c in cands:
        if not isinstance(c, dict) or "position" not in c:
            continue
        pos_list.append(list(c["position"]))

        d = c.get("direction", None)
        if d is None:
            dir_list.append([0.0, 0.0, 0.0])
        else:
            dir_list.append(list(d))

        if c.get("_is_predicted_pose", False):
            prior_list.append(float(pred_prior))
        else:
            if mode == "grid":
                prior_list.append(float(c.get("prob", 1.0)))
            elif mode == "fov":
                prior_list.append(float(c.get("visible_count", c.get("prob", 1.0))))
            else:
                prior_list.append(float(c.get("visible_count", c.get("prob", 1.0))))

    cand_pos = np.asarray(pos_list, dtype=np.float32)
    cand_dir = np.asarray(dir_list, dtype=np.float32)

    if float(np.linalg.norm(cand_dir, axis=1).max()) < 1e-6:
        cand_dir_out = None
    else:
        cand_dir_out = cand_dir / np.maximum(np.linalg.norm(cand_dir, axis=1, keepdims=True), 1e-6)

    prior = np.asarray(prior_list, dtype=np.float64)
    prior = np.maximum(prior, 0.0)
    prior = prior / max(float(prior.sum()), 1e-12)
    return cand_pos, cand_dir_out, prior


# -----------------------
# Question representation
# -----------------------
@dataclass(frozen=True)
class Question:
    qtype: str  # "label" or "rel"
    idx: int    # index into pool


HELP_TEXT = """Commands:
  y  yes
  n  no
  u  unknown/skip
  q  quit this entry
  tf show top frames posterior (this backend)
  tc show top candidates (A1 only)
  tp show top particles (A2 only)
  o  show pool sizes
  h  help
"""


def rel_item_to_tuple(rel_item: Any) -> Tuple[str, str, str]:
    t = _rel_to_tuple(rel_item)
    if t is None:
        raise ValueError(f"Could not parse relation item: {rel_item}")
    return t


# -----------------------
# Pools + IDF
# -----------------------
def build_pools(
    frames_all: Sequence[Any],
    frame_subset: Sequence[int],
    max_rel_pool: int,
    rel_min_salience: float,
    rel_unique_only: bool,
    allowed_rels: Sequence[str],
) -> Tuple[List[str], List[Any]]:
    frames_sub = [frames_all[i] for i in frame_subset]

    # labels
    try:
        label_pool = list(dsf.label_pool_from_frames(frames_all, frame_subset))
        label_pool = [str(x).strip().lower() for x in label_pool]
    except Exception:
        s = set()
        for fr in frames_sub:
            s |= set(frame_label_salience(fr).keys())
        label_pool = sorted(list(s))

    # relations
    rel_pool = call_with_supported_kwargs(
        getattr(dsf, "rel_pool_from_frames"),
        frames_all,
        frame_subset,
        max_rel=max_rel_pool,
        min_salience=rel_min_salience,
        unique_only=rel_unique_only,
    )
    rel_pool = list(rel_pool)

    if allowed_rels:
        allow = set(map(lambda x: str(x).strip().lower(), allowed_rels))
        filtered = []
        for t in rel_pool:
            tup = _rel_to_tuple(t)
            if tup and tup[1] in allow:
                filtered.append(t)
        rel_pool = filtered

    return label_pool, rel_pool


def compute_label_idf(label_pool: List[str], frame_label_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    F = len(frame_label_dicts)
    df = {lab: 0 for lab in label_pool}
    for d in frame_label_dicts:
        for lab in d.keys():
            if lab in df:
                df[lab] += 1
    # idf = log((F+1)/(df+1))
    out = {}
    for lab in label_pool:
        out[lab] = float(math.log((F + 1.0) / (df.get(lab, 0) + 1.0)))
    return out


# -----------------------
# UI helpers
# -----------------------
def show_top_frames(frames: Sequence[Any], pf: np.ndarray, top_n: int = 5, title: str = "Top frames"):
    order = np.argsort(-pf)[: min(top_n, len(pf))]
    print(title + ":")
    for k, j in enumerate(order, 1):
        fr = frames[int(j)]
        fid = getattr(fr, "frame_id", str(j))
        pos = getattr(fr, "position", getattr(fr, "pos", None))
        pos_list = pos.tolist() if hasattr(pos, "tolist") else (list(pos) if pos is not None else None)
        print(f"  {k:>2}. {fid:<12} P={pf[int(j)]:.3f} pos={pos_list}")


def show_top_candidates(cand_pos: np.ndarray, cand_dir: Optional[np.ndarray], pc: np.ndarray, top_n: int = 5):
    order = np.argsort(-pc)[: min(top_n, len(pc))]
    print("Top candidates:")
    for k, i in enumerate(order, 1):
        d = None if cand_dir is None else cand_dir[int(i)].tolist()
        print(f"  {k:>2}. idx={int(i):<4} P={pc[int(i)]:.3f} pos={cand_pos[int(i)].tolist()} dir={d}")


def show_top_particles(p_pos: np.ndarray, p_dir: np.ndarray, pw: np.ndarray, top_n: int = 5):
    order = np.argsort(-pw)[: min(top_n, len(pw))]
    print("Top particles:")
    for k, i in enumerate(order, 1):
        print(f"  {k:>2}. idx={int(i):<4} w={pw[int(i)]:.3f} pos={p_pos[int(i)].tolist()} dir={p_dir[int(i)].tolist()}")


# -----------------------
# Likelihood utilities (label salience + unknown)
# -----------------------
def salience_to_visprob(sal: float, tau: float) -> float:
    # map salience in [0,1] to visibility probability
    # if tau=0.2, then sal>=0.2 maps close to 1
    if tau <= 0:
        return float(np.clip(sal, 0.0, 1.0))
    return float(np.clip(sal / tau, 0.0, 1.0))


def salience_to_answerable(sal: float, tau: float) -> float:
    # answerable probability increases with salience
    if tau <= 0:
        return float(np.clip(sal, 0.0, 1.0))
    return float(np.clip(sal / tau, 0.0, 1.0))


def ynu_likelihood_from_prob(
    p_true: np.ndarray,
    p_ans: np.ndarray,
    alpha: float,
    p_u_base: float,
    p_u_unanswerable: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each hypothesis h:
      truth is Bernoulli with p_true[h]
      answerable with p_ans[h]
      unknown probability:
        Pu[h] = p_u_base*p_ans + p_u_unanswerable*(1-p_ans)
      then:
        P(y|h) = (1-Pu) * [alpha*p_true + (1-alpha)*(1-p_true)]
        P(n|h) = (1-Pu) * [(1-alpha)*p_true + alpha*(1-p_true)]
        P(u|h) = Pu
    """
    p_true = np.asarray(p_true, dtype=np.float64)
    p_ans = np.asarray(p_ans, dtype=np.float64)
    Pu = np.clip(p_u_base * p_ans + p_u_unanswerable * (1.0 - p_ans), 0.0, 1.0)

    Py = (1.0 - Pu) * (alpha * p_true + (1.0 - alpha) * (1.0 - p_true))
    Pn = (1.0 - Pu) * ((1.0 - alpha) * p_true + alpha * (1.0 - p_true))
    # numerical safety
    Pu = np.clip(Pu, eps, 1.0)
    Py = np.clip(Py, eps, 1.0)
    Pn = np.clip(Pn, eps, 1.0)
    return Py, Pn, Pu


# -----------------------
# Backends
# -----------------------
class CandidateBackendA1:
    def __init__(
        self,
        cand_pos: np.ndarray,
        cand_dir: Optional[np.ndarray],
        cand_prior: np.ndarray,
        c2f_pool: np.ndarray,                # (N, F_pool)
        frame_label_dicts: List[Dict[str, float]],
        frame_rel_sets: List[set],
        frame_dirs: np.ndarray,              # (F_pool,3)
        alpha_label: float,
        alpha_rel: float,
        p_u_label: float,
        p_u_rel: float,
        p_u_unanswerable: float,
        vis_tau: float,
        ans_tau: float,
        eps: float = 1e-12,
    ):
        self.cand_pos = cand_pos.astype(np.float64)
        self.cand_dir = None if cand_dir is None else cand_dir.astype(np.float64)
        self.p = cand_prior.astype(np.float64).copy()
        self.p = self.p / max(float(self.p.sum()), eps)

        self.W = np.asarray(c2f_pool, dtype=np.float64)
        self.W = self.W / np.maximum(self.W.sum(axis=1, keepdims=True), eps)

        self.frame_label_dicts = frame_label_dicts
        self.frame_rel_sets = frame_rel_sets
        self.frame_dirs = frame_dirs.astype(np.float64)

        self.alpha_label = float(alpha_label)
        self.alpha_rel = float(alpha_rel)
        self.p_u_label = float(p_u_label)
        self.p_u_rel = float(p_u_rel)
        self.p_u_unanswerable = float(p_u_unanswerable)
        self.vis_tau = float(vis_tau)
        self.ans_tau = float(ans_tau)
        self.eps = eps

        # precompute frame label probability and answerability per label on demand cache
        self._label_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def top_prob(self) -> float:
        return float(self.p.max())

    def posterior_vector(self) -> np.ndarray:
        return self.p

    def frame_posterior(self) -> np.ndarray:
        pf = self.W.T @ self.p
        s = float(pf.sum())
        return pf / s if s > self.eps else np.ones_like(pf) / len(pf)

    def _frame_label_arrays(self, label: str) -> Tuple[np.ndarray, np.ndarray]:
        # returns (p_vis_frame, p_ans_frame) over pooled frames
        label = str(label).strip().lower()
        if label in self._label_cache:
            return self._label_cache[label]
        p_vis = np.zeros(len(self.frame_label_dicts), dtype=np.float64)
        p_ans = np.zeros_like(p_vis)
        for j, d in enumerate(self.frame_label_dicts):
            sal = d.get(label, 0.0)
            p_vis[j] = salience_to_visprob(sal, self.vis_tau)
            p_ans[j] = salience_to_answerable(sal, self.ans_tau)
        self._label_cache[label] = (p_vis, p_ans)
        return p_vis, p_ans

    def label_prob_yes(self, label: str) -> Tuple[np.ndarray, np.ndarray]:
        # returns (p_yes per candidate, p_ans per candidate)
        p_vis_f, p_ans_f = self._frame_label_arrays(label)
        return self.W @ p_vis_f, self.W @ p_ans_f

    def rel_prob_true_and_answerable(self, triple: Tuple[str, str, str]) -> Tuple[np.ndarray, np.ndarray]:
        s, r, o = triple
        true_f = np.array([1.0 if triple in rels else 0.0 for rels in self.frame_rel_sets], dtype=np.float64)
        # answerable if subj and obj answerable (use p_ans from salience)
        p_s_vis, p_s_ans = self._frame_label_arrays(s)
        p_o_vis, p_o_ans = self._frame_label_arrays(o)
        ans_f = np.minimum(p_s_ans, p_o_ans)
        return self.W @ true_f, self.W @ ans_f

    def update_label(self, label: str, ans: str):
        p_true, p_ans = self.label_prob_yes(label)
        Py, Pn, Pu = ynu_likelihood_from_prob(
            p_true, p_ans,
            alpha=self.alpha_label,
            p_u_base=self.p_u_label,
            p_u_unanswerable=self.p_u_unanswerable,
            eps=self.eps,
        )
        like = Py if ans == "y" else (Pn if ans == "n" else Pu)
        self.p = bayes_update(self.p, like, eps=self.eps)

    def update_rel(self, triple: Tuple[str, str, str], ans: str):
        p_true, p_ans = self.rel_prob_true_and_answerable(triple)
        Py, Pn, Pu = ynu_likelihood_from_prob(
            p_true, p_ans,
            alpha=self.alpha_rel,
            p_u_base=self.p_u_rel,
            p_u_unanswerable=self.p_u_unanswerable,
            eps=self.eps,
        )
        like = Py if ans == "y" else (Pn if ans == "n" else Pu)
        self.p = bayes_update(self.p, like, eps=self.eps)

    def predict_pose(self) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
        i_map = int(np.argmax(self.p))
        map_pos = self.cand_pos[i_map].copy()
        mean_pos = (self.p[:, None] * self.cand_pos).sum(axis=0)

        if self.cand_dir is not None:
            map_dir = _normalize(self.cand_dir[i_map])
            mean_dir = _normalize((self.p[:, None] * self.cand_dir).sum(axis=0))
            return map_pos, map_dir, mean_pos, mean_dir

        # fallback: induced frame posterior -> direction
        pf = self.frame_posterior()
        map_dir = _normalize(self.frame_dirs[int(np.argmax(pf))])
        mean_dir = _normalize((pf[:, None] * self.frame_dirs).sum(axis=0))
        return map_pos, map_dir, mean_pos, mean_dir


class FrameBackendA3:
    def __init__(
        self,
        p0: np.ndarray,                      # (F_pool,)
        frames_pool: Sequence[Any],
        frame_label_dicts: List[Dict[str, float]],
        frame_rel_sets: List[set],
        frame_pos: np.ndarray,               # (F_pool,3)
        frame_dir: np.ndarray,               # (F_pool,3)
        alpha_label: float,
        alpha_rel: float,
        p_u_label: float,
        p_u_rel: float,
        p_u_unanswerable: float,
        vis_tau: float,
        ans_tau: float,
        eps: float = 1e-12,
    ):
        self.frames = list(frames_pool)
        self.p = np.asarray(p0, dtype=np.float64).copy()
        self.p = self.p / max(float(self.p.sum()), eps)

        self.frame_label_dicts = frame_label_dicts
        self.frame_rel_sets = frame_rel_sets
        self.frame_pos = frame_pos.astype(np.float64)
        self.frame_dir = frame_dir.astype(np.float64)

        self.alpha_label = float(alpha_label)
        self.alpha_rel = float(alpha_rel)
        self.p_u_label = float(p_u_label)
        self.p_u_rel = float(p_u_rel)
        self.p_u_unanswerable = float(p_u_unanswerable)
        self.vis_tau = float(vis_tau)
        self.ans_tau = float(ans_tau)
        self.eps = eps

        self._label_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def top_prob(self) -> float:
        return float(self.p.max())

    def posterior_vector(self) -> np.ndarray:
        return self.p

    def frame_posterior(self) -> np.ndarray:
        return self.p.copy()

    def _frame_label_arrays(self, label: str) -> Tuple[np.ndarray, np.ndarray]:
        label = str(label).strip().lower()
        if label in self._label_cache:
            return self._label_cache[label]
        p_vis = np.zeros(len(self.frame_label_dicts), dtype=np.float64)
        p_ans = np.zeros_like(p_vis)
        for j, d in enumerate(self.frame_label_dicts):
            sal = d.get(label, 0.0)
            p_vis[j] = salience_to_visprob(sal, self.vis_tau)
            p_ans[j] = salience_to_answerable(sal, self.ans_tau)
        self._label_cache[label] = (p_vis, p_ans)
        return p_vis, p_ans

    def label_prob_yes(self, label: str) -> Tuple[np.ndarray, np.ndarray]:
        return self._frame_label_arrays(label)

    def rel_prob_true_and_answerable(self, triple: Tuple[str, str, str]) -> Tuple[np.ndarray, np.ndarray]:
        s, r, o = triple
        true = np.array([1.0 if triple in rels else 0.0 for rels in self.frame_rel_sets], dtype=np.float64)
        _, a_s = self._frame_label_arrays(s)
        _, a_o = self._frame_label_arrays(o)
        ans = np.minimum(a_s, a_o)
        return true, ans

    def update_label(self, label: str, ans: str):
        p_true, p_ans = self.label_prob_yes(label)
        Py, Pn, Pu = ynu_likelihood_from_prob(
            p_true, p_ans,
            alpha=self.alpha_label,
            p_u_base=self.p_u_label,
            p_u_unanswerable=self.p_u_unanswerable,
            eps=self.eps,
        )
        like = Py if ans == "y" else (Pn if ans == "n" else Pu)
        self.p = bayes_update(self.p, like, eps=self.eps)

    def update_rel(self, triple: Tuple[str, str, str], ans: str):
        p_true, p_ans = self.rel_prob_true_and_answerable(triple)
        Py, Pn, Pu = ynu_likelihood_from_prob(
            p_true, p_ans,
            alpha=self.alpha_rel,
            p_u_base=self.p_u_rel,
            p_u_unanswerable=self.p_u_unanswerable,
            eps=self.eps,
        )
        like = Py if ans == "y" else (Pn if ans == "n" else Pu)
        self.p = bayes_update(self.p, like, eps=self.eps)

    def predict_pose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        topf = int(np.argmax(self.p))
        map_pos = self.frame_pos[topf].copy()
        map_dir = _normalize(self.frame_dir[topf])
        mean_pos = (self.p[:, None] * self.frame_pos).sum(axis=0)
        mean_dir = _normalize((self.p[:, None] * self.frame_dir).sum(axis=0))
        return map_pos, map_dir, mean_pos, mean_dir


class ParticleBackendA2:
    def __init__(
        self,
        cand_pos: np.ndarray,
        cand_dir: Optional[np.ndarray],
        cand_prior: np.ndarray,
        frame_label_dicts: List[Dict[str, float]],
        frame_rel_sets: List[set],
        frame_pos: np.ndarray,            # (F_pool,3)
        frame_dir: np.ndarray,            # (F_pool,3)
        n_particles: int,
        k_nn: int,
        sigma: float,
        jitter_pos: float,
        alpha_label: float,
        alpha_rel: float,
        p_u_label: float,
        p_u_rel: float,
        p_u_unanswerable: float,
        vis_tau: float,
        ans_tau: float,
        seed: int = 0,
        eps: float = 1e-12,
    ):
        self.frame_label_dicts = frame_label_dicts
        self.frame_rel_sets = frame_rel_sets
        self.frame_pos = np.asarray(frame_pos, dtype=np.float64)
        self.frame_dir = np.asarray(frame_dir, dtype=np.float64)

        self.k_nn = int(min(k_nn, len(self.frame_pos)))
        self.sigma = float(sigma)
        self.jitter_pos = float(jitter_pos)
        self.alpha_label = float(alpha_label)
        self.alpha_rel = float(alpha_rel)
        self.p_u_label = float(p_u_label)
        self.p_u_rel = float(p_u_rel)
        self.p_u_unanswerable = float(p_u_unanswerable)
        self.vis_tau = float(vis_tau)
        self.ans_tau = float(ans_tau)
        self.eps = eps

        rng = np.random.default_rng(seed)
        p0 = np.asarray(cand_prior, dtype=np.float64)
        p0 = p0 / max(float(p0.sum()), eps)
        idxs = rng.choice(len(cand_pos), size=int(n_particles), replace=True, p=p0)

        self.p_pos = np.asarray(cand_pos, dtype=np.float64)[idxs] + rng.normal(scale=self.jitter_pos, size=(len(idxs), 3))

        if cand_dir is not None:
            cd = np.asarray(cand_dir, dtype=np.float64)
            self.p_dir = cd[idxs].copy()
            bad = np.linalg.norm(self.p_dir, axis=1) < 1e-6
            if bad.any():
                nn = self._nearest_frame_idx(self.p_pos[bad])
                self.p_dir[bad] = self.frame_dir[nn]
        else:
            nn = self._nearest_frame_idx(self.p_pos)
            self.p_dir = self.frame_dir[nn]

        self.w = np.ones(len(self.p_pos), dtype=np.float64) / len(self.p_pos)
        self.rng = rng
        self._label_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def top_prob(self) -> float:
        return float(self.w.max())

    def posterior_vector(self) -> np.ndarray:
        return self.w

    def _nearest_frame_idx(self, X: np.ndarray) -> np.ndarray:
        d2 = ((X[:, None, :] - self.frame_pos[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d2, axis=1)

    def _knn(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d2 = ((X[:, None, :] - self.frame_pos[None, :, :]) ** 2).sum(axis=2)
        K = min(self.k_nn, d2.shape[1])
        idx = np.argpartition(d2, kth=K - 1, axis=1)[:, :K]
        d2k = np.take_along_axis(d2, idx, axis=1)
        order = np.argsort(d2k, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        d = np.sqrt(np.take_along_axis(d2, idx, axis=1))
        return idx, d

    def _gauss_w(self, d: np.ndarray) -> np.ndarray:
        if self.sigma <= 0:
            w = np.ones_like(d)
        else:
            w = np.exp(-(d ** 2) / (2.0 * self.sigma * self.sigma))
        s = w.sum(axis=1, keepdims=True)
        return w / np.maximum(s, self.eps)

    def frame_posterior(self) -> np.ndarray:
        nn = self._nearest_frame_idx(self.p_pos)
        F = len(self.frame_pos)
        pf = np.zeros(F, dtype=np.float64)
        for i, j in enumerate(nn):
            pf[int(j)] += float(self.w[i])
        s = float(pf.sum())
        return pf / s if s > self.eps else np.ones(F) / F

    def _frame_label_arrays(self, label: str) -> Tuple[np.ndarray, np.ndarray]:
        label = str(label).strip().lower()
        if label in self._label_cache:
            return self._label_cache[label]
        p_vis = np.zeros(len(self.frame_label_dicts), dtype=np.float64)
        p_ans = np.zeros_like(p_vis)
        for j, d in enumerate(self.frame_label_dicts):
            sal = d.get(label, 0.0)
            p_vis[j] = salience_to_visprob(sal, self.vis_tau)
            p_ans[j] = salience_to_answerable(sal, self.ans_tau)
        self._label_cache[label] = (p_vis, p_ans)
        return p_vis, p_ans

    def label_prob_yes(self, label: str) -> Tuple[np.ndarray, np.ndarray]:
        # particle-level p_true and p_ans
        p_vis_f, p_ans_f = self._frame_label_arrays(label)
        idx, d = self._knn(self.p_pos)
        w = self._gauss_w(d)
        p_true = (w * np.take(p_vis_f, idx)).sum(axis=1)
        p_ans = (w * np.take(p_ans_f, idx)).sum(axis=1)
        return p_true, p_ans

    def rel_prob_true_and_answerable(self, triple: Tuple[str, str, str]) -> Tuple[np.ndarray, np.ndarray]:
        s, r, o = triple
        true_f = np.array([1.0 if triple in rels else 0.0 for rels in self.frame_rel_sets], dtype=np.float64)
        _, a_s = self._frame_label_arrays(s)
        _, a_o = self._frame_label_arrays(o)
        ans_f = np.minimum(a_s, a_o)

        idx, d = self._knn(self.p_pos)
        w = self._gauss_w(d)
        p_true = (w * np.take(true_f, idx)).sum(axis=1)
        p_ans = (w * np.take(ans_f, idx)).sum(axis=1)
        return p_true, p_ans

    def _resample(self):
        P = len(self.w)
        idxs = self.rng.choice(P, size=P, replace=True, p=self.w)
        self.p_pos = self.p_pos[idxs] + self.rng.normal(scale=self.jitter_pos, size=self.p_pos.shape)
        self.p_dir = self.p_dir[idxs]
        self.w = np.ones(P, dtype=np.float64) / P

    def update_label(self, label: str, ans: str):
        p_true, p_ans = self.label_prob_yes(label)
        Py, Pn, Pu = ynu_likelihood_from_prob(
            p_true, p_ans,
            alpha=self.alpha_label,
            p_u_base=self.p_u_label,
            p_u_unanswerable=self.p_u_unanswerable,
            eps=self.eps,
        )
        like = Py if ans == "y" else (Pn if ans == "n" else Pu)
        self.w = bayes_update(self.w, like, eps=self.eps)
        self._resample()

    def update_rel(self, triple: Tuple[str, str, str], ans: str):
        p_true, p_ans = self.rel_prob_true_and_answerable(triple)
        Py, Pn, Pu = ynu_likelihood_from_prob(
            p_true, p_ans,
            alpha=self.alpha_rel,
            p_u_base=self.p_u_rel,
            p_u_unanswerable=self.p_u_unanswerable,
            eps=self.eps,
        )
        like = Py if ans == "y" else (Pn if ans == "n" else Pu)
        self.w = bayes_update(self.w, like, eps=self.eps)
        self._resample()

    def predict_pose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        i_map = int(np.argmax(self.w))
        map_pos = self.p_pos[i_map].copy()
        map_dir = _normalize(self.p_dir[i_map])
        mean_pos = (self.w[:, None] * self.p_pos).sum(axis=0)
        mean_dir = _normalize((self.w[:, None] * self.p_dir).sum(axis=0))
        return map_pos, map_dir, mean_pos, mean_dir


# -----------------------
# Question selection
# -----------------------
def question_key(q: Question, label_pool: List[str], rel_pool: List[Any]) -> Tuple:
    if q.qtype == "label":
        return ("label", label_pool[q.idx])
    s, r, o = rel_item_to_tuple(rel_pool[q.idx])
    return ("rel", s, r, o)


def compute_ig_for_question(
    p: np.ndarray,
    p_true: np.ndarray,
    p_ans: np.ndarray,
    alpha: float,
    p_u_base: float,
    p_u_unanswerable: float,
) -> float:
    H0 = entropy(p)
    Py, Pn, Pu = ynu_likelihood_from_prob(p_true, p_ans, alpha, p_u_base, p_u_unanswerable)
    # P(answer)
    P_y = float(np.dot(p, Py))
    P_n = float(np.dot(p, Pn))
    P_u = float(np.dot(p, Pu))

    Hexp = 0.0
    if P_y > 1e-12:
        Hexp += P_y * entropy(bayes_update(p, Py))
    if P_n > 1e-12:
        Hexp += P_n * entropy(bayes_update(p, Pn))
    if P_u > 1e-12:
        Hexp += P_u * entropy(bayes_update(p, Pu))

    return H0 - Hexp


def pick_next_question_system(
    backend_name: str,
    backend,
    questions: List[Question],
    label_pool: List[str],
    rel_pool: List[Any],
    idf: Dict[str, float],
    args: argparse.Namespace,
) -> Optional[Question]:
    """
    System-level selection: backend uses its own posterior and its own model.
    Strategies:
      - ig: max expected information gain (recommended)
      - binary: closest p_yes to 0.5 (fallback)
      - least_first: most extreme p_yes (fallback)
    Also:
      - downweight common labels via IDF
      - ignore labels in args.ignore_labels
      - prefer relations if any eligible (unless all relations are very low-score)
    """
    st = args.question_strategy.lower().strip()
    p = backend.posterior_vector()

    ignore = set([x.strip().lower() for x in args.ignore_labels])

    best_rel = (None, -1e18)  # (q, score)
    best_lab = (None, -1e18)

    def passes_thresholds(p_yes: float, p_ans: float, is_rel: bool) -> bool:
        if not (args.ask_min_p <= p_yes <= args.ask_max_p):
            return False
        if is_rel and p_ans < args.rel_min_answerable:
            return False
        return True

    for q in questions:
        if q.qtype == "label":
            lab = label_pool[q.idx]
            if lab in ignore:
                continue

            p_true, p_ans = backend.label_prob_yes(lab)
            p_yes = float(np.dot(p, p_true))
            p_ans_avg = float(np.dot(p, p_ans))
            if not passes_thresholds(p_yes, 1.0, False):
                continue

            if st == "ig":
                score = compute_ig_for_question(
                    p=p,
                    p_true=p_true,
                    p_ans=p_ans,
                    alpha=args.alpha_label,
                    p_u_base=args.p_u_label,
                    p_u_unanswerable=args.p_u_unanswerable,
                )
                # IDF boost
                score *= (1.0 + args.idf_weight * float(idf.get(lab, 0.0)))
            elif st == "binary":
                score = -abs(p_yes - 0.5)
            elif st in ("least_first", "least-first", "least"):
                score = -min(p_yes, 1.0 - p_yes)
            else:
                raise ValueError(f"Unknown question_strategy: {args.question_strategy}")

            if score > best_lab[1]:
                best_lab = (q, score)

        else:
            s, r, o = rel_item_to_tuple(rel_pool[q.idx])
            p_true, p_ans = backend.rel_prob_true_and_answerable((s, r, o))
            p_yes = float(np.dot(p, p_true))
            p_ans_avg = float(np.dot(p, p_ans))
            if not passes_thresholds(p_yes, p_ans_avg, True):
                continue

            if st == "ig":
                score = compute_ig_for_question(
                    p=p,
                    p_true=p_true,
                    p_ans=p_ans,
                    alpha=args.alpha_rel,
                    p_u_base=args.p_u_rel,
                    p_u_unanswerable=args.p_u_unanswerable,
                )
                score *= (1.0 + args.rel_bonus)
            elif st == "binary":
                score = -abs(p_yes - 0.5)
            elif st in ("least_first", "least-first", "least"):
                score = -min(p_yes, 1.0 - p_yes)
            else:
                raise ValueError(f"Unknown question_strategy: {args.question_strategy}")

            if score > best_rel[1]:
                best_rel = (q, score)

    # prefer relation if it exists and not terrible
    if best_rel[0] is not None and best_rel[1] >= best_lab[1] - args.rel_prefer_margin:
        return best_rel[0]
    if best_lab[0] is not None:
        return best_lab[0]
    if best_rel[0] is not None:
        return best_rel[0]
    return None


# -----------------------
# Answering: interactive or oracle
# -----------------------
def nearest_frame_to_gt(gt_pos: np.ndarray, frames_all: Sequence[Any], scene) -> int:
    # use scene.frame_pos if available
    fp = np.asarray(scene.frame_pos, dtype=np.float64)
    d = np.linalg.norm(fp - gt_pos[None, :], axis=1)
    return int(np.argmin(d))


def oracle_answer(
    q: Question,
    label_pool: List[str],
    rel_pool: List[Any],
    gt_frame_label_dict: Dict[str, float],
    gt_frame_rel_set: set,
    args: argparse.Namespace,
) -> str:
    if q.qtype == "label":
        lab = label_pool[q.idx]
        sal = float(gt_frame_label_dict.get(lab, 0.0))
        # if barely visible, treat as unknown
        ansable = salience_to_answerable(sal, args.ans_tau)
        vis = salience_to_visprob(sal, args.vis_tau)
        if ansable < args.oracle_ansable_min:
            return "u"
        return "y" if vis >= 0.5 else "n"
    else:
        s, r, o = rel_item_to_tuple(rel_pool[q.idx])
        # if subj or obj not answerable -> u
        sal_s = float(gt_frame_label_dict.get(s, 0.0))
        sal_o = float(gt_frame_label_dict.get(o, 0.0))
        ans_s = salience_to_answerable(sal_s, args.ans_tau)
        ans_o = salience_to_answerable(sal_o, args.ans_tau)
        if min(ans_s, ans_o) < args.oracle_ansable_min:
            return "u"
        return "y" if (s, r, o) in gt_frame_rel_set else "n"


# -----------------------
# Per-backend dialogue (sequential)
# -----------------------
def run_dialogue_one_backend(
    backend_name: str,
    backend,
    questions_init: List[Question],
    label_pool: List[str],
    rel_pool: List[Any],
    idf: Dict[str, float],
    args: argparse.Namespace,
    answer_cache: Optional[Dict[Tuple, str]],
    oracle_gt_frame_label_dict: Optional[Dict[str, float]],
    oracle_gt_frame_rel_set: Optional[set],
    cand_pos: Optional[np.ndarray] = None,
    cand_dir: Optional[np.ndarray] = None,
    frames_pool: Optional[Sequence[Any]] = None,
) -> int:
    questions = list(questions_init)
    asked = 0

    print(f"\n=== Dialogue for {backend_name.upper()} ===")
    print(HELP_TEXT)

    for r in range(args.max_rounds):
        tp = backend.top_prob()
        print(f"\n[{backend_name.upper()}] Round {r+1} | topP={tp:.3f}")

        if r + 1 >= args.min_rounds and tp >= args.conf_threshold:
            print(f"✅ Confident ({tp:.3f} ≥ {args.conf_threshold:.2f})")
            break

        # pick question; if none, relax thresholds progressively
        q = pick_next_question_system(backend_name, backend, questions, label_pool, rel_pool, idf, args)
        if q is None and args.auto_relax:
            # relax #1: widen p window
            old_min, old_max, old_ans = args.ask_min_p, args.ask_max_p, args.rel_min_answerable
            args.ask_min_p, args.ask_max_p = 0.01, 0.99
            q = pick_next_question_system(backend_name, backend, questions, label_pool, rel_pool, idf, args)
            # relax #2: allow relations regardless answerable
            if q is None:
                args.rel_min_answerable = 0.0
                q = pick_next_question_system(backend_name, backend, questions, label_pool, rel_pool, idf, args)
            # restore
            args.ask_min_p, args.ask_max_p, args.rel_min_answerable = old_min, old_max, old_ans

        if q is None:
            print("No more questions that satisfy thresholds.")
            break

        # render
        if q.qtype == "label":
            lab = label_pool[q.idx]
            p_true, p_ans = backend.label_prob_yes(lab)
            p_yes = float(np.dot(backend.posterior_vector(), p_true))
            p_ans_avg = float(np.dot(backend.posterior_vector(), p_ans))
            print(f"Ask[label]: Do you see **{lab}** ?  (P(yes)≈{p_yes:.2f}, P(ans)≈{p_ans_avg:.2f})")
        else:
            s, rrel, o = rel_item_to_tuple(rel_pool[q.idx])
            p_true, p_ans = backend.rel_prob_true_and_answerable((s, rrel, o))
            p_yes = float(np.dot(backend.posterior_vector(), p_true))
            p_ans_avg = float(np.dot(backend.posterior_vector(), p_ans))
            print(f"Ask[rel ]: Is **{s}** {relation_phrase(rrel)} **{o}** ? (P(true)≈{p_yes:.2f}, P(ans)≈{p_ans_avg:.2f})")

        key = question_key(q, label_pool, rel_pool)
        if answer_cache is not None and key in answer_cache and args.cache_answers:
            ans = answer_cache[key]
            print(f"[cached answer: {ans}]")
        else:
            if args.answer_mode == "oracle":
                ans = oracle_answer(q, label_pool, rel_pool, oracle_gt_frame_label_dict or {}, oracle_gt_frame_rel_set or set(), args)
                print(f"[oracle answer: {ans}]")
            else:
                ans = input("[y/n/u/q/tf/tc/tp/o/h] > ").strip().lower()

        # commands
        if ans in ("h", "?"):
            print(HELP_TEXT)
            continue
        if ans == "tf":
            if hasattr(backend, "frame_posterior") and frames_pool is not None:
                show_top_frames(frames_pool, backend.frame_posterior(), top_n=args.show_top_n, title=f"Top frames ({backend_name.upper()})")
            else:
                print("tf not available for this backend.")
            continue
        if ans == "tc":
            if backend_name != "a1" or cand_pos is None:
                print("tc only available in A1 dialogue.")
            else:
                show_top_candidates(cand_pos, cand_dir, backend.posterior_vector(), top_n=args.show_top_n)
            continue
        if ans == "tp":
            if backend_name != "a2" or not hasattr(backend, "p_pos"):
                print("tp only available in A2 dialogue.")
            else:
                show_top_particles(backend.p_pos, backend.p_dir, backend.posterior_vector(), top_n=args.show_top_n)
            continue
        if ans == "o":
            print(f"Pool sizes: labels={len(label_pool)} rels={len(rel_pool)} questions={len(questions)}")
            continue
        if ans == "q":
            print("Quit this entry.")
            return asked
        if ans not in ("y", "n", "u"):
            print("Invalid input. Type 'h' for help.")
            continue

        if answer_cache is not None and args.cache_answers:
            answer_cache[key] = ans

        asked += 1

        # update
        if q.qtype == "label":
            backend.update_label(label_pool[q.idx], ans)
        else:
            backend.update_rel(rel_item_to_tuple(rel_pool[q.idx]), ans)

        # remove asked
        questions = [qq for qq in questions if not (qq.qtype == q.qtype and qq.idx == q.idx)]

    return asked


# -----------------------
# Entry runner (build common scene data once)
# -----------------------
def run_entry(entry: Dict[str, Any], args: argparse.Namespace) -> Optional[Dict[str, Tuple[float, float, float, float]]]:
    scene_id = entry.get("scene_id", "")
    if not scene_id:
        return None

    scene = dsf.load_scene_data(Path(args.dataset_root), scene_id, dict(dsf.DEFAULT_ALIASES))
    frames_all = scene.frames

    # gt + predicted baseline
    gt_pos, gt_dir, _ = get_pose(entry, "gt_pose")
    pred_pos, pred_dir, pred_meta = get_pose(entry, "predicted_pose")

    if gt_pos is None:
        print(f"[{scene_id}] Missing gt_pose; skipping.")
        return None

    # candidates
    cand_pos, cand_dir, cand_prior = extract_candidates(
        entry,
        candidate_set=args.candidate_set,
        include_predicted_pose=args.include_predicted_pose,
        pred_prior=args.pred_candidate_prior,
    )

    # mapping
    c2f_map = dsf.build_cand_to_frame_map(
        cand_pos=cand_pos,
        cand_dir=cand_dir,
        frame_pos=scene.frame_pos,
        frame_dir=scene.frame_dir,
        k_nn=args.k_nn,
        sigma=args.sigma,
        use_direction=args.use_direction and (cand_dir is not None),
        dir_temp=args.dir_temp,
    )
    frame_subset = dsf.top_frames_by_mapping(c2f_map, max_frames=args.max_pool_frames)
    frames_pool = [frames_all[i] for i in frame_subset]

    # dense W for pooled frames
    c2f_full = c2f_to_dense(c2f_map, num_cands=int(cand_pos.shape[0]), num_frames=int(scene.frame_pos.shape[0]))
    W = c2f_full[:, frame_subset]  # (N, F_pool)

    # pooled frame pose
    pool_pos = np.asarray([scene.frame_pos[i] for i in frame_subset], dtype=np.float64)
    pool_dir = np.asarray([_normalize(scene.frame_dir[i]) for i in frame_subset], dtype=np.float64)

    # pooled semantics
    pool_label_dicts = [frame_label_salience(fr) for fr in frames_pool]
    pool_rel_sets = [frame_relations(fr) for fr in frames_pool]

    # pools
    label_pool, rel_pool = build_pools(
        frames_all=frames_all,
        frame_subset=frame_subset,
        max_rel_pool=args.max_rel_pool,
        rel_min_salience=args.rel_min_salience,
        rel_unique_only=args.rel_unique_only,
        allowed_rels=args.allowed_rels,
    )
    # label ignore list normalize
    label_pool = [lab for lab in label_pool if lab not in set([x.strip().lower() for x in args.ignore_labels])]

    # idf
    idf = compute_label_idf(label_pool, pool_label_dicts)

    # initial A3 posterior: pf0 ∝ W^T * cand_prior
    pf0 = (W.T @ cand_prior).reshape(-1)
    pf0 = pf0 / max(float(pf0.sum()), 1e-12)

    # baseline errors
    pred_pos_err = pred_rot_err = float("nan")
    if pred_pos is not None:
        pred_pos_err, pred_rot_err = pose_errors(pred_pos, pred_dir, gt_pos, gt_dir)

    # nearest candidate baseline
    d = np.linalg.norm(cand_pos.astype(np.float64) - gt_pos[None, :], axis=1)
    idx_near = int(np.argmin(d))
    near_pos = cand_pos[idx_near]
    near_dir = None if cand_dir is None else cand_dir[idx_near]
    near_pos_err, near_rot_err = pose_errors(near_pos, near_dir, gt_pos, gt_dir)

    print(f"\n=== Scene {scene_id} ===")
    print(f"Candidates={len(cand_pos)} | Frames(pool)={len(frames_pool)} | Labels={len(label_pool)} | Relations={len(rel_pool)}")
    if args.show_gt_debug:
        print(f"GT pos: {gt_pos.tolist()}")
        if gt_dir is not None:
            print(f"GT dir: {gt_dir.tolist()}")
        if pred_pos is not None:
            src = pred_meta.get("source", "unknown")
            print(f"Predicted pose source: {src}")
            print(f"Predicted_pose vs gt_pose: pos_err={pred_pos_err:.3f} m | rot_err={pred_rot_err:.2f} deg")
        print(f"Nearest candidate idx={idx_near}: pos_err={near_pos_err:.3f} m | rot_err={near_rot_err:.2f} deg")

    # oracle support (optional)
    oracle_label_dict = None
    oracle_rel_set = None
    if args.answer_mode == "oracle":
        gt_frame_idx = nearest_frame_to_gt(gt_pos, frames_all, scene)
        fr_gt = frames_all[gt_frame_idx]
        oracle_label_dict = frame_label_salience(fr_gt)
        oracle_rel_set = frame_relations(fr_gt)
        if args.show_gt_debug:
            fid = getattr(fr_gt, "frame_id", f"idx={gt_frame_idx}")
            print(f"[oracle] using nearest GT frame: {fid}")

    # question list (same pool; each backend will use its own selection)
    questions_init = [Question("rel", i) for i in range(len(rel_pool))] + [Question("label", i) for i in range(len(label_pool))]

    # helper to create fresh backends
    def fresh_backends():
        a1 = CandidateBackendA1(
            cand_pos=cand_pos,
            cand_dir=cand_dir,
            cand_prior=cand_prior,
            c2f_pool=W,
            frame_label_dicts=pool_label_dicts,
            frame_rel_sets=pool_rel_sets,
            frame_dirs=pool_dir,
            alpha_label=args.alpha_label,
            alpha_rel=args.alpha_rel,
            p_u_label=args.p_u_label,
            p_u_rel=args.p_u_rel,
            p_u_unanswerable=args.p_u_unanswerable,
            vis_tau=args.vis_tau,
            ans_tau=args.ans_tau,
        )
        a3 = FrameBackendA3(
            p0=pf0,
            frames_pool=frames_pool,
            frame_label_dicts=pool_label_dicts,
            frame_rel_sets=pool_rel_sets,
            frame_pos=pool_pos,
            frame_dir=pool_dir,
            alpha_label=args.alpha_label,
            alpha_rel=args.alpha_rel,
            p_u_label=args.p_u_label,
            p_u_rel=args.p_u_rel,
            p_u_unanswerable=args.p_u_unanswerable,
            vis_tau=args.vis_tau,
            ans_tau=args.ans_tau,
        )
        a2 = ParticleBackendA2(
            cand_pos=cand_pos,
            cand_dir=cand_dir,
            cand_prior=cand_prior,
            frame_label_dicts=pool_label_dicts,
            frame_rel_sets=pool_rel_sets,
            frame_pos=pool_pos,
            frame_dir=pool_dir,
            n_particles=args.n_particles,
            k_nn=args.p_k_nn,
            sigma=args.p_sigma,
            jitter_pos=args.p_jitter,
            alpha_label=args.alpha_label,
            alpha_rel=args.alpha_rel,
            p_u_label=args.p_u_label,
            p_u_rel=args.p_u_rel,
            p_u_unanswerable=args.p_u_unanswerable,
            vis_tau=args.vis_tau,
            ans_tau=args.ans_tau,
            seed=args.seed,
        )
        return {"a1": a1, "a2": a2, "a3": a3}

    # sequential / shared
    out: Dict[str, Tuple[float, float, float, float]] = {}
    cache: Dict[Tuple, str] = {}

    if args.eval_mode == "shared":
        # shared evidence mode (kept for debugging)
        backends = fresh_backends()
        print("\n=== Shared dialogue mode (one Q/A updates all) ===")
        # run using a selected driver, but still updates all
        questions = list(questions_init)
        asked = 0
        for r in range(args.max_rounds):
            driver = args.question_driver
            tp = backends[driver].top_prob()
            print(f"\n[Shared] Round {r+1} | driver={driver} topP={tp:.3f}")
            if r + 1 >= args.min_rounds and tp >= args.conf_threshold:
                print("✅ Confident (driver)")
                break
            q = pick_next_question_system(driver, backends[driver], questions, label_pool, rel_pool, idf, args)
            if q is None:
                print("No more questions.")
                break

            if q.qtype == "label":
                lab = label_pool[q.idx]
                print(f"Ask[label]: {lab}")
            else:
                s, rr, o = rel_item_to_tuple(rel_pool[q.idx])
                print(f"Ask[rel ]: {s} {rr} {o}")

            ans = input("[y/n/u/q] > ").strip().lower()
            if ans == "q":
                return None
            if ans not in ("y", "n", "u"):
                print("Invalid.")
                continue

            asked += 1
            if q.qtype == "label":
                lab = label_pool[q.idx]
                backends["a1"].update_label(lab, ans)
                backends["a2"].update_label(lab, ans)
                backends["a3"].update_label(lab, ans)
            else:
                tr = rel_item_to_tuple(rel_pool[q.idx])
                backends["a1"].update_rel(tr, ans)
                backends["a2"].update_rel(tr, ans)
                backends["a3"].update_rel(tr, ans)

            questions = [qq for qq in questions if not (qq.qtype == q.qtype and qq.idx == q.idx)]

        for bname, tag in [("a1", "A1"), ("a2", "A2"), ("a3", "A3")]:
            mp, md, meanp, meand = backends[bname].predict_pose()
            pe_map, re_map = pose_errors(mp, md, gt_pos, gt_dir)
            pe_mean, re_mean = pose_errors(meanp, meand, gt_pos, gt_dir)
            out[tag] = (pe_map, re_map, pe_mean, re_mean)

    else:
        # sequential system-level evaluation
        order = [x.strip().lower() for x in args.backend_order]
        order = [x for x in order if x in ("a1", "a2", "a3")]
        if not order:
            order = ["a1", "a2", "a3"]

        print("\n=== Baseline (pipeline predicted_pose) ===")
        if pred_pos is not None:
            print(f"predicted_pose vs gt_pose: pos_err={pred_pos_err:.3f} m | rot_err={pred_rot_err:.2f} deg")

        for bname in order:
            bks = fresh_backends()
            asked = run_dialogue_one_backend(
                backend_name=bname,
                backend=bks[bname],
                questions_init=questions_init,
                label_pool=label_pool,
                rel_pool=rel_pool,
                idf=idf,
                args=args,
                answer_cache=cache,
                oracle_gt_frame_label_dict=oracle_label_dict,
                oracle_gt_frame_rel_set=oracle_rel_set,
                cand_pos=cand_pos if bname == "a1" else None,
                cand_dir=cand_dir if bname == "a1" else None,
                frames_pool=frames_pool if hasattr(bks[bname], "frame_posterior") else None,
            )
            mp, md, meanp, meand = bks[bname].predict_pose()
            pe_map, re_map = pose_errors(mp, md, gt_pos, gt_dir)
            pe_mean, re_mean = pose_errors(meanp, meand, gt_pos, gt_dir)
            tag = {"a1": "A1", "a2": "A2", "a3": "A3"}[bname]
            out[tag] = (pe_map, re_map, pe_mean, re_mean)

            print(f"\n[{tag}] Questions asked: {asked}")
            print(f"[{tag}] MAP : pos_err={pe_map:.3f} m | rot_err={re_map:.2f} deg")
            print(f"[{tag}] Mean: pos_err={pe_mean:.3f} m | rot_err={re_mean:.2f} deg")

        print("\n=== Summary (system-level: each backend ran its own dialogue) ===")
        if pred_pos is not None:
            print(f"Baseline predicted_pose: pos_err={pred_pos_err:.3f} m | rot_err={pred_rot_err:.2f} deg")
        print(f"Nearest candidate idx={idx_near}: pos_err={near_pos_err:.3f} m | rot_err={near_rot_err:.2f} deg")
        for tag in ("A1", "A2", "A3"):
            pe_map, re_map, pe_mean, re_mean = out.get(tag, (float("nan"),) * 4)
            print(f"{tag}: MAP({pe_map:.3f}m,{re_map:.2f}°) | Mean({pe_mean:.3f}m,{re_mean:.2f}°)")

    return out


# -----------------------
# Aggregation helpers
# -----------------------
def safe_mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(statistics.fmean(xs)) if xs else float("nan")


def safe_median(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(statistics.median(xs)) if xs else float("nan")


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates_json", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--only_scene_id", default="")
    ap.add_argument("--limit", type=int, default=1)

    # evaluation mode
    ap.add_argument("--eval_mode", choices=["sequential", "shared"], default="sequential")
    ap.add_argument("--backend_order", nargs="*", default=["a1", "a2", "a3"])
    ap.add_argument("--cache_answers", action="store_true", help="Reuse answers for identical questions across backends.")

    # candidates
    ap.add_argument("--candidate_set", choices=["auto", "grid", "fov", "both"], default="both")
    ap.add_argument("--include_predicted_pose", action="store_true")
    ap.add_argument("--pred_candidate_prior", type=float, default=0.35)

    # mapping
    ap.add_argument("--k_nn", type=int, default=15)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--use_direction", action="store_true")
    ap.add_argument("--dir_temp", type=float, default=0.25)

    # dialogue loop
    ap.add_argument("--max_rounds", type=int, default=12)
    ap.add_argument("--min_rounds", type=int, default=2)
    ap.add_argument("--conf_threshold", type=float, default=0.85)
    ap.add_argument("--auto_relax", action="store_true", help="If no questions pass thresholds, relax automatically.")
    ap.add_argument("--ask_min_p", type=float, default=0.01)
    ap.add_argument("--ask_max_p", type=float, default=0.99)

    # question selection
    ap.add_argument("--question_strategy", choices=["ig", "binary", "least_first"], default="ig")
    ap.add_argument("--question_driver", choices=["a1", "a2", "a3"], default="a3", help="Used only in shared mode.")
    ap.add_argument("--rel_min_answerable", type=float, default=0.10)
    ap.add_argument("--rel_bonus", type=float, default=0.25, help="IG bonus multiplier for relations.")
    ap.add_argument("--rel_prefer_margin", type=float, default=0.05, help="Prefer relation if not much worse than best label.")

    # label filtering / IDF
    ap.add_argument("--idf_weight", type=float, default=1.0)
    ap.add_argument(
        "--ignore_labels",
        nargs="*",
        default=["floor", "wall", "ceiling", "room", "baseboard", "carpet"],
        help="Labels to ignore (common/unhelpful).",
    )

    # pools
    ap.add_argument("--rel_min_salience", type=float, default=0.0)
    ap.add_argument("--rel_unique_only", action="store_true")
    ap.add_argument("--max_pool_frames", type=int, default=30)
    ap.add_argument("--max_rel_pool", type=int, default=600)
    ap.add_argument("--allowed_rels", nargs="*", default=[])

    # likelihood calibration
    ap.add_argument("--alpha_label", type=float, default=0.82, help="Lower than 0.90 to reduce overconfidence.")
    ap.add_argument("--alpha_rel", type=float, default=0.70)
    ap.add_argument("--p_u_label", type=float, default=0.05)
    ap.add_argument("--p_u_rel", type=float, default=0.15)
    ap.add_argument("--p_u_unanswerable", type=float, default=0.90)
    ap.add_argument("--vis_tau", type=float, default=0.20, help="Salience->visibility scale (0..1).")
    ap.add_argument("--ans_tau", type=float, default=0.10, help="Salience->answerable scale (0..1).")

    # A2 particles
    ap.add_argument("--n_particles", type=int, default=256)
    ap.add_argument("--p_k_nn", type=int, default=10)
    ap.add_argument("--p_sigma", type=float, default=0.25)
    ap.add_argument("--p_jitter", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=0)

    # answering mode (mock evaluation)
    ap.add_argument("--answer_mode", choices=["interactive", "oracle"], default="interactive")
    ap.add_argument("--oracle_ansable_min", type=float, default=0.25)

    # UI
    ap.add_argument("--show_top_n", type=int, default=5)
    ap.add_argument("--show_gt_debug", action="store_true")

    args = ap.parse_args()

    # load relaxed JSON
    try:
        data = dsf.load_relaxed_json(Path(args.candidates_json))
    except Exception:
        txt = Path(args.candidates_json).read_text(encoding="utf-8").replace("\r\n", "\n")
        txt = re.sub(r",\s*(\}|\])", r"\1", txt)
        data = __import__("json").loads(txt)

    entries = data.get("scenes", data.get("entries", []))
    if not isinstance(entries, list):
        raise ValueError("Expected list under 'scenes' (or 'entries').")

    if args.only_scene_id:
        entries = [e for e in entries if e.get("scene_id", "") == args.only_scene_id]
    if args.limit and args.limit > 0:
        entries = entries[: args.limit]

    if not entries:
        print("No entries selected.")
        return

    # aggregate (MAP only)
    agg = {name: {"pos": [], "rot": []} for name in ("PRED", "A1", "A2", "A3")}

    for e in entries:
        # baseline predicted_pose errors
        gt_pos, gt_dir, _ = get_pose(e, "gt_pose")
        pred_pos, pred_dir, _ = get_pose(e, "predicted_pose")
        if gt_pos is not None and pred_pos is not None:
            pe, re_ = pose_errors(pred_pos, pred_dir, gt_pos, gt_dir)
            agg["PRED"]["pos"].append(pe)
            agg["PRED"]["rot"].append(re_)

        res = run_entry(e, args)
        if res is None:
            break
        for name in ("A1", "A2", "A3"):
            pe_map, re_map, _, _ = res[name]
            agg[name]["pos"].append(pe_map)
            agg[name]["rot"].append(re_map)

    if len(entries) > 1:
        print("\n=== Aggregate over entries (MAP errors) ===")
        for name in ("PRED", "A1", "A2", "A3"):
            mp = safe_mean(agg[name]["pos"])
            mdp = safe_median(agg[name]["pos"])
            mr = safe_mean(agg[name]["rot"])
            mdr = safe_median(agg[name]["rot"])
            print(f"{name}: mean_pos={mp:.3f} m | med_pos={mdp:.3f} m | mean_rot={mr:.2f}° | med_rot={mdr:.2f}°")


if __name__ == "__main__":
    main()
