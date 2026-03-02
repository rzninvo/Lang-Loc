#!/usr/bin/env python3
"""Pose-level dialogue system evaluation entry point.

System-level interactive evaluation: each backend runs its own dialogue
(its own question policy), then results are compared side-by-side against
the ground-truth pose.

Backends
--------
A1) Candidate posterior (discrete) — cand→frame mapping *W* + frame semantics.
A2) Particle filter (continuous) — KNN-to-frames proxy semantics.
A3) Frame posterior (discrete over frames) — exact frame semantics.

Requires
--------
- ``dataset_root/<scene_id>/output/descriptions/all_descriptions*.json``
  readable by ``scene_data.load_scene_data``.
- ``candidates_json`` with ``gt_pose``, ``predicted_pose``, and candidate
  lists.

Example
-------
::

    python -m src.dialogue.eval_runner \\
        --candidates_json data/candidates.json \\
        --dataset_root /path/to/3RScan \\
        --only_scene_id "scene-id" \\
        --limit 1 \\
        --answer_mode oracle
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from langloc.dialogue.backends import (
    CandidateBackendA1,
    FrameBackendA3,
    ParticleBackendA2,
)
from langloc.dialogue.candidates import extract_candidates
from langloc.dialogue.dialogue_config import DialogueConfig, extract_dialogue_config
from langloc.dialogue.frame_mapping import (
    build_cand_to_frame_map,
    top_frames_by_mapping,
)
from langloc.dialogue.dialogue_runner import (
    nearest_frame_to_gt,
    run_dialogue_one_backend,
)
from langloc.dialogue.math_utils import (
    _normalize,
    c2f_to_dense,
    get_pose,
    pose_errors,
    safe_mean,
    safe_median,
)
from langloc.dialogue.question_pool import Question, build_pools, compute_label_idf
from langloc.dialogue.question_selection import pick_next_question_system, show_top_frames
from langloc.dialogue.scene_data import (
    DEFAULT_ALIASES,
    load_relaxed_json,
    load_scene_data,
)
from langloc.dialogue.semantics import (
    frame_label_salience,
    frame_relations,
    rel_item_to_tuple,
)


# ---------------------------------------------------------------------------
# Entry runner
# ---------------------------------------------------------------------------
def run_entry(
    entry: Dict[str, Any],
    cfg: Union[DialogueConfig, argparse.Namespace],
) -> Optional[Dict[str, Tuple[float, float, float, float]]]:
    """Run the dialogue evaluation for a single scene entry.

    Loads scene data, extracts candidates and frame semantics, creates
    fresh backends, and runs sequential or shared dialogues.

    Args:
        entry: Single scene entry from the candidates JSON.
        cfg: Dialogue configuration (``DialogueConfig`` or argparse namespace).

    Returns:
        Dictionary mapping backend tags (``"A1"``, ``"A2"``, ``"A3"``) to
        ``(map_pos_err, map_rot_err, mean_pos_err, mean_rot_err)`` tuples,
        or ``None`` if the entry is skipped.
    """
    scene_id = entry.get("scene_id", "")
    if not scene_id:
        return None

    scene = load_scene_data(Path(cfg.dataset_root), scene_id, dict(DEFAULT_ALIASES))
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
        candidate_set=cfg.candidate_set,
        include_predicted_pose=cfg.include_predicted_pose,
        pred_prior=cfg.pred_candidate_prior,
    )

    # mapping
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
        max_rel_pool=cfg.max_rel_pool,
        rel_min_salience=cfg.rel_min_salience,
        rel_unique_only=cfg.rel_unique_only,
        allowed_rels=cfg.allowed_rels,
    )
    # label ignore list normalize
    label_pool = [lab for lab in label_pool if lab not in set([x.strip().lower() for x in cfg.ignore_labels])]

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
    if cfg.show_gt_debug:
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
    if cfg.answer_mode == "oracle":
        gt_frame_idx = nearest_frame_to_gt(gt_pos, frames_all, scene)
        fr_gt = frames_all[gt_frame_idx]
        oracle_label_dict = frame_label_salience(fr_gt)
        oracle_rel_set = frame_relations(fr_gt)
        if cfg.show_gt_debug:
            fid = getattr(fr_gt, "frame_id", f"idx={gt_frame_idx}")
            print(f"[oracle] using nearest GT frame: {fid}")

    # question list (same pool; each backend will use its own selection)
    questions_init = [Question("rel", i) for i in range(len(rel_pool))] + [Question("label", i) for i in range(len(label_pool))]

    # helper to create fresh backends
    def fresh_backends() -> Dict[str, Any]:
        a1 = CandidateBackendA1(
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
        a3 = FrameBackendA3(
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
        a2 = ParticleBackendA2(
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
        return {"a1": a1, "a2": a2, "a3": a3}

    # sequential / shared
    out: Dict[str, Tuple[float, float, float, float]] = {}
    cache: Dict[Tuple, str] = {}

    if cfg.eval_mode == "shared":
        # shared evidence mode (kept for debugging)
        backends = fresh_backends()
        print("\n=== Shared dialogue mode (one Q/A updates all) ===")
        # run using a selected driver, but still updates all
        questions = list(questions_init)
        asked = 0
        for r in range(cfg.max_rounds):
            driver = cfg.question_driver
            tp = backends[driver].top_prob()
            print(f"\n[Shared] Round {r+1} | driver={driver} topP={tp:.3f}")
            if r + 1 >= cfg.min_rounds and tp >= cfg.conf_threshold:
                print("Confident (driver)")
                break
            q = pick_next_question_system(driver, backends[driver], questions, label_pool, rel_pool, idf, cfg)
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
        order = [x.strip().lower() for x in cfg.backend_order]
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
                cfg=cfg,
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
            print(f"{tag}: MAP({pe_map:.3f}m,{re_map:.2f}) | Mean({pe_mean:.3f}m,{re_mean:.2f})")

    return out


# ---------------------------------------------------------------------------
# Batch runner (used by both Hydra CLI and argparse main)
# ---------------------------------------------------------------------------
def run_batch(cfg: DialogueConfig) -> None:
    """Run batch dialogue evaluation from a populated config.

    Loads the candidates JSON, filters entries, runs ``run_entry`` for
    each, and prints aggregate MAP errors.

    Args:
        cfg: Populated dialogue configuration.
    """
    # load relaxed JSON
    try:
        data = load_relaxed_json(Path(cfg.candidates_json))
    except Exception:
        txt = Path(cfg.candidates_json).read_text(encoding="utf-8").replace("\r\n", "\n")
        txt = re.sub(r",\s*(\}|\])", r"\1", txt)
        data = json.loads(txt)

    entries = data.get("scenes", data.get("entries", []))
    if not isinstance(entries, list):
        raise ValueError("Expected list under 'scenes' (or 'entries').")

    if cfg.only_scene_id:
        entries = [e for e in entries if e.get("scene_id", "") == cfg.only_scene_id]
    if cfg.limit and cfg.limit > 0:
        entries = entries[: cfg.limit]

    if not entries:
        print("No entries selected.")
        return

    # aggregate (MAP only)
    agg: Dict[str, Dict[str, List[float]]] = {name: {"pos": [], "rot": []} for name in ("PRED", "A1", "A2", "A3")}

    for e in entries:
        # baseline predicted_pose errors
        gt_pos, gt_dir, _ = get_pose(e, "gt_pose")
        pred_pos, pred_dir, _ = get_pose(e, "predicted_pose")
        if gt_pos is not None and pred_pos is not None:
            pe, re_ = pose_errors(pred_pos, pred_dir, gt_pos, gt_dir)
            agg["PRED"]["pos"].append(pe)
            agg["PRED"]["rot"].append(re_)

        res = run_entry(e, cfg)
        if res is None:
            break
        for name in ("A1", "A2", "A3"):
            if name not in res:
                continue
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
            print(f"{name}: mean_pos={mp:.3f} m | med_pos={mdp:.3f} m | mean_rot={mr:.2f} | med_rot={mdr:.2f}")


# ---------------------------------------------------------------------------
# Standalone argparse entry point (fallback)
# ---------------------------------------------------------------------------
def main() -> None:
    """Parse arguments and run batch evaluation (standalone CLI fallback)."""
    ap = argparse.ArgumentParser(
        description="Pose-level dialogue system evaluation.",
    )
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
    cfg = extract_dialogue_config(args)
    run_batch(cfg)


if __name__ == "__main__":
    main()
