#!/usr/bin/env python3
"""Generate supplementary dialogue figures as individual images for LaTeX.

Outputs into ``{output}/dialogue/{scan_id}/``::

    topdown.png
    question_log.tex
    trace/
        a1/round0.png  round1.png  ...
        a2/round0.png  ...
        a3/round0.png  ...

Markers on every heatmap:
  - Green circle  = initial predicted pose (before dialogue)
  - Red circle    = ground-truth pose
  - Blue diamond  = dialogue-refined pose (final round only)

Usage::

    python -m tools.viz.fig_supp_dialogue \
        --dataset 3rscan --root ./data/3RScan \
        --scan-id <SCAN_ID> --frame-id <FRAME_ID> \
        --graphs-3dssg ./data/3DSSG/graphs.pt \
        --output docs/figures
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Helpers (small, self-contained — avoids importing the 3k-line teaser)
# ---------------------------------------------------------------------------

def _set_eccv_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman",
                        "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
    })


def _project_to_topdown(pts_3d, intrinsic, extrinsic):
    pts_h = np.hstack([pts_3d, np.ones((len(pts_3d), 1))]).T
    pts_cam = extrinsic @ pts_h
    pts_2d = intrinsic @ pts_cam[:3, :]
    px = pts_2d[0] / (pts_2d[2] + 1e-12)
    py = pts_2d[1] / (pts_2d[2] + 1e-12)
    return px, py


# ---------------------------------------------------------------------------
# Shared dialogue state
# ---------------------------------------------------------------------------

@dataclass
class DialogueSetup:
    """All shared data needed by the three dialogue backends."""
    cand_pos: np.ndarray
    cand_dir: Optional[np.ndarray]
    cand_prior: np.ndarray
    W: np.ndarray                      # (N_cand, F_pool) mapping matrix
    frames_pool: list
    pool_label_dicts: list
    pool_rel_sets: list
    pool_pos: np.ndarray
    pool_dir: np.ndarray
    label_pool: list
    rel_pool: list
    idf: dict
    oracle_label_dict: dict
    oracle_rel_set: set
    valid_mask: np.ndarray             # bool mask into original grid
    cfg: object
    n_grid: int                        # count of valid grid cells
    scene: object                      # loaded scene data


def build_dialogue_setup(
    cams, probs, cam_dirs, pred_pos, pred_dir,
    gt_pos, gt_dir, dataset_root, scan_id, max_rounds=12,
):
    """One-time shared setup for all backends (extracted from teaser)."""
    from langloc.dialogue.dialogue_config import DialogueConfig
    from langloc.dialogue.dialogue_runner import nearest_frame_to_gt
    from langloc.dialogue.frame_mapping import build_cand_to_frame_map, top_frames_by_mapping
    from langloc.dialogue.math_utils import _normalize, c2f_to_dense
    from langloc.dialogue.question_pool import build_pools, compute_label_idf
    from langloc.dialogue.scene_data import DEFAULT_ALIASES, load_scene_data
    from langloc.dialogue.semantics import frame_label_salience, frame_relations

    cfg = DialogueConfig(
        answer_mode="oracle", max_rounds=max_rounds, min_rounds=2,
        conf_threshold=0.85, auto_relax=True,
        candidate_set="grid", include_predicted_pose=True,
        pred_candidate_prior=0.15, k_nn=5, sigma=0.5,
        use_direction=True, dir_temp=0.25, max_pool_frames=30,
        question_strategy="ig", cache_answers=False,
        ask_min_p=0.0, ask_max_p=1.0,
        allowed_rels=["left", "right", "front", "behind", "close_by"],
    )

    scene = load_scene_data(dataset_root, scan_id, dict(DEFAULT_ALIASES))
    frames_all = scene.frames

    # --- Build candidate set (shared by A1/A2, A3 projects via W) ---
    valid_mask = probs > 0
    n_grid = int(valid_mask.sum())
    cand_pos = cams[valid_mask].astype(np.float32)
    cand_prior = probs[valid_mask].astype(np.float64)
    cand_prior /= max(float(cand_prior.sum()), 1e-12)

    cd = cam_dirs[valid_mask].astype(np.float32)
    norms = np.linalg.norm(cd, axis=1, keepdims=True)
    has_dir = norms.squeeze() > 1e-6
    if has_dir.any():
        cd[has_dir] /= norms[has_dir]
    cand_dir = cd

    # Inject frame positions as extra candidates
    n_fr = len(scene.frame_pos)
    fr_pos = scene.frame_pos.astype(np.float32)
    cand_pos = np.concatenate([cand_pos, fr_pos])
    fr_prior = np.full(n_fr, 0.02, dtype=np.float64)
    total_fr = 0.02 * n_fr
    cand_prior = np.concatenate([cand_prior * (1.0 - total_fr), fr_prior])
    cand_prior = np.clip(cand_prior, 0, None)
    cand_prior /= max(float(cand_prior.sum()), 1e-12)

    fr_dir = scene.frame_dir.astype(np.float32)
    fr_dir /= np.maximum(np.linalg.norm(fr_dir, axis=1, keepdims=True), 1e-6)
    cand_dir = np.concatenate([cand_dir, fr_dir])

    # Inject predicted pose
    if pred_pos is not None:
        cand_pos = np.concatenate([cand_pos, pred_pos[None, :].astype(np.float32)])
        extra = np.array([cfg.pred_candidate_prior], dtype=np.float64)
        cand_prior = np.concatenate([cand_prior * (1.0 - cfg.pred_candidate_prior), extra])
        cand_prior /= max(float(cand_prior.sum()), 1e-12)
        if pred_dir is not None:
            pd = (pred_dir / max(float(np.linalg.norm(pred_dir)), 1e-6))[None, :].astype(np.float32)
            cand_dir = np.concatenate([cand_dir, pd])

    # Frame mapping
    c2f_map = build_cand_to_frame_map(
        cand_pos=cand_pos, cand_dir=cand_dir,
        frame_pos=scene.frame_pos, frame_dir=scene.frame_dir,
        k_nn=cfg.k_nn, sigma=cfg.sigma,
        use_direction=cfg.use_direction, dir_temp=cfg.dir_temp,
    )
    frame_subset = top_frames_by_mapping(c2f_map, max_frames=cfg.max_pool_frames)
    frames_pool = [frames_all[i] for i in frame_subset]

    W = c2f_to_dense(c2f_map, num_cands=len(cand_pos), num_frames=len(scene.frame_pos))
    W = W[:, frame_subset]

    pool_pos = np.asarray([scene.frame_pos[i] for i in frame_subset], dtype=np.float64)
    pool_dir = np.asarray([_normalize(scene.frame_dir[i]) for i in frame_subset], dtype=np.float64)

    pool_label_dicts = [frame_label_salience(fr) for fr in frames_pool]
    pool_rel_sets = [frame_relations(fr) for fr in frames_pool]

    label_pool, rel_pool = build_pools(
        frames_all=frames_all, frame_subset=frame_subset,
        max_rel_pool=cfg.max_rel_pool, rel_min_salience=cfg.rel_min_salience,
        rel_unique_only=cfg.rel_unique_only, allowed_rels=cfg.allowed_rels,
    )
    ignore = set(x.strip().lower() for x in cfg.ignore_labels)
    label_pool = [lab for lab in label_pool if lab not in ignore]
    idf = compute_label_idf(label_pool, pool_label_dicts)

    gt_frame_idx = nearest_frame_to_gt(gt_pos, frames_all, scene)
    fr_gt = frames_all[gt_frame_idx]
    oracle_label_dict = frame_label_salience(fr_gt)
    oracle_rel_set = frame_relations(fr_gt)

    return DialogueSetup(
        cand_pos=cand_pos, cand_dir=cand_dir, cand_prior=cand_prior,
        W=W, frames_pool=frames_pool,
        pool_label_dicts=pool_label_dicts, pool_rel_sets=pool_rel_sets,
        pool_pos=pool_pos, pool_dir=pool_dir,
        label_pool=label_pool, rel_pool=rel_pool, idf=idf,
        oracle_label_dict=oracle_label_dict, oracle_rel_set=oracle_rel_set,
        valid_mask=valid_mask, cfg=cfg, n_grid=n_grid, scene=scene,
    )


def _create_backend(setup: DialogueSetup, name: str):
    from langloc.dialogue.backends import CandidateBackendA1, FrameBackendA3, ParticleBackendA2
    c = setup.cfg
    kwargs = dict(
        alpha_label=c.alpha_label, alpha_rel=c.alpha_rel,
        p_u_label=c.p_u_label, p_u_rel=c.p_u_rel,
        p_u_unanswerable=c.p_u_unanswerable,
        vis_tau=c.vis_tau, ans_tau=c.ans_tau,
    )
    if name == "a1":
        return CandidateBackendA1(
            cand_pos=setup.cand_pos, cand_dir=setup.cand_dir,
            cand_prior=setup.cand_prior.copy(), c2f_pool=setup.W,
            frame_label_dicts=setup.pool_label_dicts,
            frame_rel_sets=setup.pool_rel_sets, frame_dirs=setup.pool_dir,
            **kwargs,
        )
    elif name == "a2":
        return ParticleBackendA2(
            cand_pos=setup.cand_pos, cand_dir=setup.cand_dir,
            cand_prior=setup.cand_prior.copy(),
            frame_label_dicts=setup.pool_label_dicts,
            frame_rel_sets=setup.pool_rel_sets,
            frame_pos=setup.pool_pos, frame_dir=setup.pool_dir,
            n_particles=c.n_particles, k_nn=c.p_k_nn,
            sigma=c.p_sigma, jitter_pos=c.p_jitter, seed=c.seed,
            **kwargs,
        )
    else:  # a3
        pf0 = (setup.W.T @ setup.cand_prior).reshape(-1)
        pf0 /= max(float(pf0.sum()), 1e-12)
        return FrameBackendA3(
            p0=pf0, frames_pool=setup.frames_pool,
            frame_label_dicts=setup.pool_label_dicts,
            frame_rel_sets=setup.pool_rel_sets,
            frame_pos=setup.pool_pos, frame_dir=setup.pool_dir,
            **kwargs,
        )


def _backend_to_grid(setup: DialogueSetup, backend, name: str, orig_len: int):
    """Map a backend's posterior to the original grid for heatmap rendering."""
    posterior = backend.posterior_vector().copy()
    grid_post = np.zeros(orig_len, dtype=np.float64)

    if name == "a1":
        # First n_grid entries map to cams[valid_mask]
        n = min(setup.n_grid, len(posterior))
        grid_post[setup.valid_mask] = posterior[:n]
    elif name == "a2":
        # A2: posterior is over particles → scatter to grid via Gaussian kernel
        for pi, pw in enumerate(posterior):
            if pw < 1e-12:
                continue
            pp = backend.p_pos[pi]
            dists = np.linalg.norm(
                setup.cand_pos[:setup.n_grid] - pp[None, :], axis=1)
            kernel = np.exp(-dists ** 2 / (2 * 0.5 ** 2))
            grid_post[setup.valid_mask] += pw * kernel
    else:
        # A3: posterior is over frames → scatter to grid via Gaussian kernel
        frame_weights = posterior
        for fi, fw in enumerate(frame_weights):
            if fw < 1e-12:
                continue
            fp = setup.pool_pos[fi]
            dists = np.linalg.norm(
                setup.cand_pos[:setup.n_grid] - fp[None, :], axis=1)
            kernel = np.exp(-dists ** 2 / (2 * 0.5 ** 2))
            grid_post[setup.valid_mask] += fw * kernel

    total = grid_post.sum()
    if total > 1e-12:
        grid_post /= total
    return grid_post


# ---------------------------------------------------------------------------
# Run dialogue with all 3 backends, capturing per-round snapshots
# ---------------------------------------------------------------------------

@dataclass
class RoundSnapshot:
    question: str
    answer: str
    top_probs: Dict[str, float]              # {backend_name: top_prob}
    grid_posteriors: Dict[str, np.ndarray]    # {backend_name: grid posterior}


def run_dialogue_all_backends(
    setup: DialogueSetup,
    cams: np.ndarray,
    gt_pos: np.ndarray,
    gt_dir: np.ndarray,
    max_rounds: int = 12,
) -> Tuple[List[RoundSnapshot], Dict[str, Tuple]]:
    """Run oracle dialogue with A3 picking questions; all backends updated.

    Returns:
        (snapshots, results) where results maps backend name to
        (refined_pos, refined_dir).
    """
    from langloc.dialogue.dialogue_runner import oracle_answer
    from langloc.dialogue.question_pool import Question
    from langloc.dialogue.question_selection import pick_next_question_system
    from langloc.dialogue.semantics import rel_item_to_tuple, relation_phrase

    backends = {n: _create_backend(setup, n) for n in ("a1", "a2", "a3")}
    orig_len = len(cams)

    # Build question list (A3 uses both label + relation questions)
    questions = ([Question("rel", i) for i in range(len(setup.rel_pool))]
                 + [Question("label", i) for i in range(len(setup.label_pool))])

    # Initial snapshot (before any questions)
    snapshots: List[RoundSnapshot] = []
    init_snap = RoundSnapshot(
        question="(initial)", answer="—",
        top_probs={n: float(b.top_prob()) for n, b in backends.items()},
        grid_posteriors={n: _backend_to_grid(setup, b, n, orig_len)
                         for n, b in backends.items()},
    )
    snapshots.append(init_snap)

    for r in range(max_rounds):
        # Check A3 confidence
        tp = backends["a3"].top_prob()
        if r + 1 >= setup.cfg.min_rounds and tp >= setup.cfg.conf_threshold:
            break

        # A3 picks the question
        q = pick_next_question_system(
            "a3", backends["a3"], questions,
            setup.label_pool, setup.rel_pool, setup.idf, setup.cfg)

        if q is None and setup.cfg.auto_relax:
            old_min, old_max = setup.cfg.ask_min_p, setup.cfg.ask_max_p
            old_ans = setup.cfg.rel_min_answerable
            try:
                setup.cfg.ask_min_p, setup.cfg.ask_max_p = 0.01, 0.99
                q = pick_next_question_system(
                    "a3", backends["a3"], questions,
                    setup.label_pool, setup.rel_pool, setup.idf, setup.cfg)
                if q is None:
                    setup.cfg.rel_min_answerable = 0.0
                    q = pick_next_question_system(
                        "a3", backends["a3"], questions,
                        setup.label_pool, setup.rel_pool, setup.idf, setup.cfg)
            finally:
                setup.cfg.ask_min_p, setup.cfg.ask_max_p = old_min, old_max
                setup.cfg.rel_min_answerable = old_ans

        if q is None:
            break

        # Format question text
        if q.qtype == "label":
            q_text = f"Do you see a {setup.label_pool[q.idx]}?"
        else:
            s, rel, o = rel_item_to_tuple(setup.rel_pool[q.idx])
            q_text = f"Is {s} {relation_phrase(rel)} {o}?"

        # Oracle answer
        ans = oracle_answer(
            q, setup.label_pool, setup.rel_pool,
            setup.oracle_label_dict, setup.oracle_rel_set, setup.cfg)
        print(f"  Round {r+1}: {q_text} -> {ans}")

        if ans not in ("y", "n", "u"):
            continue

        # Update ALL backends with the same question + answer
        for name, backend in backends.items():
            if q.qtype == "label":
                backend.update_label(setup.label_pool[q.idx], ans)
            else:
                backend.update_rel(rel_item_to_tuple(setup.rel_pool[q.idx]), ans)

        questions = [qq for qq in questions
                     if not (qq.qtype == q.qtype and qq.idx == q.idx)]

        snapshots.append(RoundSnapshot(
            question=q_text, answer=ans,
            top_probs={n: float(b.top_prob()) for n, b in backends.items()},
            grid_posteriors={n: _backend_to_grid(setup, b, n, orig_len)
                             for n, b in backends.items()},
        ))

    # Final poses
    results = {}
    for name, backend in backends.items():
        map_pos, map_dir, mean_pos, mean_dir = backend.predict_pose()
        if name == "a3":
            results[name] = (map_pos, map_dir)
        else:
            results[name] = (mean_pos, mean_dir)

    return snapshots, results


# ---------------------------------------------------------------------------
# Heatmap rendering (subplot-friendly, no figure creation)
# ---------------------------------------------------------------------------

def _render_heatmap_on_ax(
    ax, topdown_img, intrinsic, extrinsic,
    cams, probs, anchor_probs=None,
    gt_pos=None, gt_dir=None,
    pred_pos=None, pred_dir=None,
    refined_pos=None, refined_dir=None,
    alpha=0.65, cmap_name="inferno", up_axis="z_up",
    h_fov_deg=100.0, show_colorbar=False, title="",
):
    """Render a probability heatmap into an existing axes object."""
    H, W = topdown_img.shape[:2]
    px, py = _project_to_topdown(cams, intrinsic, extrinsic)

    coverage = anchor_probs if anchor_probs is not None else probs
    mask = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (coverage > 0)
    px_v, py_v = px[mask].astype(np.float64), py[mask].astype(np.float64)
    probs_v = probs[mask].astype(np.float64)

    if len(probs_v) < 3:
        ax.imshow(topdown_img)
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=8)
        return

    # Interpolate
    grid_res = min(256, H, W)
    gx = np.linspace(0, W - 1, grid_res)
    gy = np.linspace(0, H - 1, grid_res)
    gx_m, gy_m = np.meshgrid(gx, gy)
    points = np.column_stack([px_v, py_v])

    prob_field = griddata(points, probs_v, (gx_m, gy_m),
                          method="cubic", fill_value=0.0)
    prob_field = np.clip(prob_field, 0, None)
    prob_field = gaussian_filter(prob_field, sigma=max(grid_res * 0.01, 1.5))

    p_max = prob_field.max()
    if p_max > 1e-12:
        prob_field /= p_max

    prob_pil = Image.fromarray((prob_field * 255).astype(np.uint8), mode="L")
    prob_full = np.asarray(prob_pil.resize((W, H), Image.LANCZOS), dtype=np.float64) / 255.0

    cmap = plt.get_cmap(cmap_name)
    heatmap_rgb = cmap(prob_full)[:, :, :3]

    # Anchor-based coverage mask
    if anchor_probs is not None:
        anc_v = anchor_probs[mask].astype(np.float64)
        anc_f = griddata(points, anc_v, (gx_m, gy_m), method="cubic", fill_value=0.0)
        anc_f = np.clip(anc_f, 0, None)
        anc_full = np.asarray(
            Image.fromarray((np.clip(anc_f / max(anc_f.max(), 1e-12), 0, 1) * 255
                             ).astype(np.uint8), mode="L"
                            ).resize((W, H), Image.LANCZOS), dtype=np.float64) / 255.0
        prob_mask = anc_full > 0.02
    else:
        prob_mask = prob_full > 0.02

    mask_float = gaussian_filter(prob_mask.astype(np.float64), sigma=3.0)
    mask_float = np.clip(mask_float * alpha, 0, 1)

    base = topdown_img.astype(np.float64) / 255.0
    blend = mask_float[..., None]
    composite = (1.0 - blend) * base + blend * heatmap_rgb
    composite = (np.clip(composite, 0, 1) * 255).astype(np.uint8)

    ax.imshow(composite)

    up_idx = {"x_up": 0, "y_up": 1, "z_up": 2}.get(up_axis, 2)
    floor_axes = [i for i in range(3) if i != up_idx]
    a0, a1 = floor_axes

    def _draw_marker(pos, direction, color, edge, marker, ms, zorder):
        if pos is None:
            return
        mpx, mpy = _project_to_topdown(pos[None, :], intrinsic, extrinsic)
        mpx, mpy = float(mpx[0]), float(mpy[0])
        if not (0 <= mpx < W and 0 <= mpy < H):
            return
        # FOV wedge
        if direction is not None and h_fov_deg > 0:
            from matplotlib.patches import Polygon as MplPoly
            reach = 2.5
            half = math.radians(h_fov_deg / 2.0)
            heading = math.atan2(direction[a1], direction[a0])
            angles = np.linspace(heading - half, heading + half, 30)
            arc = np.zeros((30, 3), dtype=np.float64)
            arc[:, up_idx] = pos[up_idx]
            arc[:, a0] = pos[a0] + reach * np.cos(angles)
            arc[:, a1] = pos[a1] + reach * np.sin(angles)
            apx, apy = _project_to_topdown(arc, intrinsic, extrinsic)
            xy = [(mpx, mpy)] + [(float(x), float(y)) for x, y in zip(apx, apy)] + [(mpx, mpy)]
            ax.add_patch(MplPoly(xy, closed=True, facecolor=color, edgecolor=edge,
                                  alpha=0.30, linewidth=0.8, zorder=zorder))
        ax.plot(mpx, mpy, marker=marker, color=color, markersize=ms,
                markeredgecolor="white", markeredgewidth=0.8, zorder=zorder + 1)

    _draw_marker(pred_pos, pred_dir, "#00e676", "#00c853", "o", 7, 4)
    _draw_marker(gt_pos, gt_dir, "#ef5350", "#c62828", "o", 7, 6)
    _draw_marker(refined_pos, refined_dir, "#42a5f5", "#1565c0", "D", 7, 8)

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=8, pad=3)


# ---------------------------------------------------------------------------
# LaTeX question log
# ---------------------------------------------------------------------------

def _save_question_log_tex(snapshots, output_path):
    """Write a standalone LaTeX tabular with the dialogue transcript."""
    lines = [
        r"\begin{tabular}{clcc}",
        r"\toprule",
        r"Round & Question & Answer & $\max p$ (A3) \\",
        r"\midrule",
    ]
    ans_map = {"y": "Yes", "n": "No", "u": "Unk", "\u2014": "\u2014"}
    for i, snap in enumerate(snapshots):
        q = snap.question.replace("_", r"\_").replace("&", r"\&")
        if len(q) > 55:
            q = q[:52] + "..."
        a = ans_map.get(snap.answer, snap.answer)
        tp = f"{snap.top_probs.get('a3', 0):.2f}"
        label = "Init" if i == 0 else str(i)
        lines.append(f"  {label} & {q} & {a} & {tp} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Individual heatmap image rendering
# ---------------------------------------------------------------------------

def _save_single_heatmap(
    topdown_img, intrinsic, extrinsic, cams, grid_post,
    anchor_probs, gt_pos, gt_dir, pred_pos, pred_dir,
    refined_pos, refined_dir, up_axis, output_path,
):
    """Render one heatmap to a standalone image file (no title/axes chrome)."""
    _set_eccv_rc()
    H, W = topdown_img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, 1, figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    _render_heatmap_on_ax(
        ax, topdown_img, intrinsic, extrinsic,
        cams, grid_post, anchor_probs=anchor_probs,
        gt_pos=gt_pos, gt_dir=gt_dir,
        pred_pos=pred_pos, pred_dir=pred_dir,
        refined_pos=refined_pos, refined_dir=refined_dir,
        up_axis=up_axis,
    )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)


def render_dialogue_trace(
    snapshots, topdown_img, intrinsic, extrinsic, cams,
    gt_pos, gt_dir, pred_pos, pred_dir, results, anchor_probs,
    up_axis, out_dir,
):
    """Save individual heatmap images for all three backends.

    Directory layout under *out_dir*::

        topdown.png
        question_log.tex
        trace/
            a1/round0.png  round1.png  ...
            a2/round0.png  ...
            a3/round0.png  ...
    """
    _set_eccv_rc()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save bare top-down
    Image.fromarray(topdown_img).save(out_dir / "topdown.png", dpi=(300, 300))
    print(f"  Saved: {out_dir / 'topdown.png'}")

    # Save per-backend per-round heatmaps
    backend_names = ["a1", "a2", "a3"]
    n = len(snapshots)

    for bname in backend_names:
        bdir = out_dir / "trace" / bname
        bdir.mkdir(parents=True, exist_ok=True)
        ref_pos, ref_dir = results.get(bname, (None, None))
        for si, snap in enumerate(snapshots):
            grid_post = snap.grid_posteriors.get(bname, np.zeros(len(cams)))
            # Show refined (blue) only on the final snapshot
            rp = ref_pos if si == n - 1 else None
            rd = ref_dir if si == n - 1 else None
            fpath = bdir / f"round{si}.png"
            _save_single_heatmap(
                topdown_img, intrinsic, extrinsic, cams, grid_post,
                anchor_probs, gt_pos, gt_dir, pred_pos, pred_dir,
                rp, rd, up_axis, fpath,
            )
        print(f"  Saved: {bdir}/round{{0..{n-1}}}.png  ({n} images)")

    # Save LaTeX question log
    _save_question_log_tex(snapshots, out_dir / "question_log.tex")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Supplementary dialogue figures.")
    ap.add_argument("--dataset", choices=["3rscan", "scannet"], required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--scan-id", "--scan_id", dest="scan_id", required=True)
    ap.add_argument("--frame-id", "--frame_id", dest="frame_id", type=str, default=None)
    ap.add_argument("--graphs-3dssg", "--graphs_3dssg", dest="graphs_3dssg", required=True)
    ap.add_argument("--output", type=Path, default=Path("docs/figures"))
    ap.add_argument("--mode", choices=["trace", "all"],
                    default="all")
    ap.add_argument("--max-rounds", "--max_rounds", dest="max_rounds",
                    type=int, default=12)
    ap.add_argument("--topdown-size", "--topdown_size", dest="topdown_size",
                    type=int, default=2048)
    return ap.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Load scene
    from tools.viz.visualize_teaser import (
        load_scene_any, detect_up_axis, render_topdown, run_localization,
    )
    import json, torch

    scan_dir = args.root / args.scan_id
    print(f"[1/5] Loading scene: {args.scan_id}")
    mesh, tri2obj, obj2faces = load_scene_any(scan_dir, dataset=args.dataset)
    up_axis = detect_up_axis(mesh)

    # 2. Load frame description
    desc_dir = scan_dir / "output" / "descriptions"
    from tools.viz.visualize_teaser import _load_frame_description
    # Strip extension if user passed e.g. "003693.jpg" instead of "003693"
    frame_id = Path(args.frame_id).stem if args.frame_id else args.frame_id
    frame_data = _load_frame_description(desc_dir, frame_id)
    query_text = frame_data.get("description", "")
    print(f"[2/5] Query: \"{query_text}\"")

    from langloc.localization.frame_io import frame_to_scenegraph, camera_center_from_pose
    import numpy as np
    query_sg, _ = frame_to_scenegraph(frame_data, embedding_type="word2vec",
                                       use_attributes=True)
    scene_pose = frame_data.get("scene_pose")
    gt_pos = camera_center_from_pose(scene_pose)
    pose_mat = np.array(scene_pose, dtype=np.float64)
    gt_dir = pose_mat[:3, 2]
    gt_dir /= max(float(np.linalg.norm(gt_dir)), 1e-6)

    # 3. Run localization
    print(f"[3/5] Running localization...")
    cams, probs, obj_ids, pred_pos, pred_dir, cam_dirs = run_localization(
        mesh, tri2obj, obj2faces,
        query_sg=query_sg, graphs_3dssg=args.graphs_3dssg,
        scan_id=args.scan_id, up_axis=up_axis,
    )

    # 4. Top-down render
    print(f"[4/5] Rendering top-down view...")
    topdown_img, intrinsic, extrinsic = render_topdown(
        mesh, up_axis, args.topdown_size)

    # 5. Run dialogue + render figures
    print(f"[5/5] Running dialogue (all backends, {args.max_rounds} rounds)...")
    setup = build_dialogue_setup(
        cams, probs, cam_dirs, pred_pos, pred_dir,
        gt_pos, gt_dir, args.root, args.scan_id,
        max_rounds=args.max_rounds,
    )
    snapshots, results = run_dialogue_all_backends(
        setup, cams, gt_pos, gt_dir, max_rounds=args.max_rounds)

    anchor_probs = probs.copy()  # pre-dialogue probs for heatmap extent

    # Output directory: {output}/dialogue/{scan_id}/
    out_dir = args.output / "dialogue" / args.scan_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {out_dir}")

    if args.mode in ("trace", "all"):
        render_dialogue_trace(
            snapshots, topdown_img, intrinsic, extrinsic, cams,
            gt_pos, gt_dir, pred_pos, pred_dir,
            results, anchor_probs, up_axis, out_dir,
        )

    print(f"Done. Scene: {args.scan_id} | Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
