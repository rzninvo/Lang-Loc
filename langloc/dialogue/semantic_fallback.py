"""Semantic-first dialogue with visible-object fallback.

Implements the ``DialogueSemanticFallback`` class — a simpler, self-contained
Bayesian dialogue approach that prioritises spatial-relation questions and
falls back to visible-object questions when no informative relation can be
found.

Posterior is maintained over candidate poses ``P(c)`` and projected to frames
via the ``CandToFrameMap`` Gaussian mapping.

Also provides evaluation metrics, an interactive CLI runner, and a standalone
``main()`` entry point.

Key exports:
    RelTriple, PickedQuestion, DialogueSemanticFallback,
    parse_csv_floats, compute_metrics, run_entry_interactive, main.

Example::

    python -m langloc.dialogue.semantic_fallback \\
        --candidates_json /path/to/candidates.json \\
        --dataset_root /path/to/3RScan \\
        --only_scene_id <scene_id> --limit 1 \\
        --object_pool mapped --stop_mode frame --show_gt_debug
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from langloc.dialogue.frame_mapping import (
    CandToFrameMap,
    build_cand_to_frame_map,
    label_pool_from_frames,
    rel_pool_from_frames,
    top_frames_by_mapping,
)
from langloc.dialogue.scene_data import (
    DEFAULT_ALIASES,
    FrameInfo,
    load_relaxed_json,
    load_scene_data,
    parse_aliases,
    relation_to_phrase,
)


# ---------------------------------------------------------------------------
# Dialogue data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelTriple:
    """An immutable spatial relation triple.

    Attributes:
        subj: Subject label (canonical).
        rel: Relation predicate (canonical).
        obj: Object label (canonical).
    """

    subj: str
    rel: str
    obj: str


@dataclass
class PickedQuestion:
    """A question selected by the dialogue policy.

    Attributes:
        qtype: ``"rel"`` for relation question, ``"label"`` for visibility.
        idx: Index into the corresponding pool.
    """

    qtype: str  # "rel" or "label"
    idx: int


# ---------------------------------------------------------------------------
# Dialogue engine
# ---------------------------------------------------------------------------

class DialogueSemanticFallback:
    """Bayesian dialogue engine: semantic relations first, visible-label fallback.

    Maintains a posterior ``p`` over candidate poses and projects it onto
    frames via the ``CandToFrameMap``.  Precomputes per-candidate visibility
    and relation truth/answerable matrices for efficient Bayesian updates.

    Args:
        cand_prob: Prior probability over candidates, shape ``(N,)``.
        c2f: Candidate-to-frame mapping.
        frames: All scene frames.
        label_pool: Labels to consider for visibility questions.
        rel_pool: Relation triples to consider for semantic questions.
        ignore_labels: Labels to exclude from the pool.
    """

    def __init__(
        self,
        cand_prob: np.ndarray,
        c2f: CandToFrameMap,
        frames: List[FrameInfo],
        label_pool: Sequence[str],
        rel_pool: Sequence[Tuple[str, str, str]],
        ignore_labels: Sequence[str],
    ) -> None:
        """Initialise the dialogue engine and precompute projection matrices."""
        p = cand_prob.astype(np.float64)
        self.p = p / max(p.sum(), 1e-12)

        self.c2f = c2f
        self.frames = frames

        ignore = set(s.strip().lower() for s in ignore_labels if s.strip())

        self.labels = [l for l in sorted(set(label_pool)) if l and l not in ignore]
        self.asked_labels = np.zeros((len(self.labels),), dtype=bool)
        self.label_index = {l: i for i, l in enumerate(self.labels)}

        rels: List[RelTriple] = []
        for s, r, o in rel_pool:
            if not s or not o:
                continue
            if s in ignore or o in ignore:
                continue
            rels.append(RelTriple(s, r, o))
        self.rels = sorted(set(rels), key=lambda x: (x.subj, x.rel, x.obj))
        self.asked_rels = np.zeros((len(self.rels),), dtype=bool)

        F = len(frames)
        M = len(self.labels)
        R = len(self.rels)

        # frame label matrix FV
        self.FV = np.zeros((F, M), dtype=np.float32)
        for j, fr in enumerate(frames):
            for lab in fr.visible_labels:
                if lab in self.label_index:
                    self.FV[j, self.label_index[lab]] = 1.0

        # frame relation truth RV and answerable AV (both labels visible)
        self.RV = np.zeros((F, R), dtype=np.float32)
        self.AV = np.zeros((F, R), dtype=np.float32)
        for j, fr in enumerate(frames):
            vis = fr.visible_labels
            relset = fr.rel_triples
            for t, tr in enumerate(self.rels):
                if tr.subj in vis and tr.obj in vis:
                    self.AV[j, t] = 1.0
                if (tr.subj, tr.rel, tr.obj) in relset:
                    self.RV[j, t] = 1.0

        # candidate-level projections
        idx = c2f.idx
        w = c2f.w
        N = idx.shape[0]

        self.Vc = np.zeros((N, M), dtype=np.float32)
        self.Rc = np.zeros((N, R), dtype=np.float32)
        self.Ac = np.zeros((N, R), dtype=np.float32)

        for i in range(N):
            fr_idx = idx[i]
            wi = w[i][:, None]  # (K, 1)
            if M:
                self.Vc[i] = (wi * self.FV[fr_idx]).sum(axis=0)
            if R:
                self.Rc[i] = (wi * self.RV[fr_idx]).sum(axis=0)
                self.Ac[i] = (wi * self.AV[fr_idx]).sum(axis=0)

    # -- posteriors ----------------------------------------------------------

    def frame_posterior(self) -> np.ndarray:
        """Compute the posterior over frames by marginalizing candidates.

        Returns:
            Normalised frame posterior, shape ``(F,)``.
        """
        F = len(self.frames)
        pf = np.zeros((F,), dtype=np.float64)
        for i in range(self.c2f.idx.shape[0]):
            pf[self.c2f.idx[i]] += self.p[i] * self.c2f.w[i]
        return pf / max(pf.sum(), 1e-12)

    def label_probs(self) -> np.ndarray:
        """Expected visibility probability for each label under the posterior."""
        if self.Vc.size == 0:
            return np.zeros((0,), dtype=np.float64)
        return (self.p.reshape(1, -1) @ self.Vc).ravel()

    def rel_true_probs(self) -> np.ndarray:
        """Expected truth probability for each relation under the posterior."""
        if self.Rc.size == 0:
            return np.zeros((0,), dtype=np.float64)
        return (self.p.reshape(1, -1) @ self.Rc).ravel()

    def rel_answerable_probs(self) -> np.ndarray:
        """Expected answerability for each relation under the posterior."""
        if self.Ac.size == 0:
            return np.zeros((0,), dtype=np.float64)
        return (self.p.reshape(1, -1) @ self.Ac).ravel()

    # -- question picking ----------------------------------------------------

    def pick_relation(
        self,
        ask_min_p: float,
        ask_max_p: float,
        min_answerable: float,
    ) -> Optional[int]:
        """Pick the most informative un-asked relation question.

        Selects the relation whose truth probability is closest to 0.5
        (maximum entropy split), subject to answerability and range filters.

        Returns:
            Index into ``self.rels``, or ``None`` if no suitable relation.
        """
        if len(self.rels) == 0:
            return None
        rem = np.where(~self.asked_rels)[0]
        if len(rem) == 0:
            return None
        p_true = self.rel_true_probs()[rem]
        p_ans = self.rel_answerable_probs()[rem]

        # only ask if likely answerable
        ok = p_ans >= float(min_answerable)
        rem2 = rem[ok]
        if len(rem2) == 0:
            return None

        p2 = self.rel_true_probs()[rem2]
        in_range = (p2 >= float(ask_min_p)) & (p2 <= float(ask_max_p))
        cand = rem2[in_range] if np.any(in_range) else rem2
        pc = self.rel_true_probs()[cand]
        return int(cand[np.argmin(np.abs(pc - 0.5))])

    def pick_label(
        self,
        ask_min_p: float,
        ask_max_p: float,
    ) -> Optional[int]:
        """Pick the most informative un-asked label question.

        Selects the label whose visibility probability is closest to 0.5.

        Returns:
            Index into ``self.labels``, or ``None`` if no suitable label.
        """
        if len(self.labels) == 0:
            return None
        rem = np.where(~self.asked_labels)[0]
        if len(rem) == 0:
            return None
        p = self.label_probs()[rem]
        in_range = (p >= float(ask_min_p)) & (p <= float(ask_max_p))
        cand = rem[in_range] if np.any(in_range) else rem
        pc = self.label_probs()[cand]
        return int(cand[np.argmin(np.abs(pc - 0.5))])

    def pick_next(
        self,
        ask_min_p: float,
        ask_max_p: float,
        rel_min_answerable: float,
    ) -> Optional[PickedQuestion]:
        """Pick the next question: relation first, label as fallback.

        Returns:
            ``PickedQuestion`` or ``None`` if all questions are exhausted.
        """
        ridx = self.pick_relation(ask_min_p, ask_max_p, rel_min_answerable)
        if ridx is not None:
            return PickedQuestion("rel", ridx)
        lidx = self.pick_label(ask_min_p, ask_max_p)
        if lidx is not None:
            return PickedQuestion("label", lidx)
        return None

    # -- Bayesian updates ----------------------------------------------------

    def update_label(self, lidx: int, yes: bool, soft_eps: float = 1e-3) -> None:
        """Update the posterior given a yes/no answer to a label question.

        Args:
            lidx: Index into ``self.labels``.
            yes: Whether the user answered *yes*.
            soft_eps: Soft floor to prevent posterior collapse.
        """
        self.asked_labels[lidx] = True
        v = self.Vc[:, lidx].astype(np.float64)
        like = v if yes else (1.0 - v)
        newp = self.p * like
        if newp.sum() <= 1e-12:
            like2 = np.where(like > 0.5, 1.0, soft_eps)
            newp = self.p * like2
        self.p = newp / max(newp.sum(), 1e-12)

    def update_rel(self, ridx: int, yes: bool, soft_eps: float = 1e-3) -> None:
        """Update the posterior given a yes/no answer to a relation question.

        Unanswerable frames (where subject or object is not visible) receive
        a neutral likelihood of 0.5.

        Args:
            ridx: Index into ``self.rels``.
            yes: Whether the user answered *yes*.
            soft_eps: Soft floor to prevent posterior collapse.
        """
        self.asked_rels[ridx] = True
        A = self.Ac[:, ridx].astype(np.float64)
        T = self.Rc[:, ridx].astype(np.float64)

        # unanswerable frames -> 0.5 likelihood
        if yes:
            like = 0.5 * (1.0 - A) + T
        else:
            like = 0.5 * (1.0 - A) + (A - T)

        newp = self.p * like
        if newp.sum() <= 1e-12:
            like2 = np.where(like > 0.5, 1.0, soft_eps)
            newp = self.p * like2
        self.p = newp / max(newp.sum(), 1e-12)

    def mark_unknown(self, q: PickedQuestion) -> None:
        """Mark a question as asked without updating the posterior.

        Args:
            q: The question that was answered *unknown*.
        """
        if q.qtype == "rel":
            self.asked_rels[q.idx] = True
        else:
            self.asked_labels[q.idx] = True


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def parse_csv_floats(s: str) -> List[float]:
    """Parse a comma-separated string of floats.

    Args:
        s: Input string (e.g. ``"0.5,1.0"``).

    Returns:
        List of parsed float values.
    """
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def compute_metrics(
    pf: np.ndarray,
    frames: List[FrameInfo],
    gt_idx: Optional[int],
    gt_pos: Optional[np.ndarray],
    dist_thresholds: Sequence[float],
) -> dict:
    """Compute frame-retrieval metrics from a frame posterior.

    Args:
        pf: Frame posterior, shape ``(F,)``.
        frames: List of ``FrameInfo`` objects.
        gt_idx: Index of the ground-truth frame, or ``None``.
        gt_pos: Ground-truth 3D position, or ``None``.
        dist_thresholds: Distance thresholds for within-X-metres metrics.

    Returns:
        Dictionary of metric values (rank, hits, distances, etc.).
    """
    order = np.argsort(-pf)
    top5 = order[: min(5, len(order))]
    top5_mass = float(pf[top5].sum())

    m: dict = {"top5_mass": top5_mass}
    if gt_idx is None:
        m.update({"gt_rank": None, "top1_hit": None, "top2_hit": None, "top5_hit": None})
    else:
        gt_rank = int(np.where(order == int(gt_idx))[0][0]) + 1
        m.update({
            "gt_rank": gt_rank,
            "top1_hit": gt_rank <= 1,
            "top2_hit": gt_rank <= 2,
            "top5_hit": gt_rank <= 5,
        })

    top1 = int(order[0])
    m["top1_frame_id"] = frames[top1].frame_id
    m["top1_prob"] = float(pf[top1])

    if gt_pos is None:
        m["dist_top1_to_gt"] = None
        m["dist_top5_min_to_gt"] = None
        for thr in dist_thresholds:
            m[f"within_{thr}m_top1"] = None
            m[f"within_{thr}m_top5"] = None
    else:
        gt_pos = np.asarray(gt_pos, dtype=np.float32).reshape(3)
        d1 = float(np.linalg.norm(frames[top1].position - gt_pos))
        dmin5 = float(min(np.linalg.norm(frames[int(i)].position - gt_pos) for i in top5))
        m["dist_top1_to_gt"] = d1
        m["dist_top5_min_to_gt"] = dmin5
        for thr in dist_thresholds:
            thr = float(thr)
            m[f"within_{thr}m_top1"] = (d1 <= thr)
            m[f"within_{thr}m_top5"] = (dmin5 <= thr)
    return m


# ---------------------------------------------------------------------------
# Interactive entry runner
# ---------------------------------------------------------------------------

def run_entry_interactive(
    args: argparse.Namespace,
    entry: dict,
    aliases: Dict[str, str],
    ignore_labels: List[str],
    dist_thresholds: List[float],
) -> dict:
    """Run a single interactive dialogue entry via the CLI.

    Loads scene data, builds the candidate-to-frame mapping, creates the
    ``DialogueSemanticFallback`` engine, and loops through question rounds
    until confidence is reached or questions are exhausted.

    Args:
        args: Parsed CLI arguments.
        entry: Single scene entry from the candidates JSON.
        aliases: Label alias mapping.
        ignore_labels: Labels to exclude from the question pool.
        dist_thresholds: Distance thresholds for within-X-metres metrics.

    Returns:
        Dictionary of per-entry metrics.
    """
    scene_id = str(entry["scene_id"])
    gt_frame_id = str(entry.get("frame_id", "unknown"))

    sd = load_scene_data(args.dataset_root, scene_id, aliases)

    if args.use_candidates == "fov":
        cands = entry.get("fov_pose_candidates", [])
        cand_pos = np.array([c["position"] for c in cands], dtype=np.float32)
        cand_dir = np.array([c.get("direction", [0, 0, 1]) for c in cands], dtype=np.float32) if cands else None
        cand_prob = np.ones((len(cands),), dtype=np.float64)
    else:
        cands = entry.get("grid_point_candidates", [])
        cand_pos = np.array([c["position"] for c in cands], dtype=np.float32)
        cand_dir = None
        cand_prob = np.array([c.get("prob", 0.0) for c in cands], dtype=np.float64)
        if cand_prob.sum() <= 1e-12 and len(cands) > 0:
            cand_prob = np.ones((len(cands),), dtype=np.float64)

    if cand_pos.size == 0:
        print(f"[SKIP] {scene_id}/{gt_frame_id}: no candidates")
        return {"scene_id": scene_id, "frame_id": gt_frame_id, "skipped": True}

    c2f = build_cand_to_frame_map(
        cand_pos=cand_pos,
        cand_dir=cand_dir,
        frame_pos=sd.frame_pos,
        frame_dir=sd.frame_dir,
        k_nn=args.k_nn,
        sigma=args.sigma,
        use_direction=args.use_direction,
    )

    top_frames = top_frames_by_mapping(c2f, max_frames=args.max_pool_frames)

    if args.object_pool == "scene":
        label_pool = label_pool_from_frames(sd.frames, range(len(sd.frames)))
    else:
        label_pool = label_pool_from_frames(sd.frames, top_frames)

    rel_pool = rel_pool_from_frames(sd.frames, top_frames, max_rel=args.max_rel_pool)

    dlg = DialogueSemanticFallback(
        cand_prob=cand_prob,
        c2f=c2f,
        frames=sd.frames,
        label_pool=label_pool,
        rel_pool=rel_pool,
        ignore_labels=ignore_labels,
    )

    gt_idx = sd.frame_id_to_idx.get(gt_frame_id, None)
    gt_pos = None
    if isinstance(entry.get("gt_pose"), dict) and "position" in entry["gt_pose"]:
        gt_pos = np.array(entry["gt_pose"]["position"], dtype=np.float32)
    elif gt_idx is not None:
        gt_pos = sd.frames[gt_idx].position

    print(f"\n=== Scene {scene_id} | GT frame {gt_frame_id} ===")
    print(f"Candidates: {cand_pos.shape[0]} | Frames: {len(sd.frames)}")
    print(f"Pools: labels={len(dlg.labels)} | relations={len(dlg.rels)} (semantic-first; fallback=visible)")
    print(f"Mapping: K={args.k_nn}, sigma={args.sigma}, use_direction={args.use_direction}")
    print(f"Stop: mode={args.stop_mode}, conf={args.conf_thresh} | Ask filter: [{args.ask_min_p}, {args.ask_max_p}] | rel_min_answerable={args.rel_min_answerable}")
    if args.show_gt_debug and gt_pos is not None:
        print(f"GT pos: {gt_pos.tolist()} (debug)")

    def show_help() -> None:
        """Print available interactive commands."""
        print("\nCommands: y=yes, n=no, u=unknown/skip, q=quit this entry")
        print("          tf=top frames, tc=top candidates, o=list suggestions, h=help\n")

    show_help()

    questions = 0
    for r in range(args.max_rounds):
        pf = dlg.frame_posterior()
        topCandP = float(dlg.p.max())
        topFrameP = float(pf.max())

        stop_c = topCandP >= args.conf_thresh
        stop_f = topFrameP >= args.conf_thresh
        stop = (
            (args.stop_mode == "frame" and stop_f) or
            (args.stop_mode == "candidate" and stop_c) or
            (args.stop_mode == "either" and (stop_c or stop_f)) or
            (args.stop_mode == "both" and (stop_c and stop_f))
        )
        if stop:
            print(f"Stop ({args.stop_mode}) after {r} rounds (topCandP={topCandP:.3f}, topFrameP={topFrameP:.3f}).")
            break

        q = dlg.pick_next(args.ask_min_p, args.ask_max_p, args.rel_min_answerable)
        if q is None:
            print("No questions left (relations/labels exhausted).")
            break

        print(f"\nRound {r+1} | topCandP={topCandP:.3f}  topFrameP={topFrameP:.3f}")
        if q.qtype == "rel":
            tr = dlg.rels[q.idx]
            p_true = float(dlg.rel_true_probs()[q.idx])
            p_ans = float(dlg.rel_answerable_probs()[q.idx])
            print(f"Ask (semantic): Is **{tr.subj}** {relation_to_phrase(tr.rel)} **{tr.obj}** ?  (P(true)={p_true:.2f}, P(answerable)={p_ans:.2f})")
        else:
            lab = dlg.labels[q.idx]
            p_vis = float(dlg.label_probs()[q.idx])
            print(f"Ask (visible): Do you see **{lab}** ?  (P(visible)={p_vis:.2f})")

        ans = input("[y/n/u/q/tf/tc/o/h] > ").strip().lower()

        if ans in ("h", "?"):
            show_help()
            continue
        if ans == "tf":
            order = np.argsort(-pf)[:8]
            print("\nTop frames:")
            for k, fi in enumerate(order, 1):
                fr = sd.frames[int(fi)]
                print(f"  {k:>2}. {fr.frame_id:<12} P={float(pf[fi]):.3f}  pos={fr.position.tolist()}")
            continue
        if ans == "tc":
            order = np.argsort(-dlg.p)[:8]
            print("\nTop candidates:")
            for k, ci in enumerate(order, 1):
                print(f"  {k:>2}. idx={int(ci):<3} P={float(dlg.p[ci]):.3f}  pos={cand_pos[int(ci)].tolist()}")
            continue
        if ans == "o":
            print("\nSuggested semantic questions:")
            if len(dlg.rels) == 0:
                print("  (none)")
            else:
                pT = dlg.rel_true_probs()
                pA = dlg.rel_answerable_probs()
                rem = np.where(~dlg.asked_rels)[0]
                if len(rem) == 0:
                    print("  (none)")
                else:
                    score = np.abs(pT[rem] - 0.5) + 0.5 * (1.0 - pA[rem])
                    order = rem[np.argsort(score)[:10]]
                    for j in order:
                        tr = dlg.rels[int(j)]
                        print(f"  - {tr.subj} {relation_to_phrase(tr.rel)} {tr.obj} | P(true)={float(pT[j]):.2f}, P(ans)={float(pA[j]):.2f}")

            print("\nSuggested visible-object questions:")
            if len(dlg.labels) == 0:
                print("  (none)")
            else:
                pL = dlg.label_probs()
                remL = np.where(~dlg.asked_labels)[0]
                if len(remL) == 0:
                    print("  (none)")
                else:
                    scoreL = np.abs(pL[remL] - 0.5)
                    orderL = remL[np.argsort(scoreL)[:10]]
                    for j in orderL:
                        print(f"  - {dlg.labels[int(j)]} | P(visible)={float(pL[j]):.2f}")
            continue
        if ans == "q":
            print("Quit this entry.")
            break

        questions += 1

        if ans in ("u", ""):
            dlg.mark_unknown(q)
            continue
        if ans not in ("y", "n"):
            print("Please type y/n/u/q/tf/tc/o/h")
            continue

        if q.qtype == "rel":
            dlg.update_rel(q.idx, yes=(ans == "y"))
        else:
            dlg.update_label(q.idx, yes=(ans == "y"))

    pf = dlg.frame_posterior()
    best_f = int(np.argmax(pf))
    best_c = int(np.argmax(dlg.p))

    print("\n=== Result ===")
    print(f"Best frame: {sd.frames[best_f].frame_id}  P={float(pf[best_f]):.3f}  pos={sd.frames[best_f].position.tolist()}")
    print(f"Best candidate: idx={best_c}  P={float(dlg.p[best_c]):.3f}  pos={cand_pos[best_c].tolist()}")

    if args.show_gt_debug and gt_pos is not None:
        err_f = float(np.linalg.norm(sd.frames[best_f].position - gt_pos))
        err_c = float(np.linalg.norm(cand_pos[best_c] - gt_pos))
        print("\nDebug errors vs GT position:")
        print(f"  best frame pos err: {err_f:.3f} m")
        print(f"  best cand  pos err: {err_c:.3f} m")

    metrics = compute_metrics(pf, sd.frames, gt_idx, gt_pos, dist_thresholds)
    metrics.update({
        "scene_id": scene_id,
        "frame_id": gt_frame_id,
        "questions_asked": questions,
        "skipped": False,
        "n_labels_pool": len(dlg.labels),
        "n_rels_pool": len(dlg.rels),
    })

    print("\n=== Metrics (final) ===")
    print(f"Questions asked: {questions}")
    print(f"GT rank: {metrics.get('gt_rank')}")
    print(f"Top-1 hit: {metrics.get('top1_hit')} | Top-2 hit: {metrics.get('top2_hit')} | Top-5 hit: {metrics.get('top5_hit')}")
    print(f"Top-5 mass: {metrics.get('top5_mass'):.3f}")
    if metrics.get("dist_top1_to_gt") is not None:
        print(f"Dist(top1, GT): {metrics['dist_top1_to_gt']:.3f} m | MinDist(top5, GT): {metrics['dist_top5_min_to_gt']:.3f} m")
        for thr in dist_thresholds:
            print(f"Within {thr}m (top1): {metrics.get(f'within_{thr}m_top1')} | Within {thr}m (top5): {metrics.get(f'within_{thr}m_top5')}")
    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the semantic-fallback dialogue."""
    p = argparse.ArgumentParser(
        description="Semantic-first dialogue with visible-object fallback.",
    )
    p.add_argument("--candidates_json", type=Path, required=True)
    p.add_argument("--dataset_root", type=Path, required=True)

    p.add_argument("--use_candidates", choices=["grid", "fov"], default="grid")
    p.add_argument("--k_nn", type=int, default=15)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--use_direction", action="store_true")

    p.add_argument("--object_pool", choices=["mapped", "scene"], default="mapped")

    p.add_argument("--max_rounds", type=int, default=12)
    p.add_argument("--conf_thresh", type=float, default=0.85)
    p.add_argument("--stop_mode", choices=["frame", "candidate", "either", "both"], default="frame")

    p.add_argument("--ask_min_p", type=float, default=0.01)
    p.add_argument("--ask_max_p", type=float, default=0.99)
    p.add_argument("--rel_min_answerable", type=float, default=0.10)

    p.add_argument("--ignore_labels", type=str, default="floor,wall,ceiling,room,baseboard,carpet")
    p.add_argument("--label_aliases", type=str, default="", help="JSON or 'a=b,c=d'")

    p.add_argument("--max_pool_frames", type=int, default=30)
    p.add_argument("--max_rel_pool", type=int, default=600)

    p.add_argument("--dist_thresholds", type=str, default="0.5,1.0")

    p.add_argument("--only_scene_id", type=str, default=None)
    p.add_argument("--only_frame_id", type=str, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)

    p.add_argument("--show_gt_debug", action="store_true")
    p.add_argument("--save_metrics_json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    """Run batch interactive dialogue evaluation from the CLI."""
    args = parse_args()
    data = load_relaxed_json(args.candidates_json)
    scenes = data.get("scenes", [])
    if not scenes:
        raise RuntimeError("No 'scenes' in candidates JSON")

    aliases = dict(DEFAULT_ALIASES)
    aliases.update(parse_aliases(args.label_aliases))

    ignore_labels = [s.strip().lower() for s in str(args.ignore_labels).split(",") if s.strip()]
    dist_thresholds = parse_csv_floats(args.dist_thresholds)

    subset = scenes[args.start:]
    if args.only_scene_id:
        subset = [e for e in subset if str(e.get("scene_id")) == str(args.only_scene_id)]
    if args.only_frame_id:
        subset = [e for e in subset if str(e.get("frame_id")) == str(args.only_frame_id)]
    if args.limit and args.limit > 0:
        subset = subset[:args.limit]

    if not subset:
        print("No entries matched your filters.")
        return

    all_metrics: List[dict] = []
    for i, entry in enumerate(subset, 1):
        m = run_entry_interactive(args, entry, aliases, ignore_labels, dist_thresholds)
        all_metrics.append(m)

        if i < len(subset):
            cont = input("\nProceed to next entry? [Enter=yes / q=stop] > ").strip().lower()
            if cont == "q":
                break

    valid = [m for m in all_metrics if not m.get("skipped", False)]
    if valid:
        n = len(valid)
        top1 = sum(1 for m in valid if m.get("top1_hit") is True) / n
        top2 = sum(1 for m in valid if m.get("top2_hit") is True) / n
        top5 = sum(1 for m in valid if m.get("top5_hit") is True) / n
        mean_top5_mass = float(np.mean([m["top5_mass"] for m in valid if m.get("top5_mass") is not None]))
        mean_q = float(np.mean([m["questions_asked"] for m in valid if m.get("questions_asked") is not None]))
        ranks = [m["gt_rank"] for m in valid if m.get("gt_rank") is not None]

        print("\n====================")
        print("Aggregate summary")
        print("====================")
        print(f"Entries: {n}")
        print(f"Top-1 acc: {top1:.3f} | Top-2 acc: {top2:.3f} | Top-5 acc: {top5:.3f}")
        if ranks:
            print(f"Mean GT rank: {float(np.mean(ranks)):.2f} | Median GT rank: {float(np.median(ranks)):.0f}")
        print(f"Mean Top-5 mass: {mean_top5_mass:.3f}")
        print(f"Mean questions asked: {mean_q:.2f}")

        for thr in dist_thresholds:
            k1 = f"within_{thr}m_top1"
            k5 = f"within_{thr}m_top5"
            w1 = sum(1 for m in valid if m.get(k1) is True) / n
            w5 = sum(1 for m in valid if m.get(k5) is True) / n
            print(f"Within {thr}m (top1): {w1:.3f} | Within {thr}m (top5): {w5:.3f}")

    if args.save_metrics_json:
        args.save_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_metrics_json.write_text(json.dumps(all_metrics, indent=2))
        print(f"\nSaved metrics to: {args.save_metrics_json}")


if __name__ == "__main__":
    main()
