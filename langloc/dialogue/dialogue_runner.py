"""Per-backend dialogue execution loop.

Manages the interactive (or oracle) dialogue loop for a single backend:
pick a question, render it, obtain an answer, and update the posterior.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from langloc.dialogue.likelihood import salience_to_answerable, salience_to_visprob
from langloc.dialogue.qwen_answerer import QwenAnswerer, QwenFrameContext, qwen_answer
from langloc.dialogue.question_pool import HELP_TEXT, Question
from langloc.dialogue.question_selection import (
    pick_next_question_system,
    question_key,
    show_top_candidates,
    show_top_frames,
    show_top_particles,
)
from langloc.dialogue.semantics import rel_item_to_tuple, relation_phrase


# ---------------------------------------------------------------------------
# Oracle answering
# ---------------------------------------------------------------------------
def nearest_frame_to_gt(
    gt_pos: np.ndarray,
    frames_all: Sequence[Any],
    scene: Any,
) -> int:
    """Return the index of the frame nearest to *gt_pos*.

    Args:
        gt_pos: Ground-truth position ``(3,)``.
        frames_all: All frames in the scene.
        scene: Scene object with a ``frame_pos`` attribute.

    Returns:
        Index into *frames_all* of the nearest frame.
    """
    fp = np.asarray(scene.frame_pos, dtype=np.float64)
    d = np.linalg.norm(fp - gt_pos[None, :], axis=1)
    return int(np.argmin(d))


def oracle_answer(
    q: Question,
    label_pool: List[str],
    rel_pool: List[Any],
    gt_frame_label_dict: Dict[str, float],
    gt_frame_rel_set: set,
    cfg: Any,
) -> str:
    """Generate an oracle answer from ground-truth semantics.

    Args:
        q: The question to answer.
        label_pool: Label pool for index resolution.
        rel_pool: Relation pool for index resolution.
        gt_frame_label_dict: ``label → salience`` for the GT frame.
        gt_frame_rel_set: Set of ``(subj, rel, obj)`` tuples for the GT frame.
        cfg: Config with ``vis_tau``, ``ans_tau``, and
            ``oracle_ansable_min``.

    Returns:
        ``"y"``, ``"n"``, or ``"u"``.
    """
    if q.qtype == "label":
        lab = label_pool[q.idx]
        sal = float(gt_frame_label_dict.get(lab, 0.0))
        # if barely visible, treat as unknown
        ansable = salience_to_answerable(sal, cfg.ans_tau)
        vis = salience_to_visprob(sal, cfg.vis_tau)
        if ansable < cfg.oracle_ansable_min:
            return "u"
        return "y" if vis >= 0.5 else "n"
    else:
        s, r, o = rel_item_to_tuple(rel_pool[q.idx])
        # if subj or obj not answerable -> u
        sal_s = float(gt_frame_label_dict.get(s, 0.0))
        sal_o = float(gt_frame_label_dict.get(o, 0.0))
        ans_s = salience_to_answerable(sal_s, cfg.ans_tau)
        ans_o = salience_to_answerable(sal_o, cfg.ans_tau)
        if min(ans_s, ans_o) < cfg.oracle_ansable_min:
            return "u"
        return "y" if (s, r, o) in gt_frame_rel_set else "n"


# ---------------------------------------------------------------------------
# Main dialogue loop for a single backend
# ---------------------------------------------------------------------------
def run_dialogue_one_backend(
    backend_name: str,
    backend: Any,
    questions_init: List[Question],
    label_pool: List[str],
    rel_pool: List[Any],
    idf: Dict[str, float],
    cfg: Any,
    answer_cache: Optional[Dict[Tuple, str]],
    oracle_gt_frame_label_dict: Optional[Dict[str, float]],
    oracle_gt_frame_rel_set: Optional[set],
    cand_pos: Optional[np.ndarray] = None,
    cand_dir: Optional[np.ndarray] = None,
    frames_pool: Optional[Sequence[Any]] = None,
    qwen_answerer: Optional[QwenAnswerer] = None,
    qwen_frame_context: Optional[QwenFrameContext] = None,
) -> int:
    """Run a full dialogue loop for one backend.

    Repeatedly picks a question, obtains an answer (interactively or via
    oracle), and updates the backend's posterior until the confidence
    threshold is reached or the maximum number of rounds is exhausted.

    Args:
        backend_name: Short name (``"a1"``/``"a2"``/``"a3"``).
        backend: Backend instance with the standard API.
        questions_init: Initial pool of candidate questions.
        label_pool: Label pool for rendering questions.
        rel_pool: Relation pool for rendering questions.
        idf: Per-label IDF weights.
        cfg: Dialogue configuration (``DialogueConfig`` or argparse namespace).
        answer_cache: Shared answer cache (or ``None``).
        oracle_gt_frame_label_dict: GT label dict for oracle mode.
        oracle_gt_frame_rel_set: GT relation set for oracle mode.
        cand_pos: Candidate positions (for ``tc`` command, A1 only).
        cand_dir: Candidate directions (for ``tc`` command, A1 only).
        frames_pool: Pooled frame objects (for ``tf`` command).

    Returns:
        Number of substantive questions asked (y/n/u answers).
    """
    questions = list(questions_init)
    asked = 0

    print(f"\n=== Dialogue for {backend_name.upper()} ===")
    print(HELP_TEXT)

    for r in range(cfg.max_rounds):
        tp = backend.top_prob()
        print(f"\n[{backend_name.upper()}] Round {r+1} | topP={tp:.3f}")

        if r + 1 >= cfg.min_rounds and tp >= cfg.conf_threshold:
            print(f"Confident ({tp:.3f} >= {cfg.conf_threshold:.2f})")
            break

        # pick question; if none, relax thresholds progressively
        q = pick_next_question_system(backend_name, backend, questions, label_pool, rel_pool, idf, cfg)
        if q is None and cfg.auto_relax:
            # relax thresholds progressively, always restoring originals
            old_min, old_max, old_ans = cfg.ask_min_p, cfg.ask_max_p, cfg.rel_min_answerable
            try:
                # relax #1: widen p window
                cfg.ask_min_p, cfg.ask_max_p = 0.01, 0.99
                q = pick_next_question_system(backend_name, backend, questions, label_pool, rel_pool, idf, cfg)
                # relax #2: allow relations regardless answerable
                if q is None:
                    cfg.rel_min_answerable = 0.0
                    q = pick_next_question_system(backend_name, backend, questions, label_pool, rel_pool, idf, cfg)
            finally:
                cfg.ask_min_p, cfg.ask_max_p, cfg.rel_min_answerable = old_min, old_max, old_ans

        if q is None:
            print("No more questions that satisfy thresholds.")
            break

        if q.qtype == "label":
            lab = label_pool[q.idx]
            p_true, p_ans = backend.label_prob_yes(lab)
            p_yes = float(np.dot(backend.posterior_vector(), p_true))
            p_ans_avg = float(np.dot(backend.posterior_vector(), p_ans))
            print(f"Ask[label]: Do you see **{lab}** ?  (P(yes)~={p_yes:.2f}, P(ans)~={p_ans_avg:.2f})")
        else:
            s, rrel, o = rel_item_to_tuple(rel_pool[q.idx])
            p_true, p_ans = backend.rel_prob_true_and_answerable((s, rrel, o))
            p_yes = float(np.dot(backend.posterior_vector(), p_true))
            p_ans_avg = float(np.dot(backend.posterior_vector(), p_ans))
            print(f"Ask[rel ]: Is **{s}** {relation_phrase(rrel)} **{o}** ? (P(true)~={p_yes:.2f}, P(ans)~={p_ans_avg:.2f})")

        key = question_key(q, label_pool, rel_pool)
        if answer_cache is not None and key in answer_cache and cfg.cache_answers:
            ans = answer_cache[key]
            print(f"[cached answer: {ans}]")
        else:
            if cfg.answer_mode == "oracle":
                ans = oracle_answer(q, label_pool, rel_pool, oracle_gt_frame_label_dict or {}, oracle_gt_frame_rel_set or set(), cfg)
                print(f"[oracle answer: {ans}]")
            elif cfg.answer_mode == "qwen":
                if qwen_answerer is None or qwen_frame_context is None:
                    raise RuntimeError(
                        "answer_mode=qwen requires qwen_answerer and "
                        "qwen_frame_context to be passed to run_dialogue_one_backend."
                    )
                ans = qwen_answer(q, label_pool, rel_pool, qwen_frame_context, qwen_answerer)
                print(f"[qwen answer: {ans}]")
            else:
                ans = input("[y/n/u/q/tf/tc/tp/o/h] > ").strip().lower()

        if ans in ("h", "?"):
            print(HELP_TEXT)
            continue
        if ans == "tf":
            if hasattr(backend, "frame_posterior") and frames_pool is not None:
                show_top_frames(frames_pool, backend.frame_posterior(), top_n=cfg.show_top_n, title=f"Top frames ({backend_name.upper()})")
            else:
                print("tf not available for this backend.")
            continue
        if ans == "tc":
            if backend_name != "a1" or cand_pos is None:
                print("tc only available in A1 dialogue.")
            else:
                show_top_candidates(cand_pos, cand_dir, backend.posterior_vector(), top_n=cfg.show_top_n)
            continue
        if ans == "tp":
            if backend_name != "a2" or not hasattr(backend, "p_pos"):
                print("tp only available in A2 dialogue.")
            else:
                show_top_particles(backend.p_pos, backend.p_dir, backend.posterior_vector(), top_n=cfg.show_top_n)
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

        if answer_cache is not None and cfg.cache_answers:
            answer_cache[key] = ans

        asked += 1

        if q.qtype == "label":
            backend.update_label(label_pool[q.idx], ans)
        else:
            backend.update_rel(rel_item_to_tuple(rel_pool[q.idx]), ans)

        questions = [qq for qq in questions if not (qq.qtype == q.qtype and qq.idx == q.idx)]

    return asked
