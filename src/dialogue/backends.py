"""Bayesian inference backends for pose-level dialogue.

Three parallel implementations that maintain a posterior distribution over
hypotheses (candidate poses, particles, or frames) and update it as
yes/no/unknown answers arrive:

- :class:`CandidateBackendA1` — discrete posterior over candidate poses,
  with frame semantics accessed via a candidate→frame mapping matrix *W*.
- :class:`ParticleBackendA2` — continuous particle filter with KNN-based
  frame lookups.
- :class:`FrameBackendA3` — discrete posterior directly over pooled frames.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.dialogue.likelihood import (
    salience_to_answerable,
    salience_to_visprob,
    ynu_likelihood_from_prob,
)
from src.dialogue.math_utils import _normalize, bayes_update


# ---------------------------------------------------------------------------
# A1 — Candidate posterior
# ---------------------------------------------------------------------------
class CandidateBackendA1:
    """Discrete Bayesian posterior over candidate poses.

    Each candidate is linked to a set of frames via a row-normalised
    mapping matrix *W* (shape ``(N_candidates, F_pool)``).  Label and
    relation probabilities are aggregated across frames via *W*.

    Attributes:
        cand_pos: ``(N, 3)`` candidate positions.
        cand_dir: ``(N, 3)`` candidate directions, or ``None``.
        p: Current posterior vector over candidates.
        W: Row-normalised candidate→frame mapping matrix.
        frame_label_dicts: Per-frame ``label → salience`` dictionaries.
        frame_rel_sets: Per-frame sets of ``(subj, rel, obj)`` tuples.
        frame_dirs: ``(F_pool, 3)`` frame directions.
    """

    def __init__(
        self,
        cand_pos: np.ndarray,
        cand_dir: Optional[np.ndarray],
        cand_prior: np.ndarray,
        c2f_pool: np.ndarray,
        frame_label_dicts: List[Dict[str, float]],
        frame_rel_sets: List[set],
        frame_dirs: np.ndarray,
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

        self._label_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def top_prob(self) -> float:
        """Return the highest posterior probability."""
        return float(self.p.max())

    def posterior_vector(self) -> np.ndarray:
        """Return the current posterior vector over candidates."""
        return self.p

    def frame_posterior(self) -> np.ndarray:
        """Return the induced posterior over pooled frames."""
        pf = self.W.T @ self.p
        s = float(pf.sum())
        return pf / s if s > self.eps else np.ones_like(pf) / len(pf)

    def _frame_label_arrays(self, label: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(p_vis, p_ans)`` arrays over pooled frames for *label*."""
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
        """Return ``(p_yes, p_ans)`` per candidate for a label question."""
        p_vis_f, p_ans_f = self._frame_label_arrays(label)
        return self.W @ p_vis_f, self.W @ p_ans_f

    def rel_prob_true_and_answerable(self, triple: Tuple[str, str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(p_true, p_ans)`` per candidate for a relation question."""
        s, r, o = triple
        true_f = np.array([1.0 if triple in rels else 0.0 for rels in self.frame_rel_sets], dtype=np.float64)
        p_s_vis, p_s_ans = self._frame_label_arrays(s)
        p_o_vis, p_o_ans = self._frame_label_arrays(o)
        ans_f = np.minimum(p_s_ans, p_o_ans)
        return self.W @ true_f, self.W @ ans_f

    def update_label(self, label: str, ans: str) -> None:
        """Update the posterior given a yes/no/unknown answer about *label*."""
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

    def update_rel(self, triple: Tuple[str, str, str], ans: str) -> None:
        """Update the posterior given a yes/no/unknown answer about a relation."""
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
        """Return ``(map_pos, map_dir, mean_pos, mean_dir)``."""
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


# ---------------------------------------------------------------------------
# A3 — Frame posterior
# ---------------------------------------------------------------------------
class FrameBackendA3:
    """Discrete Bayesian posterior directly over pooled frames.

    Simpler than :class:`CandidateBackendA1` — no mapping layer.  Label
    and relation probabilities come directly from per-frame semantics.

    Attributes:
        frames: List of frame objects in the pool.
        p: Current posterior vector over frames.
        frame_label_dicts: Per-frame ``label → salience`` dictionaries.
        frame_rel_sets: Per-frame sets of ``(subj, rel, obj)`` tuples.
        frame_pos: ``(F_pool, 3)`` frame positions.
        frame_dir: ``(F_pool, 3)`` frame directions.
    """

    def __init__(
        self,
        p0: np.ndarray,
        frames_pool: Sequence[Any],
        frame_label_dicts: List[Dict[str, float]],
        frame_rel_sets: List[set],
        frame_pos: np.ndarray,
        frame_dir: np.ndarray,
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
        """Return the highest posterior probability."""
        return float(self.p.max())

    def posterior_vector(self) -> np.ndarray:
        """Return the current posterior vector over frames."""
        return self.p

    def frame_posterior(self) -> np.ndarray:
        """Return a copy of the frame posterior."""
        return self.p.copy()

    def _frame_label_arrays(self, label: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(p_vis, p_ans)`` arrays over pooled frames for *label*."""
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
        """Return ``(p_true, p_ans)`` per frame for a label question."""
        return self._frame_label_arrays(label)

    def rel_prob_true_and_answerable(self, triple: Tuple[str, str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(p_true, p_ans)`` per frame for a relation question."""
        s, r, o = triple
        true = np.array([1.0 if triple in rels else 0.0 for rels in self.frame_rel_sets], dtype=np.float64)
        _, a_s = self._frame_label_arrays(s)
        _, a_o = self._frame_label_arrays(o)
        ans = np.minimum(a_s, a_o)
        return true, ans

    def update_label(self, label: str, ans: str) -> None:
        """Update the posterior given a yes/no/unknown answer about *label*."""
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

    def update_rel(self, triple: Tuple[str, str, str], ans: str) -> None:
        """Update the posterior given a yes/no/unknown answer about a relation."""
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
        """Return ``(map_pos, map_dir, mean_pos, mean_dir)``."""
        topf = int(np.argmax(self.p))
        map_pos = self.frame_pos[topf].copy()
        map_dir = _normalize(self.frame_dir[topf])
        mean_pos = (self.p[:, None] * self.frame_pos).sum(axis=0)
        mean_dir = _normalize((self.p[:, None] * self.frame_dir).sum(axis=0))
        return map_pos, map_dir, mean_pos, mean_dir


# ---------------------------------------------------------------------------
# A2 — Particle filter
# ---------------------------------------------------------------------------
class ParticleBackendA2:
    """Continuous particle filter over pose space.

    Particles are initialised from candidate positions with jitter and
    linked to frame semantics via KNN lookups.

    Attributes:
        p_pos: ``(P, 3)`` particle positions.
        p_dir: ``(P, 3)`` particle directions.
        w: ``(P,)`` normalised particle weights.
        frame_label_dicts: Per-frame ``label → salience`` dictionaries.
        frame_rel_sets: Per-frame sets of ``(subj, rel, obj)`` tuples.
        frame_pos: ``(F_pool, 3)`` frame positions.
        frame_dir: ``(F_pool, 3)`` frame directions.
    """

    def __init__(
        self,
        cand_pos: np.ndarray,
        cand_dir: Optional[np.ndarray],
        cand_prior: np.ndarray,
        frame_label_dicts: List[Dict[str, float]],
        frame_rel_sets: List[set],
        frame_pos: np.ndarray,
        frame_dir: np.ndarray,
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
        """Return the highest particle weight."""
        return float(self.w.max())

    def posterior_vector(self) -> np.ndarray:
        """Return the current particle weight vector."""
        return self.w

    def _nearest_frame_idx(self, X: np.ndarray) -> np.ndarray:
        """Return indices of the nearest frame for each row of *X*."""
        d2 = ((X[:, None, :] - self.frame_pos[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d2, axis=1)

    def _knn(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return K-nearest frame indices and distances for each row of *X*."""
        d2 = ((X[:, None, :] - self.frame_pos[None, :, :]) ** 2).sum(axis=2)
        K = min(self.k_nn, d2.shape[1])
        idx = np.argpartition(d2, kth=K - 1, axis=1)[:, :K]
        d2k = np.take_along_axis(d2, idx, axis=1)
        order = np.argsort(d2k, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        d = np.sqrt(np.take_along_axis(d2, idx, axis=1))
        return idx, d

    def _gauss_w(self, d: np.ndarray) -> np.ndarray:
        """Gaussian kernel weights from distances."""
        if self.sigma <= 0:
            w = np.ones_like(d)
        else:
            w = np.exp(-(d ** 2) / (2.0 * self.sigma * self.sigma))
        s = w.sum(axis=1, keepdims=True)
        return w / np.maximum(s, self.eps)

    def frame_posterior(self) -> np.ndarray:
        """Return the induced posterior over frames (via nearest-neighbour)."""
        nn = self._nearest_frame_idx(self.p_pos)
        F = len(self.frame_pos)
        pf = np.zeros(F, dtype=np.float64)
        for i, j in enumerate(nn):
            pf[int(j)] += float(self.w[i])
        s = float(pf.sum())
        return pf / s if s > self.eps else np.ones(F) / F

    def _frame_label_arrays(self, label: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(p_vis, p_ans)`` arrays over pooled frames for *label*."""
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
        """Return ``(p_true, p_ans)`` per particle for a label question."""
        p_vis_f, p_ans_f = self._frame_label_arrays(label)
        idx, d = self._knn(self.p_pos)
        w = self._gauss_w(d)
        p_true = (w * np.take(p_vis_f, idx)).sum(axis=1)
        p_ans = (w * np.take(p_ans_f, idx)).sum(axis=1)
        return p_true, p_ans

    def rel_prob_true_and_answerable(self, triple: Tuple[str, str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(p_true, p_ans)`` per particle for a relation question."""
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

    def _resample(self) -> None:
        """Resample particles with replacement and add jitter."""
        P = len(self.w)
        idxs = self.rng.choice(P, size=P, replace=True, p=self.w)
        self.p_pos = self.p_pos[idxs] + self.rng.normal(scale=self.jitter_pos, size=self.p_pos.shape)
        self.p_dir = self.p_dir[idxs]
        self.w = np.ones(P, dtype=np.float64) / P

    def update_label(self, label: str, ans: str) -> None:
        """Update particle weights given a yes/no/unknown answer about *label*."""
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

    def update_rel(self, triple: Tuple[str, str, str], ans: str) -> None:
        """Update particle weights given a yes/no/unknown answer about a relation."""
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
        """Return ``(map_pos, map_dir, mean_pos, mean_dir)``."""
        i_map = int(np.argmax(self.w))
        map_pos = self.p_pos[i_map].copy()
        map_dir = _normalize(self.p_dir[i_map])
        mean_pos = (self.w[:, None] * self.p_pos).sum(axis=0)
        mean_dir = _normalize((self.w[:, None] * self.p_dir).sum(axis=0))
        return map_pos, map_dir, mean_pos, mean_dir
