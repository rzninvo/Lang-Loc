"""Bayesian likelihood model for yes/no/unknown answers.

Provides the core probabilistic model that maps label salience and spatial
relation truth values to answer likelihoods ``P(yes|h)``, ``P(no|h)``, and
``P(unknown|h)`` for each hypothesis *h*.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def salience_to_visprob(sal: float, tau: float) -> float:
    """Map a salience score to a visibility probability.

    Args:
        sal: Raw salience value in ``[0, 1]``.
        tau: Scale parameter.  A salience of *tau* maps to approximately 1.0.
            If *tau* <= 0 the salience is returned clamped.

    Returns:
        Visibility probability in ``[0.0, 1.0]``.
    """
    if tau <= 0:
        return float(np.clip(sal, 0.0, 1.0))
    return float(np.clip(sal / tau, 0.0, 1.0))


def salience_to_answerable(sal: float, tau: float) -> float:
    """Map a salience score to an answerable probability.

    Higher salience means the question about this label is more likely to
    be answerable from the given viewpoint.

    Args:
        sal: Raw salience value in ``[0, 1]``.
        tau: Scale parameter.  If *tau* <= 0 the salience is returned clamped.

    Returns:
        Answerable probability in ``[0.0, 1.0]``.
    """
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
    """Compute per-hypothesis likelihoods for yes / no / unknown answers.

    For each hypothesis *h*::

        P_u[h] = p_u_base * p_ans[h] + p_u_unanswerable * (1 - p_ans[h])
        P(yes|h) = (1 - P_u) * (alpha * p_true + (1 - alpha) * (1 - p_true))
        P(no |h) = (1 - P_u) * ((1 - alpha) * p_true + alpha * (1 - p_true))
        P(unk|h) = P_u

    Args:
        p_true: Per-hypothesis probability that the label is visible /
            relation is true.
        p_ans: Per-hypothesis probability that the question is answerable.
        alpha: Likelihood calibration parameter (< 0.9 recommended to
            reduce overconfidence).
        p_u_base: Base unknown-answer probability for answerable
            hypotheses.
        p_u_unanswerable: Unknown-answer probability for unanswerable
            hypotheses.
        eps: Numerical floor for likelihood values.

    Returns:
        Tuple ``(P_yes, P_no, P_unknown)`` — each an array of the same
        shape as *p_true*.
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
