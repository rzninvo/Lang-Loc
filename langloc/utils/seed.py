"""Project-wide reproducibility helper.

Single source of truth for the canonical seed (42) and the deterministic-mode
toggles. Every entry point that does sampling, training, evaluation, or any
form of stochastic computation must call ``set_seed`` at startup.

Usage::

    from langloc.utils.seed import set_seed, CANONICAL_SEED
    set_seed(CANONICAL_SEED)            # default: enable cudnn determinism
    # or, for a CLI script:
    parser.add_argument("--seed", type=int, default=CANONICAL_SEED)
    set_seed(args.seed)

The canonical seed is fixed at 42 across the whole project (see CLAUDE.md
§0). Do not change it without also re-running every reported number.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

CANONICAL_SEED: int = 42


def set_seed(seed: int = CANONICAL_SEED, *, deterministic_cudnn: bool = True) -> None:
    """Seed the Python, NumPy, and PyTorch RNGs.

    Args:
        seed: integer seed to apply to all RNGs.
        deterministic_cudnn: when True (default) sets ``cudnn.deterministic=True``,
            ``cudnn.benchmark=False``, and ``torch.use_deterministic_algorithms(
            True, warn_only=True)`` for byte-stable kernels. Set
            ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` in the environment if you also
            want bit-identical CuBLAS behaviour.

    Notes:
        torch's deterministic-algorithm mode emits warnings rather than errors
        for ops that lack deterministic kernels (e.g. some scatter/gather paths).
        We use ``warn_only=True`` so training does not crash; the warnings are
        worth surfacing to the user but should not abort execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[WARN] set_seed: torch.use_deterministic_algorithms not "
                f"available (got={exc!r}); falling back to "
                f"non-fully-deterministic kernels.",
                flush=True,
            )

    # CuBLAS deterministic behaviour requires this env var (PyTorch logs a
    # warning if it's missing). Set a sensible default if the caller hasn't.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def worker_init_fn(worker_id: int) -> None:
    """``DataLoader`` worker_init_fn: re-seed each worker deterministically.

    Required when any sample selection happens in ``__getitem__`` (e.g.
    random subgraph augmentation, random pair sampling). Without this, each
    DataLoader worker would inherit the parent's RNG state at fork time and
    drift independently across runs.
    """
    base_seed = torch.initial_seed() % (2 ** 31)
    seed = (base_seed + worker_id) % (2 ** 31)
    random.seed(seed)
    np.random.seed(seed)
