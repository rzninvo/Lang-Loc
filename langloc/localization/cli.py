"""Hydra CLI entry point for localization evaluation.

Run with::

    python -m langloc.localization.cli mode=standard root=$RSCAN_ROOT graphs=$GRAPHS_DIR

Or override any parameter defined in ``conf/localization/default.yaml``.
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from langloc.localization.evaluation import run_evaluation
from langloc.utils.seed import CANONICAL_SEED, set_seed


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Localization evaluation entry point.

    Reads the ``localization`` section from the merged Hydra config
    and delegates to :func:`~langloc.localization.evaluation.run_evaluation`.

    Args:
        cfg: Merged Hydra configuration.
    """
    set_seed(int(cfg.localization.get("seed", CANONICAL_SEED)))
    run_evaluation(cfg.localization, graph_cfg=cfg.graph)


if __name__ == "__main__":
    main()
