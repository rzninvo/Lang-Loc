"""Hydra CLI entry point for dialogue evaluation.

Run with::

    python -m langloc.dialogue.cli dialogue.candidates_json=path.json dialogue.dataset_root=/path/to/3RScan

Or override any parameter defined in ``conf/dialogue/default.yaml``.
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from langloc.dialogue.dialogue_config import extract_dialogue_config
from langloc.dialogue.eval_runner import run_batch
from langloc.utils.seed import CANONICAL_SEED, set_seed


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Dialogue evaluation entry point.

    Reads the ``dialogue`` section from the merged Hydra config
    and delegates to :func:`~langloc.dialogue.eval_runner.run_batch`.

    Args:
        cfg: Merged Hydra configuration.
    """
    dialogue_cfg = extract_dialogue_config(cfg.dialogue)
    set_seed(getattr(dialogue_cfg, "seed", CANONICAL_SEED))
    run_batch(dialogue_cfg)


if __name__ == "__main__":
    main()
