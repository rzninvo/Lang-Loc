"""Hydra CLI entry point for dialogue evaluation.

Run with::

    python -m src.dialogue.cli dialogue.candidates_json=path.json dialogue.dataset_root=/path/to/3RScan

Or override any parameter defined in ``conf/dialogue/default.yaml``.
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.dialogue.dialogue_config import extract_dialogue_config
from src.dialogue.eval_runner import run_batch


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Dialogue evaluation entry point.

    Reads the ``dialogue`` section from the merged Hydra config
    and delegates to :func:`~src.dialogue.eval_runner.run_batch`.

    Args:
        cfg: Merged Hydra configuration.
    """
    dialogue_cfg = extract_dialogue_config(cfg.dialogue)
    run_batch(dialogue_cfg)


if __name__ == "__main__":
    main()
