"""Load the LangLoc Hydra configuration tree as a plain dict.

Entry-point scripts that use ``@hydra.main`` receive the config
automatically.  This helper exists for standalone scripts (e.g. in
``scripts/``) that still need programmatic access to resolved paths
and dataset parameters without running inside Hydra.
"""

import json
import os
from pathlib import Path

from omegaconf import OmegaConf


_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(config_dir=None):
    """Compose and resolve the Hydra config tree, returning a plain dict.

    Parameters
    ----------
    config_dir : str | Path | None
        Override for the ``configs/`` directory.  Defaults to the repo's
        ``configs/`` relative to this file's location.
    """
    cfg_dir = Path(config_dir) if config_dir else _CONFIGS_DIR
    paths = OmegaConf.load(cfg_dir / "paths" / "default.yaml")
    dataset = OmegaConf.load(cfg_dir / "dataset" / "default.yaml")
    cfg = OmegaConf.create({"paths": paths, "dataset": dataset})
    OmegaConf.resolve(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def get_download_config():
    """Return (base_dir, label_map_path, file_types) for ScanNet downloads."""
    cfg = load_config()
    base_dir = str(cfg["paths"]["data_root"])
    label_map = os.path.join(base_dir, cfg["dataset"]["download"]["label_map_filename"])
    files = cfg["dataset"]["download"]["file_types"]
    return base_dir, label_map, files


if __name__ == "__main__":
    try:
        base_dir, label_map, files = get_download_config()
        print(json.dumps({
            "base_dir": base_dir,
            "label_map": label_map,
            "file_types": files
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        exit(1)
