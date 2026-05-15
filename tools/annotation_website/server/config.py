"""Runtime configuration for the annotation website.

Reads environment variables with sensible defaults so a single
``python -m server`` is enough to start.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class DatasetConfig:
    key: str               # 'scannet' | '3rscan' (used in URLs / DB)
    label: str             # display name
    blurb: str             # one-paragraph intro
    recommended: bool      # the recommended-by-us card on the chooser
    teaser_image: str      # /static/img/teaser_<key>.jpg
    rotate_landscape: bool # True for 3RScan (stored sideways)


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    site_root: Path
    db_path: Path
    jsonl_path: Path
    keyframes_dir: Path
    scenes_json: Path
    static_dir: Path
    templates_dir: Path

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    lease_ttl_minutes: int = 20
    redundancy_target: int = 3
    cookie_name: str = "langloc_annotator"
    cookie_dataset_name: str = "langloc_dataset"
    cookie_max_age_days: int = 365
    cookie_secret: str = "dev-secret-CHANGE-ME"

    admin_token: str = "dev-admin-CHANGE-ME"

    min_words: int = 0  # disabled — only min_chars gates submission
    min_chars: int = 40
    max_chars: int = 2000
    min_seconds_per_frame: int = 15
    session_target_frames: int = 10

    jpeg_quality: int = 85
    jpeg_long_edge: int = 1024

    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
    # map dataset key -> absolute filesystem root of the raw scan data
    # (used by the human-localizer to read GT pose at submit time)
    dataset_roots: Dict[str, Path] = field(default_factory=dict)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _datasets() -> Dict[str, DatasetConfig]:
    return {
        "scannet": DatasetConfig(
            key="scannet",
            label="ScanNet",
            blurb=(
                "Real apartments and houses scanned with iPad-class RGB-D "
                "cameras. Bigger, better-lit rooms with rich furniture. "
                "Our localization system already does well here, which makes "
                "your descriptions especially valuable for measuring how "
                "humans compare."
            ),
            recommended=True,
            teaser_image="/static/img/teaser_scannet.jpg",
            rotate_landscape=False,
        ),
        "3rscan": DatasetConfig(
            key="3rscan",
            label="3RScan",
            blurb=(
                "Smaller European apartments captured with a phone-class "
                "depth scanner. Tighter rooms and more bathrooms / "
                "corridors. The harder of the two datasets; descriptions "
                "here help us understand the failure modes."
            ),
            recommended=False,
            teaser_image="/static/img/teaser_3rscan.jpg",
            rotate_landscape=True,
        ),
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    site_root = Path(__file__).resolve().parents[1]
    repo_root = site_root.parents[1]
    return Settings(
        repo_root=repo_root,
        site_root=site_root,
        db_path=site_root / "data" / "annotations.db",
        jsonl_path=site_root / "data" / "annotations.jsonl",
        keyframes_dir=site_root / "static" / "keyframes",
        scenes_json=site_root / "data" / "scenes.json",
        static_dir=site_root / "static",
        templates_dir=site_root / "templates",
        host=_env_str("LANGLOC_HOST", "0.0.0.0"),
        port=_env_int("LANGLOC_PORT", 8000),
        reload=bool(int(_env_str("LANGLOC_RELOAD", "0"))),
        lease_ttl_minutes=_env_int("LANGLOC_LEASE_TTL_MIN", 20),
        redundancy_target=_env_int("LANGLOC_REDUNDANCY", 3),
        cookie_secret=_env_str("LANGLOC_COOKIE_SECRET", "dev-secret-CHANGE-ME"),
        admin_token=_env_str("LANGLOC_ADMIN_TOKEN", "dev-admin-CHANGE-ME"),
        datasets=_datasets(),
        dataset_roots={
            "scannet": repo_root / "data" / "scans",
            "3rscan":  repo_root / "data" / "3RScan",
        },
    )
