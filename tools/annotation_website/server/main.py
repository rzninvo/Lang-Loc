"""FastAPI entry point.

Cookie-based annotator identity: a UUID is set on first visit and signed
with ``itsdangerous`` so we can detect tampering. Nickname (and an
optional 4-digit PIN, planned in M1) live in the database, keyed on the
UUID.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import db as db_module
from .config import Settings, get_settings
from .deps import AnnotatorCookieMiddleware
from .models import Keyframe, Scene
from .routes import api as api_routes
from .routes import pages as pages_routes


log = logging.getLogger("annotation_website")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")


app = FastAPI(title="LangLoc Annotation Website")
app.add_middleware(AnnotatorCookieMiddleware)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _startup() -> None:
    s = get_settings()
    db_module.init_db()
    _seed_pool_from_json(s)
    if s.cookie_secret == "dev-secret-CHANGE-ME":
        log.warning("[WARN] cookie_secret default in use, set LANGLOC_COOKIE_SECRET in production")
    if s.admin_token == "dev-admin-CHANGE-ME":
        log.warning("[WARN] admin_token default in use, set LANGLOC_ADMIN_TOKEN in production")
    log.info("annotation website ready on http://%s:%d", s.host, s.port)


def _seed_pool_from_json(s: Settings) -> None:
    """Idempotently populate ``scenes`` and ``keyframes`` from a per-dataset
    JSON file (``data/scenes_<dataset>.json``).

    ``scripts/compute_difficulty.py`` writes one file per dataset;
    ``prepare_keyframes.py`` writes the keyframe JPEGs that those files
    reference.
    """
    factory = db_module.get_session_factory()
    sess = factory()
    try:
        seeded_datasets = []
        for ds_key in s.datasets.keys():
            pool_path = s.site_root / "data" / f"scenes_{ds_key}.json"
            if not pool_path.exists():
                log.warning(
                    "[WARN] scenes_%s.json missing at %s; skipping this dataset for now",
                    ds_key, pool_path,
                )
                continue
            pool = json.loads(pool_path.read_text())
            n_scenes = 0
            for entry in pool["scenes"]:
                scene_id = entry["scene_id"]
                existing = sess.get(Scene, scene_id)
                if existing is None:
                    sess.add(
                        Scene(
                            id=scene_id,
                            dataset=ds_key,
                            display_index=entry.get("display_index", 0),
                            difficulty_tertile=entry.get("difficulty_tertile", 1),
                            difficulty_rank=entry.get("difficulty_rank", 999),
                            num_frames=len(entry.get("frames", [])),
                        )
                    )
                    sess.flush()
                else:
                    # idempotent updates so re-running compute_difficulty.py
                    # picks up new ranks without losing prior descriptions
                    existing.dataset = ds_key
                    existing.difficulty_rank = entry.get("difficulty_rank", existing.difficulty_rank)
                    existing.difficulty_tertile = entry.get("difficulty_tertile", existing.difficulty_tertile)
                for frame in entry.get("frames", []):
                    frame_id = frame["frame_id"]
                    key = (scene_id, frame_id)
                    if sess.get(Keyframe, key) is None:
                        sess.add(
                            Keyframe(
                                scene_id=scene_id,
                                frame_id=frame_id,
                                image_path=frame["image_path"],
                            )
                        )
                sess.flush()
                n_scenes += 1
            seeded_datasets.append((ds_key, n_scenes))
        sess.commit()
        for ds_key, n in seeded_datasets:
            log.info("seeded %d scenes for dataset=%s", n, ds_key)
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


# ---------------------------------------------------------------------------
# Static + routes
# ---------------------------------------------------------------------------
_settings = get_settings()
if _settings.static_dir.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(_settings.static_dir)),
        name="static",
    )
if _settings.keyframes_dir.exists():
    app.mount(
        "/keyframes",
        StaticFiles(directory=str(_settings.keyframes_dir)),
        name="keyframes",
    )

app.include_router(pages_routes.router)
app.include_router(api_routes.router)


# Health
@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "ts": datetime.now(tz=timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# Mesh server for the human-as-localizer page (serves one .ply per scene)
# ---------------------------------------------------------------------------
from fastapi import HTTPException
from fastapi.responses import FileResponse


@app.get("/meshes/{dataset}/{scene_id}.ply")
def serve_mesh(dataset: str, scene_id: str) -> FileResponse:
    s = _settings
    if dataset not in s.dataset_roots:
        raise HTTPException(404, f"unknown dataset {dataset}")
    # paranoia: scene_id must not escape the dataset root
    if "/" in scene_id or ".." in scene_id:
        raise HTTPException(400, "invalid scene_id")
    root = s.dataset_roots[dataset]
    if dataset == "scannet":
        # prefer full-resolution mesh if present; fall back to the
        # decimated _2 variant otherwise.
        full = root / scene_id / f"{scene_id}_vh_clean.ply"
        deci = root / scene_id / f"{scene_id}_vh_clean_2.ply"
        candidate = full if full.is_file() else deci
    elif dataset == "3rscan":
        candidate = root / scene_id / "labels.instances.annotated.v2.ply"
    else:
        raise HTTPException(404, f"no mesh recipe for dataset {dataset}")
    if not candidate.is_file():
        raise HTTPException(404, f"mesh file not on disk: {candidate}")
    return FileResponse(
        path=candidate,
        media_type="application/octet-stream",
        filename=candidate.name,
        # ply files are tiny (3-7 MB), encourage browser caching
        headers={"Cache-Control": "public, max-age=86400"},
    )
