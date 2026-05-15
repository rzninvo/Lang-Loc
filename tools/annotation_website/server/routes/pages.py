"""HTML page routes (server-side rendered Jinja2 templates)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc as sa_desc, func, select
from sqlalchemy.orm import Session

from .. import lease as lease_mod
from .. import localization as loc_mod
from ..assignment import assign_next
from ..config import get_settings
from ..deps import get_annotator, get_db
from ..models import (
    Annotator,
    Description,
    FrameCompletion,
    HumanLocalization,
    Keyframe,
    Scene,
)


router = APIRouter()
templates = Jinja2Templates(directory=str(get_settings().templates_dir))


def _static_version(rel: str) -> str:
    """Return a short cache-busting token derived from the file's
    mtime, so updates to ``static/js/localize.js`` force browsers to
    re-fetch even when their HTTP cache would otherwise hold the file.
    """
    p = get_settings().static_dir / rel
    try:
        return str(int(p.stat().st_mtime))
    except FileNotFoundError:
        return "0"


# Expose the helper to all templates.
templates.env.globals["static_version"] = _static_version


def _coverage_summary(db: Session, dataset: Optional[str] = None) -> dict:
    s = get_settings()
    kf_q = select(func.count()).select_from(Keyframe)
    fc_q = select(func.count()).select_from(FrameCompletion)
    if dataset is not None:
        kf_q = kf_q.join(Scene, Scene.id == Keyframe.scene_id).where(Scene.dataset == dataset)
        fc_q = fc_q.join(Scene, Scene.id == FrameCompletion.scene_id).where(Scene.dataset == dataset)
    total_frames = db.scalar(kf_q) or 0
    total_completions = db.scalar(fc_q) or 0
    target = s.redundancy_target
    target_total = total_frames * target
    return {
        "total_frames": total_frames,
        "total_completions": total_completions,
        "target_total": target_total,
        "pct": (100.0 * total_completions / target_total) if target_total else 0.0,
    }


def _relative_time(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(tz=timezone.utc) - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        m = seconds // 60
        return f"{m} min{'s' if m != 1 else ''} ago"
    if seconds < 86400:
        h = seconds // 3600
        return f"{h} hour{'s' if h != 1 else ''} ago"
    d = seconds // 86400
    if d < 7:
        return f"{d} day{'s' if d != 1 else ''} ago"
    return dt.strftime("%b %d")


def _set_dataset_cookie(response: HTMLResponse, dataset: str, secure: bool) -> None:
    s = get_settings()
    response.set_cookie(
        s.cookie_dataset_name,
        dataset,
        max_age=s.cookie_max_age_days * 86400,
        httponly=False,   # not sensitive
        samesite="lax",
        secure=secure,
    )


def _request_is_https(request: Request) -> bool:
    return (
        request.url.scheme == "https"
        or request.headers.get("x-forwarded-proto", "").lower() == "https"
    )


@router.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    db: Session = Depends(get_db),
    annotator: Annotator = Depends(get_annotator),
) -> HTMLResponse:
    coverage = _coverage_summary(db)
    completed_count = db.scalar(
        select(func.count())
        .select_from(FrameCompletion)
        .where(FrameCompletion.annotator_id == annotator.id)
    ) or 0
    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "annotator": annotator,
            "coverage": coverage,
            "your_completed": completed_count,
            "datasets": get_settings().datasets,
            "chosen_dataset": getattr(request.state, "chosen_dataset", "") or None,
        },
    )


@router.get("/datasets", response_class=HTMLResponse)
def datasets_page(
    request: Request,
    db: Session = Depends(get_db),
    annotator: Annotator = Depends(get_annotator),
) -> HTMLResponse:
    s = get_settings()
    cards = []
    for ds_key, ds in s.datasets.items():
        cov = _coverage_summary(db, dataset=ds_key)
        n_scenes = db.scalar(
            select(func.count()).select_from(Scene).where(Scene.dataset == ds_key)
        ) or 0
        my_completed = db.scalar(
            select(func.count())
            .select_from(FrameCompletion)
            .join(Scene, Scene.id == FrameCompletion.scene_id)
            .where(
                FrameCompletion.annotator_id == annotator.id,
                Scene.dataset == ds_key,
            )
        ) or 0
        cards.append({
            "key": ds_key,
            "label": ds.label,
            "blurb": ds.blurb,
            "recommended": ds.recommended,
            "teaser": ds.teaser_image,
            "n_scenes": n_scenes,
            "n_frames": cov["total_frames"],
            "coverage_pct": cov["pct"],
            "your_completed": my_completed,
        })

    return templates.TemplateResponse(
        request,
        "datasets.html",
        {
            "annotator": annotator,
            "cards": cards,
            "chosen_dataset": getattr(request.state, "chosen_dataset", "") or None,
        },
    )


@router.get("/about", response_class=HTMLResponse)
def about(
    request: Request,
    annotator: Annotator = Depends(get_annotator),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "about.html",
        {
            "annotator": annotator,
            "chosen_dataset": getattr(request.state, "chosen_dataset", "") or None,
        },
    )


@router.get("/annotate", response_class=HTMLResponse)
def annotate(
    request: Request,
    dataset: Optional[str] = None,
    db: Session = Depends(get_db),
    annotator: Annotator = Depends(get_annotator),
) -> HTMLResponse:
    s = get_settings()

    # 1. resolve dataset preference: explicit query string > cookie > redirect to chooser
    chosen = dataset or getattr(request.state, "chosen_dataset", "")
    if chosen not in s.datasets:
        return RedirectResponse(url="/datasets", status_code=303)

    pick = assign_next(db, annotator.id, chosen)
    if pick is None:
        return RedirectResponse(url=f"/done?dataset={chosen}", status_code=303)
    scene_id, frame_id = pick
    keyframe = db.get(Keyframe, (scene_id, frame_id))
    scene = db.get(Scene, scene_id)
    if keyframe is None or scene is None:
        raise HTTPException(500, "scene/keyframe missing in db")

    prior = db.get(Description, (scene_id, frame_id, annotator.id))
    completed = db.scalar(
        select(func.count())
        .select_from(FrameCompletion)
        .join(Scene, Scene.id == FrameCompletion.scene_id)
        .where(
            FrameCompletion.annotator_id == annotator.id,
            Scene.dataset == chosen,
        )
    ) or 0

    response = templates.TemplateResponse(
        request,
        "annotate.html",
        {
            "annotator": annotator,
            "scene": scene,
            "keyframe": keyframe,
            "image_url": f"/{keyframe.image_path}",
            "prior_text": prior.text if prior is not None else "",
            "completed_total": completed,
            "min_words": s.min_words,
            "min_chars": s.min_chars,
            "max_chars": s.max_chars,
            "session_target": s.session_target_frames,
            "edit_mode": False,
            "dataset_label": s.datasets[chosen].label,
            "chosen_dataset": chosen,
        },
    )
    # remember the choice if it came in via ?dataset=...
    if dataset and dataset in s.datasets and getattr(request.state, "chosen_dataset", "") != dataset:
        _set_dataset_cookie(response, dataset, secure=_request_is_https(request))
    return response


@router.get("/edit/{scene_id}/{frame_id}", response_class=HTMLResponse)
def edit_description(
    scene_id: str,
    frame_id: str,
    request: Request,
    db: Session = Depends(get_db),
    annotator: Annotator = Depends(get_annotator),
) -> HTMLResponse:
    s = get_settings()
    desc = db.get(Description, (scene_id, frame_id, annotator.id))
    if desc is None:
        raise HTTPException(404, "no description on this frame for you yet")
    keyframe = db.get(Keyframe, (scene_id, frame_id))
    scene = db.get(Scene, scene_id)
    if keyframe is None or scene is None:
        raise HTTPException(500, "scene/keyframe missing in db")

    lease_mod.acquire(db, scene_id, frame_id, annotator.id)

    return templates.TemplateResponse(
        request,
        "annotate.html",
        {
            "annotator": annotator,
            "scene": scene,
            "keyframe": keyframe,
            "image_url": f"/{keyframe.image_path}",
            "prior_text": desc.text,
            "completed_total": 0,
            "min_words": s.min_words,
            "min_chars": s.min_chars,
            "max_chars": s.max_chars,
            "session_target": s.session_target_frames,
            "edit_mode": True,
            "dataset_label": s.datasets.get(scene.dataset, s.datasets["3rscan"]).label,
            "chosen_dataset": getattr(request.state, "chosen_dataset", "") or None,
        },
    )


@router.get("/history", response_class=HTMLResponse)
def history(
    request: Request,
    db: Session = Depends(get_db),
    annotator: Annotator = Depends(get_annotator),
) -> HTMLResponse:
    s = get_settings()
    rows = list(
        db.execute(
            select(Description, Scene.dataset)
            .join(Scene, Scene.id == Description.scene_id)
            .where(Description.annotator_id == annotator.id)
            .order_by(sa_desc(Description.submitted_at))
        ).all()
    )
    items = []
    for d, ds_key in rows:
        edited = (d.edit_count or 0) > 0
        last_seen = d.edited_at if edited else d.submitted_at
        items.append({
            "kind": "description",
            "scene_id": d.scene_id,
            "scene_short": d.scene_id[:8],
            "frame_id": d.frame_id,
            "text": d.text,
            "word_count": d.word_count,
            "edited": edited,
            "relative_time": _relative_time(last_seen),
            "edit_url": f"/edit/{d.scene_id}/{d.frame_id}",
            "image_url": f"/keyframes/{ds_key}/{d.scene_id}/{d.frame_id}.jpg",
            "dataset": ds_key,
            "dataset_label": s.datasets.get(ds_key, s.datasets["3rscan"]).label,
            "sort_key": last_seen,
        })

    loc_rows = list(
        db.execute(
            select(HumanLocalization, Scene.dataset)
            .join(Scene, Scene.id == HumanLocalization.scene_id)
            .where(HumanLocalization.annotator_id == annotator.id)
            .order_by(sa_desc(HumanLocalization.submitted_at))
        ).all()
    )
    import math as _math
    for l, ds_key in loc_rows:
        edited = (l.edit_count or 0) > 0
        last_seen = l.edited_at if edited else l.submitted_at
        dist = l.distance_error or 0.0
        ang = l.angular_error_deg
        iou_err = l.iou_error
        iou_str = ""
        if iou_err is not None:
            iou_str = f", IoU {(1.0 - iou_err):.3f}"
        text = (
            f"Pos err {dist:.2f} m"
            + (f", angle err {ang:.1f}°" if ang is not None else "")
            + iou_str
            + f" — placed at ({l.pred_x:.2f}, {l.pred_y:.2f}) "
              f"heading {(l.pred_yaw * 180.0 / _math.pi):.0f}°"
        )
        items.append({
            "kind": "localization",
            "scene_id": l.scene_id,
            "scene_short": l.scene_id[:8],
            "frame_id": l.frame_id,
            "text": text,
            "word_count": 0,
            "edited": edited,
            "relative_time": _relative_time(last_seen),
            "edit_url": f"/localize/{l.scene_id}/{l.frame_id}",
            "image_url": f"/keyframes/{ds_key}/{l.scene_id}/{l.frame_id}.jpg",
            "dataset": ds_key,
            "dataset_label": s.datasets.get(ds_key, s.datasets["3rscan"]).label,
            "sort_key": last_seen,
        })

    # newest first, mixed
    items.sort(key=lambda x: x.get("sort_key") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    return templates.TemplateResponse(
        request,
        "history.html",
        {
            "annotator": annotator,
            "items": items,
            "chosen_dataset": getattr(request.state, "chosen_dataset", "") or None,
        },
    )


@router.get("/done", response_class=HTMLResponse)
def done(
    request: Request,
    dataset: Optional[str] = None,
    db: Session = Depends(get_db),
    annotator: Annotator = Depends(get_annotator),
) -> HTMLResponse:
    s = get_settings()
    chosen = dataset or getattr(request.state, "chosen_dataset", "") or None
    q = (
        select(func.count())
        .select_from(FrameCompletion)
        .where(FrameCompletion.annotator_id == annotator.id)
    )
    if chosen and chosen in s.datasets:
        q = q.join(Scene, Scene.id == FrameCompletion.scene_id).where(Scene.dataset == chosen)
    your_count = db.scalar(q) or 0
    return templates.TemplateResponse(
        request,
        "done.html",
        {
            "annotator": annotator,
            "your_completed": your_count,
            "chosen_dataset": chosen,
            "dataset_label": s.datasets[chosen].label if chosen and chosen in s.datasets else None,
        },
    )


# ---------------------------------------------------------------------------
# Human-as-localizer pages
# ---------------------------------------------------------------------------
@router.get("/localize", response_class=HTMLResponse)
def localize_assignment(
    request: Request,
    dataset: Optional[str] = None,
    scene: Optional[str] = None,
    db: Session = Depends(get_db),
    annotator: Annotator = Depends(get_annotator),
) -> HTMLResponse:
    """Pick a frame for this annotator to localize.

    If ``scene`` is given, restrict the pick to that scene (used by
    the scene-selector page). Otherwise the default policy (close
    partial scenes, then open the easiest fresh one) applies.
    """
    s = get_settings()
    chosen = dataset or getattr(request.state, "chosen_dataset", "")
    if not chosen or chosen not in s.datasets:
        return RedirectResponse(url="/datasets?next=localize", status_code=303)

    pick = loc_mod.assign_localization_frame(
        db, annotator.id, chosen, scene_id=scene
    )
    if pick is None:
        # If the user picked a specific scene that's fully done, send
        # them to the scene browser so they can pick another one.
        if scene is not None:
            return RedirectResponse(
                url=f"/localize/scenes?dataset={chosen}", status_code=303
            )
        return RedirectResponse(url=f"/done?dataset={chosen}", status_code=303)
    scene_id, frame_id = pick
    return _render_localize(
        request, db, annotator, scene_id, frame_id, edit_mode=False
    )


@router.get("/localize/scenes", response_class=HTMLResponse)
def localize_scene_browser(
    request: Request,
    dataset: Optional[str] = None,
    db: Session = Depends(get_db),
    annotator: Annotator = Depends(get_annotator),
) -> HTMLResponse:
    """Browse all scenes with descriptions; show per-scene
    localization progress so the annotator can pick one to work on.
    """
    s = get_settings()
    chosen = dataset or getattr(request.state, "chosen_dataset", "")
    if not chosen or chosen not in s.datasets:
        return RedirectResponse(url="/datasets?next=localize", status_code=303)

    # Frames-described per scene (eligible pool size)
    frames_desc_q = (
        select(
            Description.scene_id,
            func.count(func.distinct(Description.frame_id)).label("n_desc"),
        )
        .group_by(Description.scene_id)
    )
    # Frames already localized per scene (any annotator → strict cap)
    frames_loc_q = (
        select(
            HumanLocalization.scene_id,
            func.count(func.distinct(HumanLocalization.frame_id)).label("n_loc"),
        )
        .group_by(HumanLocalization.scene_id)
    )
    # My localized frames per scene (for display: "you've done N here")
    mine_q = (
        select(
            HumanLocalization.scene_id,
            func.count(func.distinct(HumanLocalization.frame_id)).label("n_mine"),
        )
        .where(HumanLocalization.annotator_id == annotator.id)
        .group_by(HumanLocalization.scene_id)
    )

    frames_desc = {r[0]: r[1] for r in db.execute(frames_desc_q).all()}
    frames_loc = {r[0]: r[1] for r in db.execute(frames_loc_q).all()}
    mine = {r[0]: r[1] for r in db.execute(mine_q).all()}

    scenes = list(
        db.scalars(
            select(Scene)
            .where(Scene.dataset == chosen)
            .where(Scene.id.in_(frames_desc.keys()))
            .order_by(Scene.difficulty_rank)
        ).all()
    )

    items = []
    for sc in scenes:
        n_desc = frames_desc.get(sc.id, 0)
        n_loc = frames_loc.get(sc.id, 0)
        n_mine = mine.get(sc.id, 0)
        items.append({
            "scene_id": sc.id,
            "scene_short": sc.id[:14] if len(sc.id) > 14 else sc.id,
            "rank": sc.difficulty_rank,
            "is_anchor": (chosen == "scannet" and sc.difficulty_tertile == 0
                          and sc.difficulty_rank <= 10),
            "n_desc": n_desc,
            "n_loc": n_loc,
            "n_remaining": max(0, n_desc - n_loc),
            "n_mine": n_mine,
            "pct": (100.0 * n_loc / n_desc) if n_desc else 0.0,
            "fully_done": (n_desc > 0 and n_loc >= n_desc),
        })

    return templates.TemplateResponse(
        request,
        "localize_scenes.html",
        {
            "annotator": annotator,
            "chosen_dataset": chosen,
            "dataset_label": s.datasets[chosen].label,
            "items": items,
        },
    )


@router.get("/localize/{scene_id}/{frame_id}", response_class=HTMLResponse)
def localize_edit(
    scene_id: str,
    frame_id: str,
    request: Request,
    db: Session = Depends(get_db),
    annotator: Annotator = Depends(get_annotator),
) -> HTMLResponse:
    """Edit-mode: re-open a localization the annotator already
    submitted so they can refine it."""
    existing = db.get(HumanLocalization, (scene_id, frame_id, annotator.id))
    if existing is None:
        raise HTTPException(404, "no previous localization to edit")
    return _render_localize(
        request, db, annotator, scene_id, frame_id, edit_mode=True, existing=existing
    )


def _render_localize(
    request: Request,
    db: Session,
    annotator: Annotator,
    scene_id: str,
    frame_id: str,
    edit_mode: bool,
    existing: Optional[HumanLocalization] = None,
) -> HTMLResponse:
    s = get_settings()
    scene = db.get(Scene, scene_id)
    if scene is None:
        raise HTTPException(404, f"unknown scene {scene_id}")

    prompt = loc_mod.pick_prompt_description(db, scene_id, frame_id, annotator.id)
    if prompt is None:
        raise HTTPException(
            404, f"no description on file for {scene_id}/{frame_id} — cannot localize"
        )

    # mesh URL + per-dataset evaluation FoV (matches paper supp Tab. 7 /
    # langloc/configs/localization/{scannet,3rscan}.yaml). The localizer
    # UI MUST present the same horizontal/vertical FoV as the GT camera
    # so the human is solving the same perceptual problem the downstream
    # localizer scores against.
    if scene.dataset == "scannet":
        mesh_url = f"/meshes/scannet/{scene_id}.ply"
        h_fov_deg, v_fov_deg = 58.30, 45.33
    elif scene.dataset == "3rscan":
        mesh_url = f"/meshes/3rscan/{scene_id}.ply"
        h_fov_deg, v_fov_deg = 39.31, 64.76
    else:
        raise HTTPException(500, f"unsupported dataset {scene.dataset}")

    return templates.TemplateResponse(
        request,
        "localize.html",
        {
            "annotator": annotator,
            "scene_id": scene_id,
            "frame_id": frame_id,
            "dataset": scene.dataset,
            "dataset_label": s.datasets[scene.dataset].label,
            "prompt_text": prompt.text,
            "prompt_annotator_id": prompt.annotator_id,
            "mesh_url": mesh_url,
            "eye_height_m": 1.6,
            "h_fov_deg": h_fov_deg,
            "v_fov_deg": v_fov_deg,
            "edit_mode": edit_mode,
            "existing_pose": {
                "x": existing.pred_x,
                "y": existing.pred_y,
                "z": existing.pred_z,
                "yaw": existing.pred_yaw,
            } if existing is not None else None,
            "existing_errors": {
                "distance_error": existing.distance_error,
                "angular_error_deg": existing.angular_error_deg,
            } if existing is not None else None,
        },
    )
