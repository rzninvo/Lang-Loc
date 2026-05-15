"""JSON API routes called by the annotate page's JavaScript.

The annotate page does:
  - PUT  /api/save   on every textarea blur or every 5s of typing
  - POST /api/submit when the user clicks "Next frame"

Both endpoints check that the calling annotator currently holds the
lease on that (scene, frame); a stale tab whose lease has expired gets
a 409 and is redirected to ``/annotate`` to be reassigned.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import desc as sa_desc, select
from sqlalchemy.orm import Session

from .. import lease as lease_mod
from .. import localization as loc_mod
from ..config import get_settings
from ..deps import get_annotator, get_db
from ..models import Annotator, Description, FrameCompletion, HumanLocalization, Keyframe, LocalizationSkip, Scene
from ..validation import validate_description, word_count


router = APIRouter(prefix="/api")


class SavePayload(BaseModel):
    scene_id: str = Field(min_length=1)
    frame_id: str = Field(min_length=1)
    text: str = ""
    duration_ms: int = 0


class SubmitPayload(SavePayload):
    pass


def _ensure_lease(
    db: Session, scene_id: str, frame_id: str, annotator_id: str
) -> None:
    held = db.get_one  # type: ignore[attr-defined]
    from ..models import Lease

    lease = db.get(Lease, (scene_id, frame_id))
    if lease is None or lease.annotator_id != annotator_id:
        raise HTTPException(409, "lease lost or never acquired; please reload /annotate")


def _append_jsonl(record: dict) -> None:
    s = get_settings()
    s.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with s.jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


@router.put("/save")
def api_save(
    payload: SavePayload,
    db: Session = Depends(get_db),
    annotator: Annotator = Depends(get_annotator),
) -> dict:
    _ensure_lease(db, payload.scene_id, payload.frame_id, annotator.id)
    if db.get(Keyframe, (payload.scene_id, payload.frame_id)) is None:
        raise HTTPException(404, "unknown keyframe")

    text = (payload.text or "").strip()
    wc = word_count(text)
    existing = db.get(Description, (payload.scene_id, payload.frame_id, annotator.id))
    if existing is None:
        existing = Description(
            scene_id=payload.scene_id,
            frame_id=payload.frame_id,
            annotator_id=annotator.id,
            text=text,
            word_count=wc,
            duration_ms=payload.duration_ms,
        )
        db.add(existing)
    else:
        existing.text = text
        existing.word_count = wc
        existing.duration_ms = payload.duration_ms

    lease_mod.renew(db, payload.scene_id, payload.frame_id, annotator.id)
    return {"saved": True, "word_count": wc}


@router.post("/submit")
def api_submit(
    payload: SubmitPayload,
    db: Session = Depends(get_db),
    annotator: Annotator = Depends(get_annotator),
) -> dict:
    _ensure_lease(db, payload.scene_id, payload.frame_id, annotator.id)
    if db.get(Keyframe, (payload.scene_id, payload.frame_id)) is None:
        raise HTTPException(404, "unknown keyframe")

    text = (payload.text or "").strip()
    # exclude the current (scene, frame) draft from the duplicate-overlap
    # check, otherwise the auto-saved draft will fail the Jaccard test
    # against itself when the user clicks submit.
    recent = list(
        db.scalars(
            select(Description.text)
            .where(
                Description.annotator_id == annotator.id,
                ~((Description.scene_id == payload.scene_id)
                  & (Description.frame_id == payload.frame_id)),
            )
            .order_by(sa_desc(Description.submitted_at))
            .limit(3)
        )
    )
    ok, reason, flagged = validate_description(text, payload.duration_ms, recent)
    if not ok:
        raise HTTPException(400, reason or "invalid description")

    wc = word_count(text)
    existing = db.get(Description, (payload.scene_id, payload.frame_id, annotator.id))
    submitted_at = datetime.now(tz=timezone.utc)
    if existing is None:
        existing = Description(
            scene_id=payload.scene_id,
            frame_id=payload.frame_id,
            annotator_id=annotator.id,
            text=text,
            word_count=wc,
            duration_ms=payload.duration_ms,
            flagged=int(flagged),
            flag_reason=reason,
            submitted_at=submitted_at,
        )
        db.add(existing)
    else:
        existing.text = text
        existing.word_count = wc
        existing.duration_ms = payload.duration_ms
        existing.flagged = int(flagged)
        existing.flag_reason = reason
        existing.submitted_at = submitted_at

    completion = db.get(
        FrameCompletion,
        (payload.scene_id, payload.frame_id, annotator.id),
    )
    if completion is None:
        db.add(
            FrameCompletion(
                scene_id=payload.scene_id,
                frame_id=payload.frame_id,
                annotator_id=annotator.id,
                completed_at=submitted_at,
            )
        )

    lease_mod.release(db, payload.scene_id, payload.frame_id, annotator.id)

    _append_jsonl(
        {
            "scene_id": payload.scene_id,
            "frame_id": payload.frame_id,
            "annotator_id": annotator.id,
            "nickname": annotator.nickname,
            "text": text,
            "word_count": wc,
            "duration_ms": payload.duration_ms,
            "flagged": bool(flagged),
            "flag_reason": reason,
            "submitted_at": submitted_at.isoformat(),
            "kind": "submit",
        }
    )

    return {"submitted": True, "flagged": flagged, "reason": reason}


@router.post("/edit")
def api_edit(
    payload: SubmitPayload,
    db: Session = Depends(get_db),
    annotator: Annotator = Depends(get_annotator),
) -> dict:
    """Update an existing description in place. Bumps ``edited_at``."""
    _ensure_lease(db, payload.scene_id, payload.frame_id, annotator.id)
    if db.get(Keyframe, (payload.scene_id, payload.frame_id)) is None:
        raise HTTPException(404, "unknown keyframe")

    desc = db.get(Description, (payload.scene_id, payload.frame_id, annotator.id))
    if desc is None:
        raise HTTPException(404, "no description to edit; use /api/submit first")

    text = (payload.text or "").strip()
    recent = list(
        db.scalars(
            select(Description.text)
            .where(
                Description.annotator_id == annotator.id,
                ~((Description.scene_id == payload.scene_id)
                  & (Description.frame_id == payload.frame_id)),
            )
            .order_by(sa_desc(Description.submitted_at))
            .limit(3)
        )
    )
    ok, reason, flagged = validate_description(text, payload.duration_ms, recent)
    if not ok:
        raise HTTPException(400, reason or "invalid description")

    edited_at = datetime.now(tz=timezone.utc)
    desc.text = text
    desc.word_count = word_count(text)
    desc.duration_ms = max(int(desc.duration_ms or 0), int(payload.duration_ms or 0))
    desc.flagged = int(flagged)
    desc.flag_reason = reason
    desc.edited_at = edited_at
    desc.edit_count = int(desc.edit_count or 0) + 1

    lease_mod.release(db, payload.scene_id, payload.frame_id, annotator.id)

    _append_jsonl(
        {
            "scene_id": payload.scene_id,
            "frame_id": payload.frame_id,
            "annotator_id": annotator.id,
            "nickname": annotator.nickname,
            "text": text,
            "word_count": desc.word_count,
            "duration_ms": payload.duration_ms,
            "flagged": bool(flagged),
            "flag_reason": reason,
            "submitted_at": edited_at.isoformat(),
            "kind": "edit",
            "edit_count": desc.edit_count,
        }
    )

    return {"edited": True, "flagged": flagged, "reason": reason, "edit_count": desc.edit_count}


# ---------------------------------------------------------------------------
# Human-as-localizer endpoints
# ---------------------------------------------------------------------------
class LocalizeSubmitPayload(BaseModel):
    scene_id: str = Field(min_length=1)
    frame_id: str = Field(min_length=1)
    pred_x: float
    pred_y: float
    pred_z: float
    pred_yaw: float
    duration_ms: int = 0
    prompt_annotator_id: Optional[str] = None


@router.post("/localize/submit")
def api_localize_submit(
    payload: LocalizeSubmitPayload,
    db: Session = Depends(get_db),
    annotator: Annotator = Depends(get_annotator),
) -> dict:
    """Record a human-as-localizer prediction.

    Server reads the GT pose from disk and computes
    ``distance_error`` + ``angular_error_deg``. The client never gets
    the GT pose, only its own error after submitting.
    """
    if db.get(Keyframe, (payload.scene_id, payload.frame_id)) is None:
        raise HTTPException(404, "unknown keyframe")

    s = get_settings()
    scene_row = db.get(Scene, payload.scene_id)
    if scene_row is None:
        raise HTTPException(404, "unknown scene")
    data_root = s.dataset_roots.get(scene_row.dataset)
    if data_root is None:
        raise HTTPException(500, f"no dataset root configured for {scene_row.dataset}")

    gt_pose = loc_mod.gt_pose_for(payload.scene_id, payload.frame_id, data_root)
    if gt_pose is None:
        raise HTTPException(404, "no GT pose on disk for this keyframe")

    dist_err, ang_err = loc_mod.compute_pose_errors(
        gt_pose, payload.pred_x, payload.pred_y, payload.pred_z, payload.pred_yaw
    )
    # 3-D View IoU at paper FoV (None if Open3D unavailable or mesh missing)
    iou = loc_mod.compute_view_iou(
        scene_row.dataset, payload.scene_id, data_root,
        gt_pose, payload.pred_x, payload.pred_y, payload.pred_z, payload.pred_yaw,
    )
    iou_err = (1.0 - iou) if iou is not None else None

    # very-short-task flag (sanity check; same heuristic as descriptions)
    flagged = 0
    reason: Optional[str] = None
    if payload.duration_ms < 5_000:
        flagged = 1
        reason = "very short time on task"

    submitted_at = datetime.now(tz=timezone.utc)
    existing = db.get(
        HumanLocalization, (payload.scene_id, payload.frame_id, annotator.id)
    )
    if existing is None:
        db.add(
            HumanLocalization(
                scene_id=payload.scene_id,
                frame_id=payload.frame_id,
                annotator_id=annotator.id,
                pred_x=payload.pred_x,
                pred_y=payload.pred_y,
                pred_z=payload.pred_z,
                pred_yaw=payload.pred_yaw,
                prompt_annotator_id=payload.prompt_annotator_id,
                distance_error=dist_err,
                angular_error_deg=ang_err,
                iou_error=iou_err,
                duration_ms=payload.duration_ms,
                flagged=flagged,
                flag_reason=reason,
                submitted_at=submitted_at,
            )
        )
    else:
        # edit-mode: recompute all metrics with the new pose
        existing.pred_x = payload.pred_x
        existing.pred_y = payload.pred_y
        existing.pred_z = payload.pred_z
        existing.pred_yaw = payload.pred_yaw
        existing.distance_error = dist_err
        existing.angular_error_deg = ang_err
        existing.iou_error = iou_err
        existing.duration_ms = payload.duration_ms
        existing.flagged = flagged
        existing.flag_reason = reason
        existing.edited_at = submitted_at
        existing.edit_count = int(existing.edit_count or 0) + 1

    _append_jsonl(
        {
            "kind": "localize_submit",
            "scene_id": payload.scene_id,
            "frame_id": payload.frame_id,
            "annotator_id": annotator.id,
            "pred_x": payload.pred_x,
            "pred_y": payload.pred_y,
            "pred_z": payload.pred_z,
            "pred_yaw": payload.pred_yaw,
            "distance_error": dist_err,
            "angular_error_deg": ang_err,
            "iou_error": iou_err,
            "duration_ms": payload.duration_ms,
            "submitted_at": submitted_at.isoformat(),
            "flagged": bool(flagged),
            "flag_reason": reason,
        }
    )

    return {
        "submitted": True,
        "distance_error": dist_err,
        "angular_error_deg": ang_err,
        "iou_error": iou_err,
        "iou": (1.0 - iou_err) if iou_err is not None else None,
        "flagged": bool(flagged),
        "reason": reason,
    }


class LocalizeSkipPayload(BaseModel):
    scene_id: str = Field(min_length=1)
    frame_id: str = Field(min_length=1)


@router.post("/localize/skip")
def api_localize_skip(
    payload: LocalizeSkipPayload,
    db: Session = Depends(get_db),
    annotator: Annotator = Depends(get_annotator),
) -> dict:
    """Record that this annotator skipped (scene, frame) on the
    localize page. The assignment policy will not re-hand them back
    this exact frame on subsequent requests."""
    if db.get(Keyframe, (payload.scene_id, payload.frame_id)) is None:
        raise HTTPException(404, "unknown keyframe")
    existing = db.get(
        LocalizationSkip, (payload.scene_id, payload.frame_id, annotator.id)
    )
    if existing is None:
        db.add(
            LocalizationSkip(
                scene_id=payload.scene_id,
                frame_id=payload.frame_id,
                annotator_id=annotator.id,
            )
        )
    _append_jsonl(
        {
            "kind": "localize_skip",
            "scene_id": payload.scene_id,
            "frame_id": payload.frame_id,
            "annotator_id": annotator.id,
            "skipped_at": datetime.now(tz=timezone.utc).isoformat(),
        }
    )
    return {"skipped": True}
