"""Lease acquisition, renewal, and release.

A lease reserves a (scene_id, frame_id) keyframe to one annotator for
``lease_ttl_minutes``. Auto-renewed on every save so the user can spend
longer than the TTL on a single frame without losing their work.

Race-safety: every transaction uses ``BEGIN IMMEDIATE`` (configured in
``db.py``), so a SELECT-then-INSERT sequence holds the writer lock from
the start. Lease creation uses the natural primary key
(scene_id, frame_id) so a concurrent second writer who picked the same
frame loses the race with ``IntegrityError``; the caller retries with a
different frame.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import get_settings
from .models import Lease


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _expiry() -> datetime:
    return _now() + timedelta(minutes=get_settings().lease_ttl_minutes)


def cleanup_expired(session: Session) -> int:
    """Delete leases past their TTL. Returns the number deleted."""
    res = session.execute(
        delete(Lease).where(Lease.expires_at < _now())
    )
    return res.rowcount or 0


def acquire(session: Session, scene_id: str, frame_id: str, annotator_id: str) -> bool:
    """Try to acquire a fresh lease. Returns True on success.

    Returns False if another live lease already exists for this keyframe
    held by a different annotator.  If the same annotator already holds
    the lease, this is a no-op and returns True.
    """
    cleanup_expired(session)
    existing = session.get(Lease, (scene_id, frame_id))
    if existing is not None:
        if existing.annotator_id == annotator_id:
            existing.expires_at = _expiry()
            return True
        return False

    lease = Lease(
        scene_id=scene_id,
        frame_id=frame_id,
        annotator_id=annotator_id,
        acquired_at=_now(),
        expires_at=_expiry(),
    )
    session.add(lease)
    try:
        session.flush()
    except IntegrityError:
        session.rollback()
        return False
    return True


def renew(session: Session, scene_id: str, frame_id: str, annotator_id: str) -> bool:
    """Bump the expiry on the lease for this annotator. Returns True if found."""
    res = session.execute(
        update(Lease)
        .where(
            Lease.scene_id == scene_id,
            Lease.frame_id == frame_id,
            Lease.annotator_id == annotator_id,
        )
        .values(expires_at=_expiry())
    )
    return (res.rowcount or 0) > 0


def release(session: Session, scene_id: str, frame_id: str, annotator_id: str) -> bool:
    """Release the lease held by this annotator. Returns True if removed."""
    res = session.execute(
        delete(Lease).where(
            Lease.scene_id == scene_id,
            Lease.frame_id == frame_id,
            Lease.annotator_id == annotator_id,
        )
    )
    return (res.rowcount or 0) > 0


def held_by(session: Session, annotator_id: str) -> Optional[Lease]:
    """Return the live lease this annotator currently holds, if any."""
    cleanup_expired(session)
    return session.scalars(
        select(Lease).where(Lease.annotator_id == annotator_id)
    ).first()
