"""ORM models for the annotation website.

The data model is small enough to fit in one file. The natural keys are:
- ``annotators.id``: a per-browser UUID stored in a signed cookie.
- ``scenes.id``: 3RScan scene UUID (matches ``data/3RScan/<id>``).
- ``keyframes(scene_id, frame_id)``: e.g. ``frame-000042``.
- ``leases(scene_id, frame_id)``: at most one live lease per keyframe.
- ``frame_completions(scene_id, frame_id, annotator_id)``: prevents
  showing the same frame to the same annotator twice.
- ``descriptions(scene_id, frame_id, annotator_id)``: the research output.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Float,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    DateTime,
    Index,
)
from sqlalchemy.orm import declarative_base


Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class Annotator(Base):
    __tablename__ = "annotators"

    id = Column(String, primary_key=True)
    nickname = Column(String, nullable=True)
    pin_hash = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)


class Scene(Base):
    __tablename__ = "scenes"

    id = Column(String, primary_key=True)
    dataset = Column(String, nullable=False, default="3rscan")
    display_index = Column(Integer, nullable=False)
    difficulty_tertile = Column(Integer, nullable=False, default=1)
    difficulty_rank = Column(Integer, nullable=False, default=999)  # lower = easier; assignment fills fresh scenes in this order
    num_frames = Column(Integer, nullable=False)
    __table_args__ = (
        Index("ix_scenes_dataset_rank", "dataset", "difficulty_rank"),
    )


class Keyframe(Base):
    __tablename__ = "keyframes"

    scene_id = Column(String, primary_key=True)
    frame_id = Column(String, primary_key=True)
    image_path = Column(String, nullable=False)
    __table_args__ = (
        ForeignKeyConstraint(["scene_id"], ["scenes.id"]),
        Index("ix_keyframes_scene", "scene_id"),
    )


class Lease(Base):
    __tablename__ = "leases"

    scene_id = Column(String, primary_key=True)
    frame_id = Column(String, primary_key=True)
    annotator_id = Column(String, nullable=False)
    acquired_at = Column(DateTime, default=_utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(["scene_id", "frame_id"], ["keyframes.scene_id", "keyframes.frame_id"]),
        ForeignKeyConstraint(["annotator_id"], ["annotators.id"]),
        Index("ix_leases_expires_at", "expires_at"),
        Index("ix_leases_annotator", "annotator_id"),
    )


class FrameCompletion(Base):
    __tablename__ = "frame_completions"

    scene_id = Column(String, primary_key=True)
    frame_id = Column(String, primary_key=True)
    annotator_id = Column(String, primary_key=True)
    completed_at = Column(DateTime, default=_utcnow, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(["scene_id", "frame_id"], ["keyframes.scene_id", "keyframes.frame_id"]),
        ForeignKeyConstraint(["annotator_id"], ["annotators.id"]),
        Index("ix_completions_annotator", "annotator_id"),
        Index("ix_completions_frame", "scene_id", "frame_id"),
    )


class Description(Base):
    __tablename__ = "descriptions"

    scene_id = Column(String, primary_key=True)
    frame_id = Column(String, primary_key=True)
    annotator_id = Column(String, primary_key=True)
    text = Column(Text, nullable=False)
    word_count = Column(Integer, nullable=False, default=0)
    duration_ms = Column(Integer, nullable=False, default=0)
    flagged = Column(Integer, nullable=False, default=0)
    flag_reason = Column(String, nullable=True)
    submitted_at = Column(DateTime, default=_utcnow, nullable=False)
    edited_at = Column(DateTime, nullable=True)
    edit_count = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        ForeignKeyConstraint(["scene_id", "frame_id"], ["keyframes.scene_id", "keyframes.frame_id"]),
        ForeignKeyConstraint(["annotator_id"], ["annotators.id"]),
        Index("ix_descriptions_annotator", "annotator_id"),
        Index("ix_descriptions_frame", "scene_id", "frame_id"),
    )


class HumanLocalization(Base):
    """Human-as-localizer submissions.

    A localizer reads a description (someone else's, or possibly their
    own) and places a first-person camera in the scene mesh at the
    position + heading they think the description was written from.
    The natural key is the same shape as ``descriptions``:
    ``(scene_id, frame_id, annotator_id)`` — at most one localization
    per annotator per keyframe.

    The "ground truth" we compare against is the keyframe's stored
    ``scene_pose`` (camera-to-world 4x4 from the GT recorder). The
    server computes ``distance_error`` and ``angular_error_deg`` at
    submit time and stores them so we don't have to re-compute on
    every export.

    Eye height is held fixed at 1.6 m by the front end — ``pred_z`` is
    recorded for sanity but should be ≈ 1.6 m at all times.
    """
    __tablename__ = "human_localizations"

    scene_id = Column(String, primary_key=True)
    frame_id = Column(String, primary_key=True)
    annotator_id = Column(String, primary_key=True)

    # predicted camera centre (in scene mesh coords; metres)
    pred_x = Column(Float, nullable=False)
    pred_y = Column(Float, nullable=False)
    pred_z = Column(Float, nullable=False)
    # predicted yaw (heading around vertical axis); radians
    pred_yaw = Column(Float, nullable=False)

    # which description was the annotator looking at? (NULL if they
    # localized from the image — currently unused but reserved)
    prompt_annotator_id = Column(String, nullable=True)

    # computed at submit-time, stored to avoid recomputing
    distance_error = Column(Float, nullable=True)
    angular_error_deg = Column(Float, nullable=True)
    iou_error = Column(Float, nullable=True)  # 1 − 3-D View IoU

    duration_ms = Column(Integer, nullable=False, default=0)
    flagged = Column(Integer, nullable=False, default=0)
    flag_reason = Column(String, nullable=True)
    submitted_at = Column(DateTime, default=_utcnow, nullable=False)
    edited_at = Column(DateTime, nullable=True)
    edit_count = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        ForeignKeyConstraint(["scene_id", "frame_id"], ["keyframes.scene_id", "keyframes.frame_id"]),
        ForeignKeyConstraint(["annotator_id"], ["annotators.id"]),
        Index("ix_human_localizations_annotator", "annotator_id"),
        Index("ix_human_localizations_frame", "scene_id", "frame_id"),
    )


class LocalizationSkip(Base):
    """A per-annotator record that this (scene, frame) was offered and
    skipped. Used by the assignment policy to avoid re-handing the same
    skipped frame back. Separate from ``HumanLocalization`` so we don't
    pollute the metrics table with sentinel/empty rows.
    """
    __tablename__ = "localization_skips"

    scene_id = Column(String, primary_key=True)
    frame_id = Column(String, primary_key=True)
    annotator_id = Column(String, primary_key=True)
    skipped_at = Column(DateTime, default=_utcnow, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(["scene_id", "frame_id"], ["keyframes.scene_id", "keyframes.frame_id"]),
        ForeignKeyConstraint(["annotator_id"], ["annotators.id"]),
        Index("ix_localization_skips_annotator", "annotator_id"),
    )
