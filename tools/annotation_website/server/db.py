"""SQLite engine + session factory with the right PRAGMAs.

WAL mode + a 5 s busy timeout + foreign keys are non-negotiable here:
without them concurrent auto-saves from two annotators will randomly fail
under SQLite's default rollback-journal serialisation.

Every transaction is started with ``BEGIN IMMEDIATE`` so that the writer
lock is acquired at the start of the transaction, not when the first
``INSERT`` runs.  This makes the SELECT-then-INSERT race in lease
acquisition impossible without losing throughput at our scale (<= 5
concurrent users).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _set_pragmas(dbapi_connection, _connection_record) -> None:
    cur = dbapi_connection.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=5000")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine
    s = get_settings()
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{s.db_path}"
    _engine = create_engine(
        url,
        connect_args={"check_same_thread": False},
        future=True,
    )
    event.listen(_engine, "connect", _set_pragmas)

    @event.listens_for(_engine, "begin")
    def _begin_immediate(conn):
        conn.exec_driver_sql("BEGIN IMMEDIATE")

    _SessionLocal = sessionmaker(
        bind=_engine, autoflush=False, autocommit=False, future=True
    )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a session; commit on success, roll back on exception."""
    factory = get_session_factory()
    sess = factory()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def init_db() -> None:
    """Create tables if they do not exist. Idempotent."""
    from . import models  # noqa: F401

    engine = get_engine()
    models.Base.metadata.create_all(engine)
