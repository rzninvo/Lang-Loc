"""FastAPI dependency providers + cookie middleware.

The cookie middleware runs on every request: it reads the signed
annotator-id cookie, materialises a fresh UUID if missing or tampered
with, and stashes the id on ``request.state.annotator_id``. After the
route runs, it sets/refreshes the cookie on the outgoing response.

Setting the cookie via ``response.set_cookie`` on a ``Depends(Response)``
object does NOT work when the route returns a custom Response such as
``TemplateResponse`` or ``RedirectResponse``: FastAPI uses the route's
response and discards the dependency's response. Middleware is the
correct layer for this.
"""
from __future__ import annotations

import uuid
from typing import Iterator

from fastapi import Depends, Request
from itsdangerous import BadSignature, URLSafeSerializer
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from . import db as db_module
from .config import get_settings
from .models import Annotator


def _serializer() -> URLSafeSerializer:
    return URLSafeSerializer(get_settings().cookie_secret, salt="annotator-id")


class AnnotatorCookieMiddleware(BaseHTTPMiddleware):
    """Cookie-based annotator identity for every request."""

    async def dispatch(self, request: Request, call_next):
        s = get_settings()
        cookie = request.cookies.get(s.cookie_name)
        annotator_id = None
        if cookie:
            try:
                annotator_id = _serializer().loads(cookie)
            except BadSignature:
                annotator_id = None
        is_new = annotator_id is None
        if is_new:
            annotator_id = str(uuid.uuid4())
        request.state.annotator_id = annotator_id

        # the dataset cookie is unsigned; it just records which dataset
        # the user picked on /datasets so /annotate can pull from the right
        # pool without a query param
        chosen = request.cookies.get(s.cookie_dataset_name, "")
        if chosen not in s.datasets:
            chosen = ""
        request.state.chosen_dataset = chosen

        response: Response = await call_next(request)

        # When the request reached us over HTTPS (directly or via a
        # trusted proxy that sets X-Forwarded-Proto), tag the cookie
        # `Secure` so it isn't sent over HTTP back to a stale connection.
        is_https = (
            request.url.scheme == "https"
            or request.headers.get("x-forwarded-proto", "").lower() == "https"
        )
        # always re-issue the cookie so its expiry rolls forward
        response.set_cookie(
            s.cookie_name,
            _serializer().dumps(annotator_id),
            max_age=s.cookie_max_age_days * 86400,
            httponly=True,
            samesite="lax",
            secure=is_https,
        )
        return response


def get_db() -> Iterator[Session]:
    factory = db_module.get_session_factory()
    sess = factory()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def get_annotator(
    request: Request,
    db: Session = Depends(get_db),
) -> Annotator:
    """Resolve the current annotator from middleware-set state."""
    annotator_id = getattr(request.state, "annotator_id", None)
    if annotator_id is None:
        # belt-and-braces: middleware should always have set this
        annotator_id = str(uuid.uuid4())
    annotator = db.get(Annotator, annotator_id)
    if annotator is None:
        annotator = Annotator(id=annotator_id, nickname=None)
        db.add(annotator)
        db.flush()
    return annotator
