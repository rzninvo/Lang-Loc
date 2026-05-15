"""Server-side description validation.

Returns a tuple ``(ok, reason, flagged)``:
    ``ok``      reject if False (HTTP 400)
    ``flagged`` accept but mark for author review if True
    ``reason``  short string describing why
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional, Tuple

from .config import get_settings


_WORD_RE = re.compile(r"[A-Za-z']+")


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _repeated_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    counter = Counter(text.replace(" ", "").lower())
    if not counter:
        return 0.0
    most = counter.most_common(1)[0][1]
    return most / sum(counter.values())


def _jaccard_token(a: str, b: str) -> float:
    ta = set(_WORD_RE.findall(a.lower()))
    tb = set(_WORD_RE.findall(b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def validate_description(
    text: str,
    duration_ms: int,
    annotator_recent_descriptions: Iterable[str] = (),
) -> Tuple[bool, Optional[str], bool]:
    s = get_settings()
    text = (text or "").strip()
    n = len(text)
    if n < s.min_chars:
        return False, f"description must be at least {s.min_chars} characters", False
    if n > s.max_chars:
        return False, f"description must be under {s.max_chars} characters", False

    wc = word_count(text)
    if s.min_words > 0 and wc < s.min_words:
        return False, f"description must be at least {s.min_words} words", False

    if _repeated_char_ratio(text) > 0.5:
        return False, "description contains too many repeated characters", False

    flagged = False
    reason: Optional[str] = None

    if duration_ms < s.min_seconds_per_frame * 1000:
        flagged = True
        reason = "very short time on task"

    for prev in annotator_recent_descriptions:
        if _jaccard_token(text, prev) > 0.7:
            flagged = True
            reason = "high overlap with the same annotator's previous description"
            break

    return True, reason, flagged
