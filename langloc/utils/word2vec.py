"""Word2Vec label embeddings for fine-localization matching (paper §3.3).

Implements the matcher-side label embedder used by mk5's
``visualize_loc_prob.py::topk_matched_objects`` when
``homogenize_label_embeddings=True``.  This is the embedding source the
LangLoc paper's published Table 4 numbers come from — see
``docs/3307_LangLoc_Tell_Me_What_You_.pdf`` §3.3 ("We embed each node
label and relationship predicate into ℝᵈ using **Word2Vec [Mikolov et
al., 2013]**.") and Master Report Fig. 5.2 caption.

The embedding is a length-300 ℓ₂-normalised blend of:

- A spaCy ``en_core_web_lg`` token vector (300D GloVe), and
- A deterministic hashed n-gram lexical embedding (n ∈ {3, 4, 5})
  used as a robust out-of-vocabulary fallback.

Blend weight ``W2V_BLEND = 0.8`` (i.e. 80% Word2Vec + 20% lexical),
matching mk5's ``_MATCH_W2V_BLEND`` constant.

Line-by-line port of
``whereami-text2sgm/playground/graph_models/models/
visualize_loc_prob.py:181-256``.

The first call lazily loads spaCy ``en_core_web_lg``; subsequent calls
are O(1) cache lookups via ``label_embedding_for_matching``.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional

import numpy as np


EMBED_DIM = 300
W2V_BLEND = 0.8

_MATCH_W2V_CACHE: dict[str, np.ndarray] = {}
_MATCH_EMBED_CACHE: dict[str, np.ndarray] = {}
_NLP = None  # spaCy pipeline, lazy-loaded.


def _load_nlp():
    """Lazily load spaCy ``en_core_web_lg``.  Cached on first call."""
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_lg")
    return _NLP


def canonical_label(label: str) -> str:
    """Light canonicalization for label identity matching.

    Mirrors mk5's ``visualize_loc_prob.canonical_label`` (lines 181-188):
    lowercase, strip apostrophe-s, replace ``_-`` with spaces, drop
    non-alphanumeric, collapse whitespace.

    Args:
        label: Raw object label (e.g. ``"Wall_north_upper"``).

    Returns:
        Canonical form (e.g. ``"wall north upper"``).
    """
    text = str(label).strip().lower()
    text = re.sub(r"\b([a-z0-9]+)'s\b", r"\1", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a vector; return zeros if the norm is degenerate."""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    nrm = float(np.linalg.norm(arr))
    if not np.isfinite(nrm) or nrm < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / nrm).astype(np.float32)


def _stable_hash32(text: str) -> int:
    """Deterministic 32-bit Blake2b hash of *text*.

    Matches mk5's ``_stable_hash32`` byte-for-byte so the lexical n-gram
    fallback produces identical bucket assignments.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _hashed_label_embedding(key: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic lexical embedding (n-gram hashed) used for OOV labels.

    Builds a sparse signed-bag-of-character-n-grams over n ∈ {3, 4, 5}
    with a leading ``^`` and trailing ``$`` boundary marker, then
    L2-normalises.  A line-by-line port of mk5's
    ``_hashed_label_embedding`` (lines 210-225).
    """
    out = np.zeros(dim, dtype=np.float32)
    if not key:
        return out
    wrapped = f"^{key}$"
    for n in (3, 4, 5):
        if len(wrapped) < n:
            continue
        for i in range(len(wrapped) - n + 1):
            gram = wrapped[i:i + n]
            h = _stable_hash32(gram)
            out[h % dim] += 1.0 if ((h >> 1) & 1) == 0 else -1.0
    if not np.any(out):
        out[_stable_hash32(wrapped) % dim] = 1.0
    return _l2_normalize(out)


def _word2vec_vector(key: str) -> np.ndarray:
    """spaCy ``en_core_web_lg`` token vector for *key* (cached).

    mk5's ``get_word2vec`` (graph_loader_utils.py:118-125) takes
    ``nlp(desc)[0].vector``.  We mirror that — first-token vector — for
    consistency with both the matcher and the dataset-level
    embeddings.  Returns a length-``EMBED_DIM`` ``float32`` array
    (zero vector for empty input).
    """
    if not key:
        return np.zeros(EMBED_DIM, dtype=np.float32)
    cached = _MATCH_W2V_CACHE.get(key)
    if cached is not None:
        return cached
    nlp = _load_nlp()
    doc = nlp(key)
    if len(doc) == 0:
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
    else:
        vec = np.asarray(doc[0].vector, dtype=np.float32).reshape(-1)
    _MATCH_W2V_CACHE[key] = vec
    return vec


def label_embedding_for_matching(label: str) -> np.ndarray:
    """Return the matcher-side label embedding (paper §3.3).

    The result is a length-``EMBED_DIM`` ℓ₂-normalised vector built as:

    1. A spaCy ``en_core_web_lg`` first-token vector (Word2Vec / GloVe).
    2. A deterministic hashed-n-gram lexical fallback.
    3. Blend ``0.8 * w2v + 0.2 * lexical``, then L2-normalise.

    Special cases:

    - When the Word2Vec vector is degenerate (norm ≈ 0, OOV), the
      lexical embedding is used alone.
    - When the lexical embedding is degenerate (impossibly short
      label), the Word2Vec embedding is used alone.

    Cached by canonical key for O(1) repeated lookups.

    A line-by-line port of mk5's ``label_embedding_for_matching``
    (visualize_loc_prob.py:228-256).
    """
    key = canonical_label(label)
    cached = _MATCH_EMBED_CACHE.get(key)
    if cached is not None:
        return cached

    lexical = _hashed_label_embedding(key, dim=EMBED_DIM)
    w2v_arr = _word2vec_vector(key)

    if w2v_arr.size == 0:
        w2v_arr = np.zeros(EMBED_DIM, dtype=np.float32)
    elif w2v_arr.size > EMBED_DIM:
        w2v_arr = w2v_arr[:EMBED_DIM]
    elif w2v_arr.size < EMBED_DIM:
        w2v_arr = np.pad(w2v_arr, (0, EMBED_DIM - w2v_arr.size))

    w2v_unit = _l2_normalize(w2v_arr)
    if not np.any(w2v_unit):
        out = lexical
    elif not np.any(lexical):
        out = w2v_unit
    else:
        out = _l2_normalize(W2V_BLEND * w2v_unit + (1.0 - W2V_BLEND) * lexical)

    _MATCH_EMBED_CACHE[key] = out
    return out
