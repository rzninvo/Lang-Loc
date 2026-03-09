"""Shared word2vec embedding helper with module-level caching.

Provides a single ``_embed_word2vec`` function used across scene-graph
construction, localization evaluation, and description pre-processing.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from langloc.graphs.create_text_embeddings import create_embedding_nlp
from langloc.graphs.graph_loader_utils import get_word2vec


# Module-level caches — shared by all callers within a single process.
_EMBED_CACHE: Dict[str, np.ndarray] = {}
_EMBED_CACHE_TOKEN: Dict[str, np.ndarray] = {}
_W2V_HASH: Dict[str, np.ndarray] = {}


def _embed_word2vec(text: str, mode: str = "token") -> List[float]:
    """Return a word2vec embedding for *text*, using a module-level cache.

    Args:
        text: Free-form text string to embed.
        mode: ``"token"`` for word2vec token embeddings (default),
              ``"doc"`` for spaCy doc-vector embeddings.

    Returns:
        Embedding vector as a list of floats.
    """
    text = str(text)
    key = text.strip().lower()
    if mode == "doc":
        cached = _EMBED_CACHE.get(key)
        if cached is None:
            vec = np.asarray(create_embedding_nlp(text), dtype=np.float32)
            cached = vec
            _EMBED_CACHE[key] = cached
        return cached.tolist()

    cached = _EMBED_CACHE_TOKEN.get(key)
    if cached is None:
        w2v = get_word2vec(text, _W2V_HASH)
        # graph_loader_utils.get_word2vec returns (vec, cache) for non-empty text.
        vec = w2v[0] if isinstance(w2v, tuple) else w2v
        cached = np.asarray(vec, dtype=np.float32)
        _EMBED_CACHE_TOKEN[key] = cached
    return cached.tolist()
