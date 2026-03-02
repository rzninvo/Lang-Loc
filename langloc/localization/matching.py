"""Cosine-similarity Top-K object matcher for localization.

Provides a simple node-level matching strategy between a query scene graph
(built from a caption) and a reference 3D scene graph, based on cosine
similarity of node-feature vectors.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from langloc.data_processing.scene_graph import SceneGraph


def topk_matched_objects(qg: SceneGraph, sg: SceneGraph, k: int = 5) -> List:
    """Return scene-graph node IDs whose features best match the query.

    Computes the full cosine-similarity matrix between query-graph and
    scene-graph node features, then greedily picks the top-*k* unique
    scene-graph nodes from the flattened similarity ranking.

    Args:
        qg: Query scene graph (e.g. built from a caption).
        sg: Reference 3D scene graph.
        k: Maximum number of matched object IDs to return.

    Returns:
        A list of up to *k* scene-graph node IDs (in descending
        similarity order).
    """
    qf, _, _ = qg.to_pyg()
    sf, _, _ = sg.to_pyg()
    qf = F.normalize(torch.tensor(np.asarray(qf), dtype=torch.float32), dim=1)
    sf = F.normalize(torch.tensor(np.asarray(sf), dtype=torch.float32), dim=1)

    sim = qf @ sf.T                                   # (|Q|, |S|)
    topv, topi = torch.topk(sim.flatten(), min(k, sim.numel()))
    sids  = list(sg.nodes)
    S     = sf.size(0)
    picks = []
    for idx in topi.tolist():
        sid = sids[idx % S]
        if sid not in picks:
            picks.append(sid)
        if len(picks) == k:
            break
    return picks
