"""Object matching strategies for localization.

Provides three matching strategies between a query scene graph (built from
a caption) and a reference 3D scene graph:

- **global_topk** (default): Simple global top-k from flattened cosine
  similarity matrix.
- **per_node**: Per-query-node best match first, then fill remaining
  slots from global ranking.  Prevents a single repeated label from
  dominating all k slots.
- **relation_aware**: Per-node matching re-ranked by relation consistency
  (``(1-alpha)*node_sim + alpha*relation_bonus``).  Greedy: most-confident
  query nodes anchor first.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from langloc.graphs.scene_graph import SceneGraph


# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------

def _compute_sim_matrix(qg: SceneGraph, sg: SceneGraph) -> torch.Tensor:
    """Compute the cosine-similarity matrix between two scene graphs.

    Extracts node features via ``to_pyg()``, L2-normalises them, and
    returns the ``(|Q|, |S|)`` cosine-similarity matrix.

    Args:
        qg: Query scene graph.
        sg: Reference scene graph.

    Returns:
        Float tensor of shape ``(|Q|, |S|)``.
    """
    qf, _, _ = qg.to_pyg()
    sf, _, _ = sg.to_pyg()
    qf = F.normalize(torch.tensor(np.asarray(qf), dtype=torch.float32), dim=1)
    sf = F.normalize(torch.tensor(np.asarray(sf), dtype=torch.float32), dim=1)
    return qf @ sf.T


def _fill_from_global(sim: torch.Tensor, sids: List[int],
                      picks: List[int], k: int) -> List[int]:
    """Fill remaining slots from global similarity ranking.

    Args:
        sim: ``(|Q|, |S|)`` similarity matrix.
        sids: Scene-graph node IDs (ordered as columns of ``sim``).
        picks: Already-selected node IDs (mutated in-place).
        k: Target number of picks.

    Returns:
        The ``picks`` list (same object, extended up to *k*).
    """
    if len(picks) >= k:
        return picks[:k]
    _, topi = torch.topk(sim.flatten(), min(k * 3, sim.numel()))
    S = len(sids)
    for idx in topi.tolist():
        sid = sids[idx % S]
        if sid not in picks:
            picks.append(sid)
        if len(picks) >= k:
            break
    return picks[:k]


# ---------------------------------------------------------------------------
#  Strategy selector
# ---------------------------------------------------------------------------

def match_objects(qg: SceneGraph,
                  sg: SceneGraph,
                  k: int = 5,
                  strategy: str = "global_topk",
                  relation_alpha: float = 0.5) -> List[int]:
    """Unified entry point for object matching.

    Args:
        qg: Query scene graph.
        sg: Reference 3D scene graph.
        k: Maximum number of matched object IDs to return.
        strategy: ``"global_topk"``, ``"per_node"``, or
            ``"relation_aware"``.
        relation_alpha: Blend weight for ``relation_aware`` strategy.
            0 = pure node similarity, 1 = pure relation consistency.

    Returns:
        A list of up to *k* scene-graph node IDs.
    """
    if strategy == "per_node":
        return per_node_matched_objects(qg, sg, k)
    if strategy == "relation_aware":
        return relation_aware_matched_objects(qg, sg, k, alpha=relation_alpha)
    return topk_matched_objects(qg, sg, k)


# ---------------------------------------------------------------------------
#  Global top-k (original)
# ---------------------------------------------------------------------------

def topk_matched_objects(qg: SceneGraph, sg: SceneGraph, k: int = 5) -> List[int]:
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
    sim = _compute_sim_matrix(qg, sg)
    sids = list(sg.nodes)
    return _fill_from_global(sim, sids, [], k)


# ---------------------------------------------------------------------------
#  Per-node best match
# ---------------------------------------------------------------------------

def per_node_matched_objects(qg: SceneGraph, sg: SceneGraph, k: int = 5) -> List[int]:
    """Match each query node to its best scene-graph node, then fill with global top-k.

    Prevents a single repeated label (e.g. 9 "wall" nodes) from
    dominating all k slots.

    Args:
        qg: Query scene graph.
        sg: Reference 3D scene graph.
        k: Maximum number of matched object IDs to return.

    Returns:
        A list of up to *k* scene-graph node IDs.
    """
    sim = _compute_sim_matrix(qg, sg)
    sids = list(sg.nodes)

    picks: List[int] = []
    for qi in range(sim.size(0)):
        best_si = int(sim[qi].argmax())
        sid = sids[best_si]
        if sid not in picks:
            picks.append(sid)

    return _fill_from_global(sim, sids, picks, k)


# ---------------------------------------------------------------------------
#  Relation-aware matching helpers
# ---------------------------------------------------------------------------

def _build_edge_lookup(sg: SceneGraph) -> Dict[Tuple[int, int], Tuple[str, np.ndarray]]:
    """Build a fast edge lookup from a SceneGraph's edge lists.

    Returns:
        Dict mapping ``(from_id, to_id)`` to ``(relation_str, relation_embedding)``.
    """
    lookup: Dict[Tuple[int, int], Tuple[str, np.ndarray]] = {}
    if not sg.edge_idx or len(sg.edge_idx) < 2:
        return lookup
    for i, (f, t) in enumerate(zip(sg.edge_idx[0], sg.edge_idx[1])):
        emb = (np.asarray(sg.edge_features[i], dtype=np.float32)
               if i < len(sg.edge_features) else None)
        lookup[(int(f), int(t))] = (sg.edge_relations[i], emb)
    return lookup


def _relation_bonus(q_edge_lookup: Dict[Tuple[int, int], Tuple[str, np.ndarray]],
                    s_edge_lookup: Dict[Tuple[int, int], Tuple[str, np.ndarray]],
                    assignment: Dict[int, int]) -> float:
    """Score how well a trial assignment preserves query-graph relations.

    For each query edge ``(qi, qj)``, checks if the assigned scene nodes
    ``(si, sj)`` share a matching relation in the scene graph, using cosine
    similarity of relation embeddings.

    Returns:
        Average relation similarity in [0, 1], or 0.0 if no edges can be
        checked.
    """
    total, checked = 0.0, 0
    for (qi, qj), (_, q_emb) in q_edge_lookup.items():
        si = assignment.get(qi)
        sj = assignment.get(qj)
        if si is None or sj is None:
            continue
        s_edge = s_edge_lookup.get((si, sj)) or s_edge_lookup.get((sj, si))
        checked += 1
        if s_edge is not None and q_emb is not None and s_edge[1] is not None:
            q_n = q_emb / (np.linalg.norm(q_emb) + 1e-8)
            s_n = s_edge[1] / (np.linalg.norm(s_edge[1]) + 1e-8)
            total += max(float(q_n @ s_n), 0.0)
    return total / checked if checked > 0 else 0.0


# ---------------------------------------------------------------------------
#  Relation-aware matching
# ---------------------------------------------------------------------------

def relation_aware_matched_objects(qg: SceneGraph,
                                   sg: SceneGraph,
                                   k: int = 5,
                                   alpha: float = 0.5) -> List[int]:
    """Match query nodes using both node similarity and relation consistency.

    For each query node, considers the top-C candidates by cosine similarity,
    then re-ranks by ``(1-alpha)*node_sim + alpha*relation_bonus``.
    Greedy: most-confident nodes are assigned first to anchor the graph.

    Args:
        qg: Query scene graph.
        sg: Reference 3D scene graph.
        k: Maximum number of matched object IDs to return.
        alpha: Blend weight. 0 = pure node similarity, 1 = pure relation
            consistency.

    Returns:
        A list of up to *k* scene-graph node IDs.
    """
    sim = _compute_sim_matrix(qg, sg)
    q_ids = list(qg.nodes)
    s_ids = list(sg.nodes)

    q_edge_lookup = _build_edge_lookup(qg)
    s_edge_lookup = _build_edge_lookup(sg)

    C = min(10, len(s_ids))
    topk_vals, topk_idx = sim.topk(C, dim=1)  # (|Q|, C)

    # Greedy assignment: anchor most-confident query nodes first
    assignment: Dict[int, int] = {}
    used_sids: set = set()
    q_order = sorted(range(len(q_ids)), key=lambda qi: -float(sim[qi].max()))

    for qi in q_order:
        qid = q_ids[qi]
        best_score = -float("inf")
        best_sid = None

        for c in range(C):
            si = int(topk_idx[qi, c])
            sid = s_ids[si]
            if sid in used_sids:
                continue

            node_sim = float(topk_vals[qi, c])

            # Tentatively assign and score relation consistency
            trial = dict(assignment)
            trial[qid] = sid
            r_bonus = _relation_bonus(q_edge_lookup, s_edge_lookup, trial)

            combined = (1.0 - alpha) * node_sim + alpha * r_bonus
            if combined > best_score:
                best_score = combined
                best_sid = sid

        if best_sid is not None:
            assignment[qid] = best_sid
            used_sids.add(best_sid)

    picks = list(assignment.values())
    return _fill_from_global(sim, s_ids, picks, k)
