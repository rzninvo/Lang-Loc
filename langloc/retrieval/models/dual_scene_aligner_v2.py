"""Dual-branch scene graph encoder (V2) — paper §3.2 Eqs. 4–6.

The base graph encoder. ``final_proj`` does NOT concatenate the scene-CLIP
descriptor inside the model — scene-CLIP is fused externally by the
``SimpleGraphMatcher`` wrapper (see ``simple_graph_matcher.py``).

Layer-by-layer state-dict shapes are bit-identical to the published paper
checkpoint's ``base_model.*`` keys.

Inputs (via batch dict):
    node_feats: ``(N, 518)``  -- ``[centroid(3) | color(3) | CLIP(512)]``
    geom_edges: ``(2, E_g)``  -- k-NN spatial edges
    geom_attr:  ``(E_g, 8)``  -- 8-D edge features
    text_edges: ``(2, E_t)``  -- relation edges
    text_attr:  ``(E_t, 512)`` -- pre-computed CLIP embeddings of relation strings
    src_batch / ref_batch: ``(N,)`` graph assignment

Output: ``z(G)`` in ``R^d`` where ``d = hidden_dim`` (default 256). Scene-CLIP
fusion happens in the wrapper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from langloc.retrieval.models.networks.edge_gat import MultiGAT_Edge


class GatedFusion(nn.Module):
    """Gated fusion of geometric and text branch features (paper Eq. 6).

    Args:
        hidden_dim: Dimension of the input feature vectors from each branch.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.transform = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, g: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([g, t], dim=-1)
        gate = self.gate(combined)
        fused = self.transform(combined)
        return self.norm(gate * g + (1 - gate) * t + 0.1 * fused)


class DualSceneAlignerV2(nn.Module):
    """Dual-branch graph encoder (paper §3.2). Scene-CLIP fused externally.

    Args:
        node_input_dim: Input node feature dim (518 = 3+3+512).
        hidden_dim: Internal/output hidden dim (256).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        node_input_dim: int = 518,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Eq. 3: x_i = MLP(f_i)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Eq. 4: geometric branch — 8-D spatial edge features
        self.gat_geom = MultiGAT_Edge(
            n_units=[hidden_dim, hidden_dim, hidden_dim],
            n_heads=[2, 2],
            edge_dim=8,
            dropout=dropout,
        )
        self.norm_geom = nn.LayerNorm(hidden_dim)

        # Eq. 5: relation branch — 512-D CLIP edge features
        self.gat_text = MultiGAT_Edge(
            n_units=[hidden_dim, hidden_dim, hidden_dim],
            n_heads=[2, 2],
            edge_dim=512,
            dropout=dropout,
        )
        self.norm_text = nn.LayerNorm(hidden_dim)

        # Eq. 6: gated fusion (geom + text)
        self.fusion = GatedFusion(hidden_dim)

        # Eq. 7 (paper): z(G) = MLP(mean-pool(h_i)) — scene-CLIP NOT concatenated
        # here. Concat is moved to ``SimpleGraphMatcher.fusion``.
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
        )

    def encode_scene(
        self,
        node_feats: torch.Tensor,
        geom_edges: torch.Tensor,
        geom_attr: torch.Tensor,
        text_edges: torch.Tensor,
        text_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes a single scene graph to a 256-D vector (no scene-CLIP fusion)."""
        x = self.node_encoder(node_feats)

        g = self.norm_geom(self.gat_geom(x, geom_edges, geom_attr.float()))

        if text_edges.size(1) > 0:
            t = self.norm_text(self.gat_text(x, text_edges, text_attr.float()))
        else:
            t = g

        h = self.fusion(g, t)
        pooled = scatter_mean(h, batch, dim=0)
        z = self.final_proj(pooled)
        return z

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward over paired (src, ref) graphs.

        Note: scene-CLIP is NOT consumed here; the wrapper is responsible.
        """
        src_emb = self.encode_scene(
            batch["node_feats_src"], batch["geom_edges_src"],
            batch["geom_attr_src"], batch["text_edges_src"],
            batch["text_attr_src"], batch["src_batch"],
        )
        ref_emb = self.encode_scene(
            batch["node_feats_ref"], batch["geom_edges_ref"],
            batch["geom_attr_ref"], batch["text_edges_ref"],
            batch["text_attr_ref"], batch["ref_batch"],
        )
        return {"src_emb": src_emb, "ref_emb": ref_emb}
