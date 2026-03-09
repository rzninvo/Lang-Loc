"""Dual-branch scene graph encoder for language-based scene retrieval.

Matches the architecture described in Section 3.2 of the ECCV 2026 paper:
  - Eq.3: x_i = MLP(f_i)                     -- node encoder
  - Eq.4: g_i = GATv2_geom(x, e^geom)        -- geometric branch
  - Eq.5: t_i = GATv2_rel(x, phi(r_ij))      -- relation branch (CLIP edges)
  - Eq.6: h_i = alpha*g + (1-alpha)*t         -- gated fusion
  - Eq.7: z(G) = MLP([mean-pool(h) || u(G)])  -- graph embedding with scene CLIP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from langloc.retrieval.models.networks.edge_gat import MultiGAT_Edge


class GatedFusion(nn.Module):
    """Gated fusion of geometric and text branch features (Eq.6).

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
        """Fuses geometric and text features via a learned gate.

        Args:
            g: Geometric branch features of shape ``(N, hidden_dim)``.
            t: Text branch features of shape ``(N, hidden_dim)``.

        Returns:
            Fused features of shape ``(N, hidden_dim)``.
        """
        combined = torch.cat([g, t], dim=-1)
        gate = self.gate(combined)
        fused = self.transform(combined)
        return self.norm(gate * g + (1 - gate) * t + 0.1 * fused)


class DualSceneAligner(nn.Module):
    """Dual-branch scene graph encoder with built-in scene CLIP fusion.

    Inputs (via batch dict):
      - node_feats: ``(N, 518)``  -- ``[centroid(3) | color(3) | CLIP(512)]``
      - geom_edges: ``(2, E_g)``  -- k-NN spatial edges
      - geom_attr:  ``(E_g, 8)``  -- ``[delta_c | ||delta_c|| | r_i | r_j | 0 | 0]``
      - text_edges: ``(2, E_t)``  -- relation edges
      - text_attr:  ``(E_t, 512)`` -- pre-computed CLIP embeddings of relation strings
      - scene_clip: ``(B, 512)``  -- ``phi("A room with l_1, ..., l_K")``

    Output: ``z(G)`` in ``R^d`` (default ``d=256``).

    Args:
        node_input_dim: Dimension of input node features.
        hidden_dim: Hidden dimension for GNN layers and fusion.
        dropout: Dropout probability.
    """

    def __init__(self, node_input_dim: int = 518, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Eq.3: x_i = MLP(f_i)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Eq.4: geometric branch — 8D spatial edge features
        self.gat_geom = MultiGAT_Edge(
            n_units=[hidden_dim, hidden_dim, hidden_dim],
            n_heads=[2, 2],
            edge_dim=8,
            dropout=dropout,
        )
        self.norm_geom = nn.LayerNorm(hidden_dim)

        # Eq.5: relation branch — 512D CLIP edge features
        self.gat_text = MultiGAT_Edge(
            n_units=[hidden_dim, hidden_dim, hidden_dim],
            n_heads=[2, 2],
            edge_dim=512,
            dropout=dropout,
        )
        self.norm_text = nn.LayerNorm(hidden_dim)

        # Eq.6: gated fusion
        self.fusion = GatedFusion(hidden_dim)

        # Eq.7: z(G) = MLP([mean-pool(h_i) || u(G)])
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim + 512, 256),
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
        scene_clip: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes a scene graph to a single vector z(G).

        Args:
            node_feats: Node features of shape ``(N, node_input_dim)``.
            geom_edges: Geometric edge index of shape ``(2, E_g)``.
            geom_attr: Geometric edge attributes of shape ``(E_g, 8)``.
            text_edges: Text relation edge index of shape ``(2, E_t)``.
            text_attr: CLIP text edge attributes of shape ``(E_t, 512)``.
            batch: Batch assignment vector of shape ``(N,)``.
            scene_clip: Scene-level CLIP embedding of shape ``(B, 512)``.

        Returns:
            Graph embedding of shape ``(B, 256)``.
        """
        x = self.node_encoder(node_feats)

        g = self.norm_geom(self.gat_geom(x, geom_edges, geom_attr.float()))

        # Relation branch falls back to geometric if no text edges exist
        if text_edges.size(1) > 0:
            t = self.norm_text(self.gat_text(x, text_edges, text_attr.float()))
        else:
            t = g

        h = self.fusion(g, t)

        pooled = scatter_mean(h, batch, dim=0)
        z = self.final_proj(torch.cat([pooled, scene_clip], dim=-1))
        return z

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass for paired scene graphs.

        Args:
            batch: Dictionary with keys ``*_src`` / ``*_ref`` for source and
                reference, plus ``scene_clip_src``, ``scene_clip_ref``,
                ``src_batch``, ``ref_batch``.

        Returns:
            Dictionary with ``src_emb`` and ``ref_emb``, each of shape ``(B, 256)``.
        """
        src_emb = self.encode_scene(
            batch["node_feats_src"], batch["geom_edges_src"],
            batch["geom_attr_src"], batch["text_edges_src"],
            batch["text_attr_src"], batch["src_batch"],
            batch["scene_clip_src"],
        )
        ref_emb = self.encode_scene(
            batch["node_feats_ref"], batch["geom_edges_ref"],
            batch["geom_attr_ref"], batch["text_edges_ref"],
            batch["text_attr_ref"], batch["ref_batch"],
            batch["scene_clip_ref"],
        )
        return {"src_emb": src_emb, "ref_emb": ref_emb}


# ── Self-test ────────────────────────────────────────────────
if __name__ == "__main__":
    model = DualSceneAligner(node_input_dim=518, hidden_dim=256, dropout=0.1)
    params = sum(p.numel() for p in model.parameters())
    print(f"DualSceneAligner: {params:,} parameters")

    B, N_s, N_r = 2, 15, 20
    dummy = {
        "node_feats_src": torch.randn(N_s, 518),
        "geom_edges_src": torch.randint(0, N_s, (2, 40)),
        "geom_attr_src": torch.randn(40, 8),
        "text_edges_src": torch.randint(0, N_s, (2, 25)),
        "text_attr_src": torch.randn(25, 512),
        "src_batch": torch.cat([torch.zeros(8), torch.ones(7)]).long(),
        "scene_clip_src": torch.randn(B, 512),
        "node_feats_ref": torch.randn(N_r, 518),
        "geom_edges_ref": torch.randint(0, N_r, (2, 50)),
        "geom_attr_ref": torch.randn(50, 8),
        "text_edges_ref": torch.randint(0, N_r, (2, 30)),
        "text_attr_ref": torch.randn(30, 512),
        "ref_batch": torch.cat([torch.zeros(10), torch.ones(10)]).long(),
        "scene_clip_ref": torch.randn(B, 512),
    }
    out = model(dummy)
    print(f"src_emb: {out['src_emb'].shape}, ref_emb: {out['ref_emb'].shape}")
    print("Forward pass OK")
