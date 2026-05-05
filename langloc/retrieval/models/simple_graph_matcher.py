"""SimpleGraphMatcher — late scene-CLIP fusion wrapper around DualSceneAlignerV2.

Architecture used to produce the paper's Tables 1-3 numbers
(``DualSceneAlignerV2 + SimpleGraphMatcher``, 1,722,112 trainable
parameters total).

Forward path:

    base_emb = base_model(batch)          # 256-D, scene-CLIP not used here
    fused    = LayerNorm(d + 512) → Linear(d+512, 256) → ReLU → Dropout(0.3)
                → Linear(256, 256) → LayerNorm(256)
    [src/ref]_emb = fused([base_emb_src/ref || scene_clip_src/ref])

State-dict at indices 0/1/4/5 of ``self.fusion`` corresponds to the trainable
weights (ReLU and Dropout have no params).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from langloc.retrieval.models.dual_scene_aligner_v2 import DualSceneAlignerV2


class SimpleGraphMatcher(nn.Module):
    """Late scene-CLIP fusion wrapper for DualSceneAlignerV2.

    Args:
        base_model: An instantiated ``DualSceneAlignerV2``.
        scene_clip_dim: Scene-CLIP embedding dim (512 for ViT-B/32).
        hidden_dim: Output dim (256).
    """

    def __init__(
        self,
        base_model: DualSceneAlignerV2,
        scene_clip_dim: int = 512,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.fusion = nn.Sequential(
            nn.LayerNorm(base_model.hidden_dim + scene_clip_dim),
            nn.Linear(base_model.hidden_dim + scene_clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        scene_clip_src: torch.Tensor | None = None,
        scene_clip_ref: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run base model then fuse scene-CLIP at the head.

        Accepts scene-CLIP either as explicit kwargs
        or pulled from the batch (``batch['scene_clip_src/ref']``).
        """
        if scene_clip_src is None:
            scene_clip_src = batch["scene_clip_src"]
        if scene_clip_ref is None:
            scene_clip_ref = batch["scene_clip_ref"]

        out = self.base_model(batch)
        gnn_src = out["src_emb"]
        gnn_ref = out["ref_emb"]

        src_emb = self.fusion(torch.cat([gnn_src, scene_clip_src], dim=-1))
        ref_emb = self.fusion(torch.cat([gnn_ref, scene_clip_ref], dim=-1))
        return {"src_emb": src_emb, "ref_emb": ref_emb}
