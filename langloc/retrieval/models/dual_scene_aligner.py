

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max

from langloc.retrieval.models.networks.edge_gat import MultiGAT_Edge


# ============================================================
# 1. Relation Embedding (Dummy - not really used)
# ============================================================

class RelationEmbedding(nn.Module):
    """
    Simple learnable relation embeddings.
    Since we're generating synthetic relations, we don't need CLIP.
    """
    def __init__(self, num_relations, emb_dim=64):
        super().__init__()
        self.emb = nn.Embedding(num_relations, emb_dim)
        # Initialize with reasonable variance
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.5)

    def forward(self, rel_ids):
        """
        Args:
            rel_ids: (E,) relation IDs
        Returns:
            (E, emb_dim) relation embeddings
        """
        rel_ids = rel_ids.view(-1).long()
        return self.emb(rel_ids)


# ============================================================
# 2. Pooling Layers
# ============================================================

class AttentionPool(nn.Module):
    """Attention-based pooling for graph-level representations."""
    
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x, batch):
        """
        Args:
            x: (N, dim) node features
            batch: (N,) batch assignment
        Returns:
            (B, dim) pooled features
        """
        # Compute attention scores
        scores = self.att(x).squeeze(-1)  # (N,)
        
        # Stable softmax per graph
        max_scores = scatter_max(scores, batch, dim=0)[0]
        scores = scores - max_scores[batch]
        
        exp_scores = torch.exp(scores)
        sum_exp = scatter_mean(exp_scores, batch, dim=0) * \
                  torch.bincount(batch).float()  # Convert mean back to sum
        
        alpha = exp_scores / (sum_exp[batch] + 1e-8)
        
        # Weighted sum
        weighted = alpha.unsqueeze(-1) * x
        pooled = scatter_mean(weighted, batch, dim=0) * \
                torch.bincount(batch).unsqueeze(-1).float()
        
        return pooled


class MeanPool(nn.Module):
    """Simple mean pooling."""
    
    def forward(self, x, batch):
        return scatter_mean(x, batch, dim=0)


class MaxPool(nn.Module):
    """Simple max pooling."""
    
    def forward(self, x, batch):
        return scatter_max(x, batch, dim=0)[0]


# ============================================================
# 3. Fusion Layer
# ============================================================

class GatedFusion(nn.Module):
    """
    Gated fusion of geometric and text features.
    Learns to weight the importance of each modality.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.transform = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, g, t):
        """
        Args:
            g: (N, hidden_dim) geometric features
            t: (N, hidden_dim) text features
        Returns:
            (N, hidden_dim) fused features
        """
        combined = torch.cat([g, t], dim=-1)
        
        # Gating mechanism
        gate = self.gate(combined)
        
        # Transformation
        fused = self.transform(combined)
        
        # Residual connection with gating
        output = gate * g + (1 - gate) * t + 0.1 * fused
        
        return self.norm(output)


# ============================================================
# 4. Main Model
# ============================================================

class DualSceneAligner(nn.Module):
    """
    Dual-branch scene graph encoder with:
    - Geometric branch (processes spatial relationships)
    - Text branch (processes semantic relationships)
    - Fusion layer (combines modalities)
    - Graph-level pooling
    """
    
    def __init__(
        self,
        node_input_dim=518,      # 3 (pos) + 3 (color) + 512 (features)
        relation_dim=64,         # Learnable relation embedding size
        hidden_dim=128,
        rel_clip_matrix=None,    # Ignored - we use learnable embeddings
        dropout=0.1,
        pool_type='mean'         # 'mean', 'max', or 'attention'
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # ===== Input Processing =====
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # ===== Relation Embedding =====
        # Use learnable embeddings instead of CLIP
        self.rel_emb = RelationEmbedding(
            num_relations=10,  # Generous size
            emb_dim=relation_dim
        )
        
        # ===== Geometric Branch =====
        self.gat_geom = MultiGAT_Edge(
            n_units=[hidden_dim, hidden_dim, hidden_dim],
            n_heads=[2, 2],
            edge_dim=8,
            dropout=dropout
        )
        self.norm_geom = nn.LayerNorm(hidden_dim)
        
        # ===== Text Branch =====
        self.gat_text = MultiGAT_Edge(
            n_units=[hidden_dim, hidden_dim, hidden_dim],
            n_heads=[2, 2],
            edge_dim=relation_dim,
            dropout=dropout
        )
        self.norm_text = nn.LayerNorm(hidden_dim)
        
        # ===== Fusion =====
        self.fusion = GatedFusion(hidden_dim)
        
        # ===== Pooling =====
        if pool_type == 'attention':
            self.pool = AttentionPool(hidden_dim)
        elif pool_type == 'max':
            self.pool = MaxPool()
        else:  # mean
            self.pool = MeanPool()
        
        # ===== Final Projection =====
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256)
        )

    def encode_scene(
        self,
        node_feats,
        geom_edges,
        geom_attr,
        text_edges,
        text_attr,
        batch
    ):
        """
        Encode a batched scene graph.
        
        Args:
            node_feats: (N_total, node_input_dim)
            geom_edges: (2, E_geom)
            geom_attr: (E_geom, 8)
            text_edges: (2, E_text)
            text_attr: (E_text, 1)
            batch: (N_total,) batch assignment
        
        Returns:
            (B, 256) scene embeddings
        """
        # ===== 1. Encode nodes =====
        x = self.node_encoder(node_feats)  # (N_total, hidden_dim)
        
        # ===== 2. Geometric branch =====
        g = self.gat_geom(x, geom_edges, geom_attr)
        g = self.norm_geom(g)
        
        # ===== 3. Text branch =====
        # Get relation embeddings
        if text_edges.size(1) > 0:
            rel_ids = text_attr.squeeze(-1).long()
            rel_emb = self.rel_emb(rel_ids)  # (E_text, relation_dim)
            
            t = self.gat_text(g, text_edges, rel_emb)  # Use geometric features as input
            t = self.norm_text(t)
        else:
            # No text edges - just use geometric features
            t = g
        
        # ===== 4. Fusion =====
        fused = self.fusion(g, t)  # (N_total, hidden_dim)
        
        # ===== 5. Add noise during training (prevent collapse) =====
        if self.training:
            noise = torch.randn_like(fused) * 0.01
            fused = fused + noise
        
        # ===== 6. Pool to graph level =====
        pooled = self.pool(fused, batch)  # (B, hidden_dim)
        
        # ===== 7. Final projection =====
        scene_emb = self.final_proj(pooled)  # (B, 256)
        
        return scene_emb

    def forward(self, batch):
        """
        Forward pass for batched scene graphs.
        
        Args:
            batch: Dictionary containing:
                - node_feats_src, geom_edges_src, etc. for source
                - node_feats_ref, geom_edges_ref, etc. for reference
                - src_batch, ref_batch: batch assignments
        
        Returns:
            Dictionary with:
                - src_emb: (B, 256)
                - ref_emb: (B, 256)
        """
        src_emb = self.encode_scene(
            batch["node_feats_src"],
            batch["geom_edges_src"],
            batch["geom_attr_src"],
            batch["text_edges_src"],
            batch["text_attr_src"],
            batch["src_batch"]
        )

        ref_emb = self.encode_scene(
            batch["node_feats_ref"],
            batch["geom_edges_ref"],
            batch["geom_attr_ref"],
            batch["text_edges_ref"],
            batch["text_attr_ref"],
            batch["ref_batch"]
        )

        return {
            "src_emb": src_emb,
            "ref_emb": ref_emb
        }


# ============================================================
# Factory Function
# ============================================================

def create_dual_scene_aligner(
    node_input_dim=518,
    hidden_dim=128,
    dropout=0.1,
    pool_type='mean'
):
    """
    Factory function to create a DualSceneAligner model.
    
    Args:
        node_input_dim: Dimension of input node features
        hidden_dim: Hidden dimension for GNN layers
        dropout: Dropout rate
        pool_type: 'mean', 'max', or 'attention'
    
    Returns:
        DualSceneAligner model
    """
    # Create dummy CLIP matrix (not used)
    dummy_clip = torch.zeros(10, 512)
    
    model = DualSceneAligner(
        node_input_dim=node_input_dim,
        relation_dim=64,
        hidden_dim=hidden_dim,
        rel_clip_matrix=dummy_clip,
        dropout=dropout,
        pool_type=pool_type
    )
    
    return model


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("Dual Scene Aligner - Refactored Version")
    print("="*70)
    
    # Create model
    model = create_dual_scene_aligner(
        node_input_dim=518,
        hidden_dim=128,
        dropout=0.1,
        pool_type='mean'
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    n_nodes_src = 20
    n_nodes_ref = 25
    
    dummy_batch = {
        "node_feats_src": torch.randn(n_nodes_src, 518),
        "geom_edges_src": torch.randint(0, n_nodes_src, (2, 50)),
        "geom_attr_src": torch.randn(50, 8),
        "text_edges_src": torch.randint(0, n_nodes_src, (2, 30)),
        "text_attr_src": torch.randint(0, 7, (30, 1)),
        "src_batch": torch.cat([torch.zeros(10), torch.ones(10)]).long(),
        
        "node_feats_ref": torch.randn(n_nodes_ref, 518),
        "geom_edges_ref": torch.randint(0, n_nodes_ref, (2, 60)),
        "geom_attr_ref": torch.randn(60, 8),
        "text_edges_ref": torch.randint(0, n_nodes_ref, (2, 40)),
        "text_attr_ref": torch.randint(0, 7, (40, 1)),
        "ref_batch": torch.cat([torch.zeros(12), torch.ones(13)]).long(),
    }
    
    print(f"\nTesting forward pass...")
    output = model(dummy_batch)
    
    print(f"  Source embedding: {output['src_emb'].shape}")
    print(f"  Reference embedding: {output['ref_emb'].shape}")
    print(f"  ✓ Forward pass successful!")
    
    print("\n" + "="*70)
    print("Ready to train!")
    print("="*70)



# """
# Dual Scene Aligner (full CLIP support, stable dimensions, no collapse)
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_scatter import scatter_mean
# from torch_scatter import scatter_add
# from torch_scatter import scatter_max


# from langloc.retrieval.models.networks.edge_gat import MultiGAT_Edge


# # ============================================================
# # 1. Relation Embedding using CLIP (already precomputed)
# # ============================================================

# class CLIPRelationEmbedding(nn.Module):
#     """Lookup table for relation CLIP embeddings (E, 512)."""

#     def __init__(self, clip_matrix):  
#         super().__init__()
#         # clip_matrix = tensor (num_relations, 512)
#         self.register_buffer("clip_embs", clip_matrix)

#     def forward(self, rel_ids):
#         rel_ids = rel_ids.view(-1).long()
#         return self.clip_embs[rel_ids]
    

# # ============================================================
# # 2. Attention pooling (batch-aware)
# # ============================================================

# class AttentionPool(nn.Module):

#     def __init__(self, dim):
#         super().__init__()
#         self.att = nn.Sequential(
#             nn.Linear(dim, dim // 2),
#             nn.GELU(),
#             nn.Linear(dim // 2, 1)
#         )

#     def forward(self, x, batch):

#         # attention weights (N,1)
#         scores = self.att(x)

#         # softmax per graph
#         scores = scores.squeeze(-1)  # (N,)
#         scores_max = scatter_add(scores, batch, dim=0).index_select(0, batch)
#         scores = scores - scores_max   # stability

#         exp_w = torch.exp(scores)
#         denom = scatter_add(exp_w, batch, dim=0).index_select(0, batch)

#         alpha = exp_w / (denom + 1e-8)

#         pooled = scatter_add(alpha.unsqueeze(-1) * x, batch, dim=0)
#         print("\n=== AttentionPool DEBUG ===")
#         print("x:", x.shape)
#         print("batch:", batch.shape)
#         print("Att scores:", scores.shape, "std:", scores.std().item())
#         print("Pooled:", pooled.shape)
#         return pooled


# ##################### MAX POOLING VERSION #####################
# class MaxPool(nn.Module):
#     def forward(self, x, batch):
#         return scatter_max(x, batch, dim=0)[0]

# ############### FUSION NETWORK ####################
# class Fusion(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()

#         self.norm_in = nn.LayerNorm(hidden_dim * 2)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )
#         self.norm_out = nn.LayerNorm(hidden_dim)

#     def forward(self, g, t):
#         x = torch.cat([g, t], dim=-1)

#         x = self.norm_in(x)
#         x = self.mlp(x)

#         x = self.norm_out(x)

#         # scaled residual
#         return g + t + 0.01 * x
    

# # ============================================================
# # 3. Dual Scene Aligner (CLIP version)
# # ============================================================
# class DualSceneAligner(nn.Module):
#     def __init__(
#         self,
#         node_input_dim=518,       # 3 + 3 + 512
#         relation_dim=512,
#         hidden_dim=128,

#         rel_clip_matrix=None,    # (num_relations, 512)
#         dropout=0.0
#     ):
#         super().__init__()

#         self.rel_emb = CLIPRelationEmbedding(rel_clip_matrix)

#         # -----------------------------------------------------
#         # GEOMETRY GAT
#         # -----------------------------------------------------
#         self.gat_geom = MultiGAT_Edge(
#             n_units=[hidden_dim, hidden_dim, hidden_dim],
#             n_heads=[2, 2],
#             edge_dim=8,
#             dropout=dropout
#         )

#         # -----------------------------------------------------
#         # TEXT GAT
#         # -----------------------------------------------------
#         self.gat_text = MultiGAT_Edge(
#             n_units=[hidden_dim, hidden_dim, hidden_dim],
#             n_heads=[2, 2],
#             edge_dim=relation_dim,   # ← MUST MATCH CLIP
#             dropout=dropout
#         )

#         # -----------------------------------------------------
#         # Fusion layer (very important!)
#         # -----------------------------------------------------
#         self.fusion = Fusion(hidden_dim)

#         # -----------------------------------------------------
#         # Attention pooling
#         # -----------------------------------------------------
#         # self.pool = AttentionPool(hidden_dim)
#         self.pool = MaxPool()

#         self.node_encoder = nn.Sequential(
#             nn.Linear(node_input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, hidden_dim),
#             nn.ReLU(),
# )
#         # -----------------------------------------------------
#         # Final scene embedding
#         # -----------------------------------------------------
#         self.final = nn.Sequential(
#             nn.Linear(hidden_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#         )
            
#         self.pre_norm_g = nn.LayerNorm(hidden_dim)
#         self.pre_norm_t = nn.LayerNorm(hidden_dim)

#     # ---------------------------------------------------------
#     # PER-SCENE ENCODING
#     # ---------------------------------------------------------
#     def encode_scene(self, node_feats, geom_edges, geom_attr,
#                      text_edges, text_attr, batch):
#         print("\n=== ENCODE SCENE DEBUG ===")
#         print("Node feats:", node_feats.shape)
#         print("Geom edges:", geom_edges.shape)
#         print("Geom attr:", geom_attr.shape)
#         print("Text edges:", text_edges.shape)
#         print("Text attr:", text_attr.shape)
#         # Normalize CLIP NODE embeddings
#         node_feats[:, 6:] = F.normalize(node_feats[:, 6:], p=2, dim=-1)
#         node_feats = self.node_encoder(node_feats)
#         # 1) Geometry
#         g = self.gat_geom(
#             node_feats,
#             geom_edges,
#             geom_attr,
#             # edge_weight_factor=0.2,
#             # skip_edges_first_layer=True
#         )

#         # 2) Text relation embeddings
#         rel_ids = text_attr.squeeze(-1).long()    # (E,)

#         print("Relation IDs:", rel_ids[:10])
#         print("Unique relation IDs:", rel_ids.unique())
#         print("CLIP relation embedding:", self.rel_emb(rel_ids)[:5])
#         rel = self.rel_emb(rel_ids)               # (E,512)
#         # 3) Text GAT
#         t = self.gat_text(
#             g,
#             text_edges,
#             rel,
#             # edge_weight_factor=0.1,
#             # skip_edges_first_layer=True
#         )
#         print("G geom:", g.shape, "mean:", g.mean().item(), "std:", g.std().item())
#         print("T text:", t.shape, "mean:", t.mean().item(), "std:", t.std().item())
#         g = self.pre_norm_g(g)
#         t = self.pre_norm_t(t)

#         # 4) Fusion at node level (THIS FIXES ALIGNMENT)
#         fused = self.fusion(g, t)
#         print("Fused:", fused.shape, "mean:", fused.mean().item(), "std:", fused.std().item())

#         if self.training:
#             fused = fused + 0.01 * torch.randn_like(fused)

#         # 5) Pool to graph level
#         pooled = self.pool(fused, batch)
#         print("Pooled:", pooled.shape, "mean:", pooled.mean().item(), "std:", pooled.std().item())

#         # 6) Final projection
#         scene_emb = self.final(pooled)
#         print("Scene embedding:", scene_emb.shape, scene_emb.std(dim=0).mean().item())
#         return scene_emb

#     # ---------------------------------------------------------
#     def forward(self, batch):

#         src = self.encode_scene(
#             batch["node_feats_src"],
#             batch["geom_edges_src"],
#             batch["geom_attr_src"],
#             batch["text_edges_src"],
#             batch["text_attr_src"],
#             batch["src_batch"]
#         )

#         ref = self.encode_scene(
#             batch["node_feats_ref"],
#             batch["geom_edges_ref"],
#             batch["geom_attr_ref"],
#             batch["text_edges_ref"],
#             batch["text_attr_ref"],
#             batch["ref_batch"]
#         )

#         return {"src_emb": src, "ref_emb": ref}