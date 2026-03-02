"""
Training with FIXED architecture:
- 518D node features (NO scene CLIP in nodes!)
- 64D learned relation embeddings  
- Scene CLIP added AFTER GNN (fusion layer)
- Subgraph augmentation (50% of time)
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import argparse
import json

from src.retrieval.datasets.dual_scene_graph_dataset import DualSceneGraphDataset
from src.retrieval.models.dual_scene_aligner import DualSceneAligner
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import random


# ============================================================
# Model with Scene CLIP Fusion
# ============================================================

class DualSceneAlignerWithSceneCLIPFusion(nn.Module):
    """
    GNN processes 518D node features → 256D graph embedding
    Then: Fuse with scene CLIP → final embedding
    
    This prevents scene CLIP from dominating node features!
    """
    def __init__(self, base_model, scene_clip_dim=512, hidden_dim=256):
        super().__init__()
        self.base_model = base_model
        
        # Fusion layer: [GNN(256D) + scene_CLIP(512D)] → 256D
        # Added layer norm to prevent scene CLIP from dominating
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim + scene_clip_dim),  # ← Normalize inputs
            nn.Linear(hidden_dim + scene_clip_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),  # ← Increased dropout
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, batch, scene_clip_src, scene_clip_ref):
        """
        Args:
            batch: Graph batch with 518D node features
            scene_clip_src: [batch_size, 512] scene CLIP
            scene_clip_ref: [batch_size, 512] scene CLIP
        """
        # Get GNN embeddings from graph structure
        out = self.base_model(batch)
        gnn_src = out["src_emb"]  # [batch_size, 256]
        gnn_ref = out["ref_emb"]  # [batch_size, 256]
        print(f"GNN norm: {gnn_src.norm(dim=-1).mean():.4f}")
        print(f"Scene CLIP norm: {scene_clip_src.norm(dim=-1).mean():.4f}")
        print(f"GNN std: {gnn_src.std():.4f}")
        print(f"Scene std: {scene_clip_src.std():.4f}")
        # Fuse GNN + scene CLIP
        src_combined = torch.cat([gnn_src, scene_clip_src], dim=-1)  # [B, 768]
        ref_combined = torch.cat([gnn_ref, scene_clip_ref], dim=-1)  # [B, 768]
        
        src_emb = self.fusion(src_combined)  # [B, 256]
        ref_emb = self.fusion(ref_combined)  # [B, 256]
        
        return {
            "src_emb": src_emb,
            "ref_emb": ref_emb
        }


# ============================================================
# Supervised Contrastive Loss
# ============================================================

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels, hard_neg_idx=None, hard_neg_weight=3.0):
        """
        embeddings: [2B, D]   (src followed by ref)
        labels:     [2B]
        hard_neg_idx: [B]  index of hard ref for each src
        """
        device = embeddings.device
        embeddings = F.normalize(embeddings, dim=-1)

        sim = embeddings @ embeddings.T / self.temperature
        exp_sim = torch.exp(sim)

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)

        neg_mask = 1.0 - pos_mask
        neg_mask.fill_diagonal_(0)

        B = labels.size(0) // 2

        # ---------------------------
        # Build HARD negative mask
        # ---------------------------
        hard_mask = torch.zeros_like(neg_mask)

        if hard_neg_idx is not None:
            for i in range(B):
                j = hard_neg_idx[i].item()
                hard_mask[i, B + j] = 1.0  # ONLY src → ref

        loss = 0.0
        valid = 0

        for i in range(2 * B):
            pos_sum = (exp_sim[i] * pos_mask[i]).sum()
            if pos_sum == 0:
                continue

            # 🔴 KEY CHANGE: separate hard negatives
            easy_neg = (exp_sim[i] * neg_mask[i] * (1 - hard_mask[i])).sum()
            hard_neg = (exp_sim[i] * hard_mask[i]).sum()

            denom = pos_sum + easy_neg + hard_neg_weight * hard_neg

            loss += -torch.log(pos_sum / (denom + 1e-8))
            valid += 1

        return loss / max(valid, 1)
# ============================================================
# Collate Function
# ============================================================

def collate_graph_batch_with_scene_clip(batch_list):
    """Collate function that handles scene CLIP separately."""
    batch_size = len(batch_list)

    node_feats_src_list = []
    geom_edges_src_list = []
    geom_attr_src_list = []
    text_edges_src_list = []
    text_attr_src_list = []

    node_feats_ref_list = []
    geom_edges_ref_list = []
    geom_attr_ref_list = []
    text_edges_ref_list = []
    text_attr_ref_list = []

    src_batch_idx = []
    ref_batch_idx = []
    
    # Scene CLIP (not in node features!)
    scene_clip_src_list = []
    scene_clip_ref_list = []
    room_ids = []

    src_node_offset = 0
    ref_node_offset = 0

    for i, sample in enumerate(batch_list):
        # SOURCE
        n_src = sample["node_feats_src"].size(0)
        node_feats_src_list.append(sample["node_feats_src"])

        g_edges = sample["geom_edges_src"]
        if g_edges.size(1) > 0:
            g_edges = g_edges + src_node_offset
        geom_edges_src_list.append(g_edges)
        geom_attr_src_list.append(sample["geom_attr_src"])

        t_edges = sample["text_edges_src"]
        if t_edges.size(1) > 0:
            t_edges = t_edges + src_node_offset
        text_edges_src_list.append(t_edges)
        text_attr_src_list.append(sample["text_attr_src"].view(-1, 1))

        src_batch_idx.extend([i] * n_src)
        src_node_offset += n_src

        # REFERENCE
        n_ref = sample["node_feats_ref"].size(0)
        node_feats_ref_list.append(sample["node_feats_ref"])

        g_edges = sample["geom_edges_ref"]
        if g_edges.size(1) > 0:
            g_edges = g_edges + ref_node_offset
        geom_edges_ref_list.append(g_edges)
        geom_attr_ref_list.append(sample["geom_attr_ref"])

        t_edges = sample["text_edges_ref"]
        if t_edges.size(1) > 0:
            t_edges = t_edges + ref_node_offset
        text_edges_ref_list.append(t_edges)
        text_attr_ref_list.append(sample["text_attr_ref"].view(-1, 1))

        ref_batch_idx.extend([i] * n_ref)
        ref_node_offset += n_ref
        
        # Scene CLIP
        scene_clip_src_list.append(sample['scene_clip_src'])
        scene_clip_ref_list.append(sample['scene_clip_ref'])
        room_ids.append(sample["room_id"])

    return {
        "node_feats_src": torch.cat(node_feats_src_list, dim=0),
        "geom_edges_src": torch.cat(geom_edges_src_list, dim=1),
        "geom_attr_src": torch.cat(geom_attr_src_list, dim=0),
        "text_edges_src": torch.cat(text_edges_src_list, dim=1),
        "text_attr_src": torch.cat(text_attr_src_list, dim=0),

        "node_feats_ref": torch.cat(node_feats_ref_list, dim=0),
        "geom_edges_ref": torch.cat(geom_edges_ref_list, dim=1),
        "geom_attr_ref": torch.cat(geom_attr_ref_list, dim=0),
        "text_edges_ref": torch.cat(text_edges_ref_list, dim=1),
        "text_attr_ref": torch.cat(text_attr_ref_list, dim=0),

        "src_batch": torch.tensor(src_batch_idx, dtype=torch.long),
        "ref_batch": torch.tensor(ref_batch_idx, dtype=torch.long),
        "batch_size": batch_size,
        
        # Scene CLIP (512D per graph)
        "scene_clip_src": torch.stack(scene_clip_src_list),
        "scene_clip_ref": torch.stack(scene_clip_ref_list),
        "room_ids": room_ids,
    }


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = DualSceneGraphDataset(
        dataset_dir=args.dataset_dir,
        metadata_path=args.metadata_path,
        augment_ratio=0.0
    )
    
    print(f"\n{'='*70}")
    print("ARCHITECTURE")
    print(f"{'='*70}")
    print(f"✓ Node features: 518D (centroid + color + node_CLIP)")
    print(f"✓ Edge features: 64D learned embeddings")
    print(f"✓ Scene CLIP: Added AFTER GNN (fusion layer)")
    print(f"✓ Subgraph aug: 50% of time")
    print(f"{'='*70}\n")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_graph_batch_with_scene_clip
    )
    
    # Build model
    num_relations = max(dataset.rel2id.values()) + 1
    
    # Learned relation embeddings (64D)
    rel_embeddings = nn.Embedding(num_relations, 64).to(device)
    nn.init.normal_(rel_embeddings.weight, mean=0, std=0.1)
    
    # Base GNN model
    base_model = DualSceneAligner(
        node_input_dim=518,  # ← 518D (no scene CLIP!)
        relation_dim=64,     # ← 64D learned embeddings
        hidden_dim=256,      # Output dimension
        rel_clip_matrix=rel_embeddings.weight,  # Use learned embeddings
        dropout=0.2  # IMPORTANT: Prevent scene CLIP from dominating
    ).to(device)
    
    # Wrap with scene CLIP fusion
    model = DualSceneAlignerWithSceneCLIPFusion(
        base_model=base_model,
        scene_clip_dim=512,
        hidden_dim=256
    ).to(device)
    
    # Initialize fusion layer properly to prevent collapse
    for m in model.fusion.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)  # Small initialization
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Loss and optimizer
    loss_fn = SupervisedContrastiveLoss(temperature=0.2)  # Higher temp for stability
    
    # Optimize both GNN and relation embeddings
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(rel_embeddings.parameters()),
        lr=args.lr * 0.5,  # Lower LR to prevent collapse
        weight_decay=1e-3   # Stronger regularization
    )
    
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: max(0.1, 1.0 - step / (args.epochs * len(dataloader))))
    
    # Training loop
    print("Starting training...\n")
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for batch in dataloader:
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward with scene CLIP fusion
            out = model(
                batch,
                scene_clip_src=batch['scene_clip_src'],
                scene_clip_ref=batch['scene_clip_ref']
            )
            # out = base_model(batch)
            
            src_emb = out['src_emb']
            ref_emb = out['ref_emb']
            
            # Combine for contrastive loss
            all_embeddings = torch.cat([src_emb, ref_emb], dim=0)
            
            # FIXED: Create labels that allow for negative pairs
            # Each src should match its corresponding ref, but NOT other refs
            batch_size = batch['batch_size']
            
            # Create labels: [0, 1, 2, 3, 0, 1, 2, 3]
            # This means src_0 matches ref_0 (both labeled 0), but NOT ref_1, ref_2, ref_3
            src_labels = torch.arange(batch_size, device=device)
            ref_labels = torch.arange(batch_size, device=device)
            room_labels = torch.cat([src_labels, ref_labels], dim=0)
            
            # ------------------------------------------------------------
            # HARD NEGATIVE MINING (Scene-CLIP based)
            # ------------------------------------------------------------
            with torch.no_grad():
                scene_sim = F.cosine_similarity(
                    batch['scene_clip_src'].unsqueeze(1),  # [B,1,512]
                    batch['scene_clip_ref'].unsqueeze(0),  # [1,B,512]
                    dim=-1
                )  # [B,B]

                # exclude true positives
                scene_sim.fill_diagonal_(-1)

                # for each src_i, pick most similar wrong ref_j
                hard_neg_idx = scene_sim.argmax(dim=1)  # [B]
                
            
            # Loss
            loss_contrastive = loss_fn(
                all_embeddings,
                room_labels,
                hard_neg_idx=hard_neg_idx,
                hard_neg_weight=3.0  # start with 2.0–3.0
            )

            # Variance regularization to prevent collapse
            # Encourage embeddings to spread out
            src_std = src_emb.std(dim=0).mean()
            ref_std = ref_emb.std(dim=0).mean()
            var_loss = torch.relu(1.0 - src_std) + torch.relu(1.0 - ref_std)
            
            # Total loss
            loss = loss_contrastive + 0.5 * var_loss  # Balance contrastive + variance
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent collapse
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(rel_embeddings.parameters(), max_norm=1.0)
            
            # Gradient norm (after clipping)
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Logging
            if global_step % args.log_every == 0:
                with torch.no_grad():
                    # DEBUG: Check if src and ref scenes are actually different
                    scene_clip_src_norm = batch['scene_clip_src']
                    scene_clip_ref_norm = batch['scene_clip_ref']
                    scene_diff = (scene_clip_src_norm - scene_clip_ref_norm).abs().mean().item()
                    scene_cosine = F.cosine_similarity(scene_clip_src_norm, scene_clip_ref_norm, dim=-1).mean().item()
                    
                    # Metrics
                    src_norm_emb = F.normalize(src_emb, dim=-1)
                    ref_norm_emb = F.normalize(ref_emb, dim=-1)
                    
                    cross_sim = src_norm_emb @ ref_norm_emb.T
                    
                    # Positive pairs: diagonal elements (src_i matches ref_i)
                    pos_sim = cross_sim.diag().mean().item()
                    
                    # Negative pairs: off-diagonal elements
                    batch_size = cross_sim.shape[0]
                    mask = ~torch.eye(batch_size, dtype=torch.bool, device=cross_sim.device)
                    neg_sim = cross_sim[mask].mean().item()
                    
                    num_positives = batch_size
                    num_negatives = batch_size * (batch_size - 1)
                    
                    separation = pos_sim - neg_sim
                    
                    src_var = src_emb.var(dim=0).mean().item()
                    ref_var = ref_emb.var(dim=0).mean().item()

                print(f"\n{'='*70}")
                print(f"[Epoch {epoch}] Step {global_step}")
                print(f"{'='*70}")
                print(f"  🔍 DATA VERIFICATION:")
                print(f"    Scene CLIP L1 diff: {scene_diff:.6f} (should be >0.01)")
                print(f"    Scene CLIP cosine: {scene_cosine:.4f} (should be <0.95 for different scans)")
                if scene_cosine > 0.98:
                    print(f"    ❌ WARNING: Src/Ref scenes TOO SIMILAR - may be same scan!")
                elif scene_diff < 0.001:
                    print(f"    ❌ WARNING: Scene CLIP embeddings identical - data loading issue!")
                else:
                    print(f"    ✓ Scenes are different")
                print(f"  ---")
                print(f"  Total Loss       = {loss.item():.4f}")
                print(f"  Contrastive Loss = {loss_contrastive.item():.4f}")
                print(f"  Variance Loss    = {var_loss.item():.4f}")
                print(f"  ---")
                print(f"  📊 VARIANCE & STD:")
                print(f"    Var(src, ref) = {src_var:.4f}, {ref_var:.4f}")
                print(f"  ---")
                print(f"  🎯 CONTRASTIVE QUALITY:")
                print(f"    Positives: {num_positives} pairs, avg sim: {pos_sim:.3f}")
                print(f"    Negatives: {num_negatives} pairs, avg sim: {neg_sim:.3f}")
                print(f"    Separation: {separation:.3f}")
                
                if separation > 0.4:
                    print(f"    ⭐⭐ EXCELLENT! Model discriminating well!")
                elif separation > 0.3:
                    print(f"    ⭐ GOOD! Learning progressing!")
                elif separation > 0.15:
                    print(f"    ⚠️  Moderate - keep training")
                else:
                    print(f"    ❌ Poor separation - check for collapse")
                
                if src_var > 0.5 and ref_var > 0.5:
                    print(f"  ✓ Healthy variance!")
                elif src_var < 0.3 or ref_var < 0.3:
                    print(f"  ⚠️  Low variance - risk of collapse")
                    print(f"      → Try increasing dropout or batch size")
                
                print(f"  ---")
                print(f"  Grad norm = {grad_norm:.4f}")
                print(f"  LR = {scheduler.get_last_lr()[0]:.6f}")
                print(f"{'='*70}\n")
        
        # Epoch summary
        print(f"\n{'='*70}")
        print(f"Epoch {epoch} Summary")
        print(f"{'='*70}")
        print(f"  Avg Loss = {epoch_loss/len(dataloader):.4f}")
        print(f"{'='*70}\n")
        if epoch < 3:
            import numpy as np
            # Analyze subgraph sizes in this epoch
            batch_node_counts_src = []
            batch_node_counts_ref = []
            subgraph_samples = []
            for batch_idx, batch in enumerate(dataloader):
                for i in range(batch['batch_size']):
                    # Source
                    src_mask = (batch['src_batch'] == i)
                    num_src_nodes = src_mask.sum().item()
                    batch_node_counts_src.append(num_src_nodes)
                    
                    # Reference
                    ref_mask = (batch['ref_batch'] == i)
                    num_ref_nodes = ref_mask.sum().item()
                    batch_node_counts_ref.append(num_ref_nodes)
                    
                    # Record sample details
                    if epoch < 3:
                        print(f"\n  📊 SUBGRAPH STATISTICS:")
                        print(f"    SRC node counts: min={min(batch_node_counts_src)}, max={max(batch_node_counts_src)}, avg={np.mean(batch_node_counts_src):.1f}")
                        print(f"    REF node counts: min={min(batch_node_counts_ref)}, max={max(batch_node_counts_ref)}, avg={np.mean(batch_node_counts_ref):.1f}")
                        print(f"    Total batches: {len(batch_node_counts_src) // batch['batch_size']}")
            
            # Check if we're seeing variety (sign of subgraph aug working)
            unique_src = len(set(batch_node_counts_src))
            unique_ref = len(set(batch_node_counts_ref))
            print(f"    Unique SRC sizes: {unique_src}")
            print(f"    Unique REF sizes: {unique_ref}")
            
            if unique_src > 5 and unique_ref > 5:
                print(f"    ✓ Good variety - subgraph augmentation working!")
            elif unique_src <= 2 and unique_ref <= 2:
                print(f"    ❌ NO VARIETY - subgraph augmentation NOT working!")
                print(f"       All graphs have same size - check _create_subgraph")
            else:
                print(f"    ⚠️  Some variety but could be better")
            
            # Print detailed samples
            epoch_samples = [s for s in subgraph_samples if s['epoch'] == epoch]
            if epoch_samples:
                print(f"\n  🔍 DETAILED SUBGRAPH SAMPLES (Epoch {epoch}):")
                for i, sample in enumerate(epoch_samples[:6]):  # Show first 6
                    print(f"\n    Sample {i+1} (Batch {sample['batch']}, Index {sample['sample_idx']}):")
                    print(f"      Scene: {sample['scene_id'][:40]}")
                    print(f"      Nodes: {sample['num_nodes']}, Edges: {sample['num_edges']}")
                    print(f"      Objects: {', '.join(sample['labels'][:10])}")
                    if len(sample['labels']) > 10:
                        print(f"      ... and {len(sample['labels']) - 10} more")
                    
                    # Check if this looks like a subgraph or full scene
                    if sample['num_nodes'] < 10:
                        print(f"      → Likely SUBGRAPH (small)")
                    elif sample['num_nodes'] > 18:
                        print(f"      → Likely FULL SCENE (large)")
                    else:
                        print(f"      → Medium-sized graph")
        
        
        # Save checkpoint
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': base_model.state_dict(),
                'fusion_state_dict': model.fusion.state_dict(),
                'rel_embeddings': rel_embeddings.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss/len(dataloader)
            }, f"{args.save_dir}/model_epoch_{epoch}.pth")
            print(f"✓ Saved checkpoint: epoch_{epoch}.pth\n")
    
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--save_dir", default="checkpoints_518d_fusion")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=10)
    
    args = parser.parse_args()
    train(args)