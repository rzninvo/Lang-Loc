"""
Training script for DualSceneAligner (Section 3.2).

Uses InfoNCE contrastive loss with scene-CLIP fusion built into the model.
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import clip

import hydra
from omegaconf import DictConfig

from langloc.retrieval.datasets.dual_scene_graph_dataset import DualSceneGraphDataset
from langloc.retrieval.models.dual_scene_aligner import DualSceneAligner


# ── Loss ──────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss (paper line 178)."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (2B, D) — concatenation of src and ref embeddings
            labels: (2B,) — matching labels indicate positive pairs
        """
        embeddings = F.normalize(embeddings, dim=-1, p=2)
        sim = embeddings @ embeddings.T / self.temperature

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)
        neg_mask = 1 - pos_mask
        neg_mask.fill_diagonal_(0)

        exp_sim = torch.exp(sim)
        num_pos = pos_mask.sum(dim=1)

        loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        count = 0
        for i in range(embeddings.size(0)):
            if num_pos[i] == 0:
                continue
            pos_sum = (exp_sim[i] * pos_mask[i]).sum()
            all_sum = (exp_sim[i] * (pos_mask[i] + neg_mask[i])).sum()
            loss = loss + (-torch.log(pos_sum / (all_sum + 1e-8)))
            count += 1

        return loss / max(count, 1)


# ── Collate ───────────────────────────────────────────────────

def collate_fn(batch_list):
    """Collate variable-size graphs into a single batch."""
    batch_size = len(batch_list)

    node_feats_src, geom_edges_src, geom_attr_src = [], [], []
    text_edges_src, text_attr_src = [], []
    node_feats_ref, geom_edges_ref, geom_attr_ref = [], [], []
    text_edges_ref, text_attr_ref = [], []
    src_batch_idx, ref_batch_idx = [], []
    scene_clip_src, scene_clip_ref = [], []
    is_positive_list = []

    src_offset, ref_offset = 0, 0

    for i, sample in enumerate(batch_list):
        n_src = sample["node_feats_src"].size(0)
        node_feats_src.append(sample["node_feats_src"])
        ge = sample["geom_edges_src"]
        geom_edges_src.append(ge + src_offset if ge.size(1) > 0 else ge)
        geom_attr_src.append(sample["geom_attr_src"])
        te = sample["text_edges_src"]
        text_edges_src.append(te + src_offset if te.size(1) > 0 else te)
        text_attr_src.append(sample["text_attr_src"])
        src_batch_idx.extend([i] * n_src)
        src_offset += n_src

        n_ref = sample["node_feats_ref"].size(0)
        node_feats_ref.append(sample["node_feats_ref"])
        ge = sample["geom_edges_ref"]
        geom_edges_ref.append(ge + ref_offset if ge.size(1) > 0 else ge)
        geom_attr_ref.append(sample["geom_attr_ref"])
        te = sample["text_edges_ref"]
        text_edges_ref.append(te + ref_offset if te.size(1) > 0 else te)
        text_attr_ref.append(sample["text_attr_ref"])
        ref_batch_idx.extend([i] * n_ref)
        ref_offset += n_ref

        scene_clip_src.append(sample["scene_clip_src"])
        scene_clip_ref.append(sample["scene_clip_ref"])
        is_positive_list.append(sample["is_positive"])

    return {
        "node_feats_src": torch.cat(node_feats_src, dim=0),
        "geom_edges_src": torch.cat(geom_edges_src, dim=1),
        "geom_attr_src":  torch.cat(geom_attr_src, dim=0),
        "text_edges_src": torch.cat(text_edges_src, dim=1),
        "text_attr_src":  torch.cat(text_attr_src, dim=0),
        "node_feats_ref": torch.cat(node_feats_ref, dim=0),
        "geom_edges_ref": torch.cat(geom_edges_ref, dim=1),
        "geom_attr_ref":  torch.cat(geom_attr_ref, dim=0),
        "text_edges_ref": torch.cat(text_edges_ref, dim=1),
        "text_attr_ref":  torch.cat(text_attr_ref, dim=0),
        "src_batch":      torch.tensor(src_batch_idx, dtype=torch.long),
        "ref_batch":      torch.tensor(ref_batch_idx, dtype=torch.long),
        "batch_size":     batch_size,
        "scene_clip_src": torch.stack(scene_clip_src),
        "scene_clip_ref": torch.stack(scene_clip_ref),
        "is_positive":    torch.tensor(is_positive_list, dtype=torch.bool),
    }


# ── Training ──────────────────────────────────────────────────

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    rcfg = cfg.retrieval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # CLIP model for relation embeddings
    clip_model, _ = clip.load("ViT-B/32", device=device)

    dataset = DualSceneGraphDataset(
        dataset_dir=rcfg.dataset_dir,
        metadata_path=rcfg.metadata_path,
        negative_ratio=0.5,
        clip_model=clip_model,
        device=device,
    )
    dataloader = DataLoader(
        dataset, batch_size=rcfg.batch_size,
        shuffle=True, drop_last=True,
        num_workers=0, collate_fn=collate_fn,
    )

    model = DualSceneAligner(
        node_input_dim=rcfg.node_input_dim,
        hidden_dim=rcfg.hidden_dim,
        dropout=rcfg.dropout,
    ).to(device)

    if rcfg.pretrained_checkpoint:
        ckpt = torch.load(rcfg.pretrained_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {rcfg.pretrained_checkpoint}")

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    loss_fn = InfoNCELoss(temperature=rcfg.temperature).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=rcfg.lr, weight_decay=rcfg.weight_decay,
    )

    # Warmup + cosine decay
    total_steps = rcfg.epochs * len(dataloader)
    warmup_steps = int(rcfg.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(rcfg.save_dir, exist_ok=True)
    global_step = 0

    for epoch in range(rcfg.epochs):
        model.train()
        epoch_losses, epoch_seps = [], []

        for batch_idx, batch in enumerate(dataloader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            out = model(batch)
            bs = batch["batch_size"]
            is_pos = batch["is_positive"]

            # Build labels: matching indices = positive pair
            src_labels = torch.arange(bs, device=device)
            ref_labels = torch.arange(bs, device=device).clone()
            ref_labels[~is_pos] = ref_labels[~is_pos] + bs
            labels = torch.cat([src_labels, ref_labels], dim=0)

            all_emb = torch.cat([out["src_emb"], out["ref_emb"]], dim=0)
            loss = loss_fn(all_emb, labels)

            optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            global_step += 1

            with torch.no_grad():
                s = F.normalize(out["src_emb"], dim=-1)
                r = F.normalize(out["ref_emb"], dim=-1)
                cross = s @ r.T
                pos_sim = cross.diag()[is_pos].mean().item() if is_pos.sum() > 0 else 0
                neg_sim = cross.diag()[~is_pos].mean().item() if (~is_pos).sum() > 0 else 0
                epoch_losses.append(loss.item())
                epoch_seps.append(pos_sim - neg_sim)

            if (batch_idx + 1) % rcfg.log_every == 0:
                print(f"  [{epoch}/{rcfg.epochs}] step {global_step}  "
                      f"loss={loss.item():.4f}  sep={pos_sim - neg_sim:.3f}")

        avg_loss = np.mean(epoch_losses)
        avg_sep = np.mean(epoch_seps)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}  separation={avg_sep:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % 10 == 0 or epoch == rcfg.epochs - 1:
            path = os.path.join(rcfg.save_dir, f"epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "rel2id": dataset.rel2id,
            }, path)
            print(f"Saved: {path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
