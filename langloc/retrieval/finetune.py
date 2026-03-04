"""
Fine-tune DualSceneAligner on ScanScribe-3DSSG cross-domain pairs.

Loads a pretrained checkpoint and fine-tunes with low learning rate
on ScanScribe text graphs paired with 3DSSG scene graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os

import hydra
from omegaconf import DictConfig

from langloc.retrieval.datasets.scanscribe_3dssg_dataset import ScanScribe3DSSGDataset
from langloc.retrieval.models.dual_scene_aligner import DualSceneAligner
from langloc.retrieval.train_dual_scene import InfoNCELoss, collate_fn


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    rcfg = cfg.retrieval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    assert rcfg.get("scanscribe_pt"), "Set retrieval.scanscribe_pt"
    assert rcfg.get("dssg_json_dir"), "Set retrieval.dssg_json_dir"
    assert rcfg.pretrained_checkpoint, "Set retrieval.pretrained_checkpoint"

    dataset = ScanScribe3DSSGDataset(
        scanscribe_pt_path=rcfg.scanscribe_pt,
        dssg_json_dir=rcfg.dssg_json_dir,
        negative_ratio=0.5,
    )
    dataloader = DataLoader(
        dataset, batch_size=rcfg.get("finetune_batch_size", 8),
        shuffle=True, drop_last=True,
        num_workers=4, collate_fn=collate_fn,
    )

    model = DualSceneAligner(
        node_input_dim=rcfg.node_input_dim,
        hidden_dim=rcfg.hidden_dim,
        dropout=rcfg.dropout,
    ).to(device)

    # Load pretrained checkpoint
    ckpt = torch.load(rcfg.pretrained_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded pretrained weights from {rcfg.pretrained_checkpoint}")

    finetune_lr = rcfg.get("finetune_lr", 1e-5)
    finetune_epochs = rcfg.get("finetune_epochs", 30)

    loss_fn = InfoNCELoss(temperature=rcfg.temperature).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=finetune_lr, weight_decay=rcfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)

    os.makedirs(rcfg.save_dir, exist_ok=True)

    for epoch in range(finetune_epochs):
        model.train()
        epoch_losses, epoch_seps = [], []

        for batch in dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            out = model(batch)
            bs = batch["batch_size"]
            is_pos = batch["is_positive"]

            src_labels = torch.arange(bs, device=device)
            ref_labels = torch.arange(bs, device=device).clone()
            ref_labels[~is_pos] = torch.arange(
                bs, bs + (~is_pos).sum(), device=device
            )
            labels = torch.cat([src_labels, ref_labels])
            all_emb = torch.cat([out["src_emb"], out["ref_emb"]])

            loss = loss_fn(all_emb, labels)

            optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            with torch.no_grad():
                s = F.normalize(out["src_emb"], dim=-1)
                r = F.normalize(out["ref_emb"], dim=-1)
                cross = s @ r.T
                pos_sim = cross.diag()[is_pos].mean().item() if is_pos.sum() > 0 else 0
                neg_sim = cross.diag()[~is_pos].mean().item() if (~is_pos).sum() > 0 else 0
                epoch_losses.append(loss.item())
                epoch_seps.append(pos_sim - neg_sim)

        scheduler.step()
        print(f"Epoch {epoch}: loss={np.mean(epoch_losses):.4f} "
              f"sep={np.mean(epoch_seps):.3f}")

        if (epoch + 1) % 5 == 0:
            path = os.path.join(rcfg.save_dir, f"finetune_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            }, path)
            print(f"Saved: {path}")

    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()
