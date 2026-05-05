"""Canonical training for ``DualSceneAlignerV2 + SimpleGraphMatcher``.

Reproduces the recipe that produced the published paper checkpoint:

    epochs        = 70
    batch_size    = 16
    optimizer     = AdamW(lr=lr*0.5, weight_decay=1e-4, betas=(0.9, 0.999))
    schedule      = 10% linear warmup → cosine to floor 0.1
    loss          = SimpleContrastiveLoss(temperature=0.07) on
                    [src_emb; ref_emb] with labels constructed from is_positive
    base_dropout  = 0.0       # V2 base model — fusion head has its own 0.3
    fusion_drop.  = 0.3       # already inside SimpleGraphMatcher
    grad_clip     = 1.0
    seed          = 42
    dataset       = combined_dataset_clip (100 3DSSG + ~3356 ScanScribe paraphrases)
    sampler       = negative_ratio = 0.5

The saved checkpoint filename follows the ``epoch_{N}_163_cliprel.pth``
convention (``163`` is the literal training-set scan_id count).

Run::

    python -m langloc.retrieval.train \\
        --dataset_dir   data/processed_data/combined_dataset_clip \\
        --metadata_path data/3RScan/3RScan.json \\
        --save_dir      data/model_checkpoints/graph2graph/canonical_v2
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from langloc.retrieval.datasets.dual_scene_graph_dataset import DualSceneGraphDataset
from langloc.retrieval.models.dual_scene_aligner_v2 import DualSceneAlignerV2
from langloc.retrieval.models.simple_graph_matcher import SimpleGraphMatcher


# Reproducibility helpers come from the shared utility — see
# langloc/utils/seed.py and CLAUDE.md §0 (canonical project seed = 42).
from langloc.utils.seed import set_seed, worker_init_fn  # noqa: E402


# ---------------------------------------------------------------------------
# Collate: batch variable-size graphs by concatenating tensors and shifting
# edge indices into a single big-graph index space, plus a per-node
# ``*_batch`` vector that ``scatter_mean`` reads inside the model.
# ---------------------------------------------------------------------------
def graph_pair_collate(samples: list[dict]) -> dict[str, torch.Tensor | int]:
    out: dict[str, torch.Tensor | int] = {"batch_size": len(samples)}
    is_positive: list[bool] = []
    for branch in ("src", "ref"):
        node_feats: list[torch.Tensor] = []
        geom_edges: list[torch.Tensor] = []
        geom_attr: list[torch.Tensor] = []
        text_edges: list[torch.Tensor] = []
        text_attr: list[torch.Tensor] = []
        scene_clip: list[torch.Tensor] = []
        batch_idx: list[torch.Tensor] = []

        node_offset = 0
        for i, s in enumerate(samples):
            nf = s[f"node_feats_{branch}"]
            n_nodes = nf.size(0)
            node_feats.append(nf)
            batch_idx.append(torch.full((n_nodes,), i, dtype=torch.long))

            g_e = s[f"geom_edges_{branch}"]
            geom_edges.append(g_e + node_offset if g_e.numel() > 0 else g_e)
            geom_attr.append(s[f"geom_attr_{branch}"])

            t_e = s[f"text_edges_{branch}"]
            text_edges.append(t_e + node_offset if t_e.numel() > 0 else t_e)
            text_attr.append(s[f"text_attr_{branch}"])

            scene_clip.append(s[f"scene_clip_{branch}"])
            node_offset += n_nodes

        out[f"node_feats_{branch}"] = torch.cat(node_feats, dim=0)
        out[f"geom_edges_{branch}"] = torch.cat(geom_edges, dim=1) if any(e.numel() > 0 for e in geom_edges) else torch.zeros(2, 0, dtype=torch.long)
        out[f"geom_attr_{branch}"] = torch.cat(geom_attr, dim=0) if geom_attr else torch.zeros(0, 8)
        out[f"text_edges_{branch}"] = torch.cat(text_edges, dim=1) if any(e.numel() > 0 for e in text_edges) else torch.zeros(2, 0, dtype=torch.long)
        out[f"text_attr_{branch}"] = torch.cat(text_attr, dim=0) if text_attr else torch.zeros(0, 512)
        out[f"scene_clip_{branch}"] = torch.stack(scene_clip, dim=0)
        out[f"{branch}_batch"] = torch.cat(batch_idx, dim=0)

    for s in samples:
        is_positive.append(bool(s.get("is_positive", True)))
    out["is_positive"] = torch.tensor(is_positive, dtype=torch.bool)
    return out


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


# ---------------------------------------------------------------------------
# SimpleContrastiveLoss (vectorized).
#
# Embeddings: ``[src; ref]`` of shape ``(2B, D)``.
# Labels   : ``src_labels = arange(B)``; ``ref_labels = arange(B)`` but with
#            ``ref_labels[~is_positive] += B`` so explicit-negative refs end
#            up in their own class. Pairs with matching labels are positives;
#            others are negatives. Diagonal (self) is excluded.
# ---------------------------------------------------------------------------
class SimpleContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=-1, p=2)
        sim = embeddings @ embeddings.t() / self.temperature        # (2B, 2B)

        labels_col = labels.view(-1, 1)
        pos_mask = (labels_col == labels_col.t()).float()
        pos_mask.fill_diagonal_(0)
        neg_mask = 1.0 - pos_mask
        neg_mask.fill_diagonal_(0)

        # Per-row softmax denominator over (positives + negatives), excluding self.
        exp_sim = torch.exp(sim)
        row_denom = (exp_sim * (pos_mask + neg_mask)).sum(dim=1)
        # Per-row positive sum.
        row_pos = (exp_sim * pos_mask).sum(dim=1)

        # Only rows that have at least one positive contribute.
        row_has_pos = (pos_mask.sum(dim=1) > 0).float()
        eps = 1e-12
        per_row_loss = -torch.log((row_pos + eps) / (row_denom + eps))
        valid = row_has_pos.sum().clamp(min=1.0)
        return (per_row_loss * row_has_pos).sum() / valid


# ---------------------------------------------------------------------------
# Schedule: 10% linear warmup → cosine decay with floor multiplier 0.1.
# ---------------------------------------------------------------------------
def warmup_cosine_floor(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_total_steps: int,
    floor: float = 0.1,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = (step - num_warmup_steps) / float(max(1, num_total_steps - num_warmup_steps))
        return max(floor, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TRAIN] save_dir={save_dir.resolve()}", flush=True)
    print(f"[TRAIN] device={device}, seed={args.seed}", flush=True)

    print("[TRAIN] loading CLIP for relation embeddings…", flush=True)
    clip_model, _ = clip.load(args.clip_model, device=device)

    print("[TRAIN] building dataset…", flush=True)
    dataset = DualSceneGraphDataset(
        dataset_dir=args.dataset_dir,
        metadata_path=args.metadata_path,
        negative_ratio=args.negative_ratio,
        clip_model=clip_model,
        device=str(device),
        clip_model_name=args.clip_model,
    )

    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=graph_pair_collate,
        generator=g,
        worker_init_fn=worker_init_fn,
        persistent_workers=False,
    )

    print(f"[TRAIN] dataset={len(dataset)} graphs, "
          f"batches/epoch={len(loader)} (batch_size={args.batch_size})", flush=True)

    assert args.hidden_dim == 256, (
        f"hidden_dim must be 256 to match the published checkpoint; got {args.hidden_dim}"
    )

    base_model = DualSceneAlignerV2(
        node_input_dim=args.node_input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model = SimpleGraphMatcher(base_model, scene_clip_dim=512, hidden_dim=args.hidden_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[TRAIN] model params={n_params:,}", flush=True)

    if args.pretrained_checkpoint:
        print(f"[TRAIN] loading pretrained {args.pretrained_checkpoint}", flush=True)
        sd = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(sd.get("model_state_dict", sd), strict=True)

    # Effective LR is half the CLI value (paper recipe halves the cli lr at
    # optimizer construction time so default cli lr=1e-3 → optimizer lr=5e-4).
    effective_lr = args.lr * 0.5
    optimizer = AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = args.epochs * len(loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = warmup_cosine_floor(optimizer, warmup_steps, total_steps, floor=args.cosine_floor)
    print(f"[TRAIN] total_steps={total_steps}, warmup_steps={warmup_steps}, "
          f"effective_lr={effective_lr}, wd={args.weight_decay}, τ={args.temperature}, "
          f"cosine_floor={args.cosine_floor}", flush=True)

    loss_fn = SimpleContrastiveLoss(temperature=args.temperature).to(device)

    best_loss = float("inf")
    global_step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_sep = 0.0
        n_batches = 0
        for batch in loader:
            batch = to_device(batch, device)
            B = batch["batch_size"]
            is_positive = batch["is_positive"]

            out = model(
                batch,
                scene_clip_src=batch["scene_clip_src"],
                scene_clip_ref=batch["scene_clip_ref"],
            )
            src_emb = out["src_emb"]
            ref_emb = out["ref_emb"]

            # Label scheme: src and matching-ref share the same id;
            # explicit-negative refs go to id+B so they're a distinct class.
            src_labels = torch.arange(B, device=device)
            ref_labels = torch.arange(B, device=device)
            ref_labels = ref_labels.clone()
            ref_labels[~is_positive] = ref_labels[~is_positive] + B
            labels = torch.cat([src_labels, ref_labels], dim=0)
            all_emb = torch.cat([src_emb, ref_emb], dim=0)

            loss = loss_fn(all_emb, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                src_n = F.normalize(src_emb, dim=-1)
                ref_n = F.normalize(ref_emb, dim=-1)
                cross = src_n @ ref_n.t()
                diag = cross.diag()
                pos_sim = diag[is_positive].mean().item() if is_positive.any() else 0.0
                neg_sim = diag[~is_positive].mean().item() if (~is_positive).any() else 0.0
                separation = pos_sim - neg_sim

            running_loss += loss.item()
            running_sep += separation
            n_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = running_loss / n_batches
                avg_sep = running_sep / n_batches
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(
                    f"[E{epoch:03d} S{global_step:05d}] loss={avg_loss:.4f} "
                    f"sep={avg_sep:+.3f} lr={lr_now:.2e} "
                    f"elapsed={elapsed / 60:.1f}min",
                    flush=True,
                )

        epoch_loss = running_loss / max(n_batches, 1)
        epoch_sep = running_sep / max(n_batches, 1)
        print(
            f"[E{epoch:03d}] DONE | loss={epoch_loss:.4f} sep={epoch_sep:+.3f} "
            f"({n_batches} batches)",
            flush=True,
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = save_dir / f"epoch_{epoch}_163_cliprel.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "separation": epoch_sep,
                    "config": vars(args),
                },
                ckpt_path,
            )
            print(f"[E{epoch:03d}] saved {ckpt_path}", flush=True)

        last_path = save_dir / "last.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
                "separation": epoch_sep,
                "config": vars(args),
            },
            last_path,
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss

    elapsed = time.time() - t0
    print(f"[TRAIN] DONE in {elapsed / 60:.1f} min, "
          f"min_train_loss={best_loss:.4f} → see last.pth + per-epoch ckpts",
          flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Canonical training for V2+SimpleGraphMatcher.")
    ap.add_argument("--dataset_dir", default="data/processed_data/combined_dataset_clip")
    ap.add_argument("--metadata_path", default="data/3RScan/3RScan.json")
    ap.add_argument("--save_dir", default="data/model_checkpoints/graph2graph/canonical_v2")
    ap.add_argument("--pretrained_checkpoint", default=None)

    # Architecture — V2 base uses dropout=0.0; the fusion head has its own 0.3.
    ap.add_argument("--node_input_dim", type=int, default=518)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)

    # Training (paper recipe)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="CLI lr — the optimizer uses lr*0.5 internally.")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--cosine_floor", type=float, default=0.1)
    ap.add_argument("--negative_ratio", type=float, default=0.5)

    # Misc
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--clip_model", default="ViT-B/32")
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
