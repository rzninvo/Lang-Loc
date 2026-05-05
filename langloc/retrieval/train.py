"""Canonical training for ``DualSceneAlignerV2 + SimpleGraphMatcher``.

Mirrors Shirley's ``train_contrastive_2.py`` (paper §4): symmetric InfoNCE on
positive (src, ref) graph pairs from ``DualSceneGraphDataset`` (room-grouped
via 3RScan.json). The dataset class already pairs same-room scenes (or
random fallbacks) and supports subgraph augmentation; we treat the resulting
src/ref pair as a positive for a CLIP-style symmetric InfoNCE loss with
``τ = 0.1`` and rely on in-batch negatives.

Hyperparameters match the ``epoch_70_163_cliprel.pth`` checkpoint:

    epochs       = 70
    batch_size   = 16
    optimizer    = AdamW(lr=1e-3, weight_decay=1e-4)
    schedule     = 10% linear warmup → cosine to 0
    temperature  = 0.1
    seed         = 42

Saves ``epoch_<N>.pth`` every ``--save_every`` epochs and ``best.pth`` (lowest
moving-average loss) into ``--save_dir``.

Run::

    python -m langloc.retrieval.train \\
        --dataset_dir   data/processed_data/scene_graph_clip \\
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


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism: cudnn.deterministic + benchmark off; warn-only because
    # GATv2's scatter ops do not have fully deterministic CUDA kernels.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] set_seed: expected deterministic_algorithms support, "
              f"got={exc!r}, fallback=non-deterministic kernels", flush=True)


def worker_init_fn(worker_id: int) -> None:
    """Per-worker reseed so ``random.choice`` in __getitem__ is reproducible."""
    base_seed = torch.initial_seed() % (2 ** 31)
    seed = (base_seed + worker_id) % (2 ** 31)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Collate: batch variable-size graphs by concatenating tensors and shifting
# edge indices into a single big-graph index space, plus a per-node
# ``*_batch`` vector that ``scatter_mean`` reads inside the model.
# ---------------------------------------------------------------------------
def graph_pair_collate(samples: list[dict]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
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
        out[f"geom_edges_{branch}"] = torch.cat(geom_edges, dim=1) if geom_edges and any(e.numel() > 0 for e in geom_edges) else torch.zeros(2, 0, dtype=torch.long)
        out[f"geom_attr_{branch}"] = torch.cat(geom_attr, dim=0) if geom_attr else torch.zeros(0, 8)
        out[f"text_edges_{branch}"] = torch.cat(text_edges, dim=1) if text_edges and any(e.numel() > 0 for e in text_edges) else torch.zeros(2, 0, dtype=torch.long)
        out[f"text_attr_{branch}"] = torch.cat(text_attr, dim=0) if text_attr else torch.zeros(0, 512)
        out[f"scene_clip_{branch}"] = torch.stack(scene_clip, dim=0)
        out[f"{branch}_batch"] = torch.cat(batch_idx, dim=0)
    return out


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Loss: symmetric InfoNCE on (src_emb, ref_emb), τ = 0.1
# ---------------------------------------------------------------------------
def info_nce_symmetric(
    src_emb: torch.Tensor, ref_emb: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    src = F.normalize(src_emb, dim=-1)
    ref = F.normalize(ref_emb, dim=-1)
    logits = src @ ref.t() / temperature           # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    with torch.no_grad():
        acc_s2r = (logits.argmax(dim=-1) == labels).float().mean()
        acc_r2s = (logits.t().argmax(dim=-1) == labels).float().mean()
        acc = 0.5 * (acc_s2r + acc_r2s)
    return loss, acc


# ---------------------------------------------------------------------------
# Schedule: warmup → cosine
# ---------------------------------------------------------------------------
def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_total_steps: int
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = (step - num_warmup_steps) / float(max(1, num_total_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

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
        clip_model=clip_model,
        device=str(device),
        clip_model_name=args.clip_model,
    )

    # Free CLIP from device memory; relation cache is already populated.
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
        persistent_workers=False,  # see worker_init_fn — re-seed each epoch
    )

    print(f"[TRAIN] dataset={len(dataset)} graphs, "
          f"batches/epoch={len(loader)} (batch_size={args.batch_size})", flush=True)

    # Eval pipeline (precompute_eval_embeddings.py) hardcodes hidden_dim=256;
    # diverging here would silently produce a checkpoint that fails strict-load.
    assert args.hidden_dim == 256, (
        f"hidden_dim must be 256 for paper-faithful eval; got {args.hidden_dim}"
    )

    base_model = DualSceneAlignerV2(
        node_input_dim=args.node_input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model = SimpleGraphMatcher(base_model, scene_clip_dim=512, hidden_dim=args.hidden_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[TRAIN] model params={n_params:,}", flush=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[TRAIN] total_steps={total_steps}, warmup_steps={warmup_steps}, "
          f"lr={args.lr}, wd={args.weight_decay}, τ={args.temperature}", flush=True)

    best_loss = float("inf")
    global_step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0
        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch)
            loss, acc = info_nce_symmetric(
                out["src_emb"], out["ref_emb"], temperature=args.temperature
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_acc += acc.item()
            n_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = running_loss / n_batches
                avg_acc = running_acc / n_batches
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(
                    f"[E{epoch:03d} S{global_step:05d}] loss={avg_loss:.4f} "
                    f"acc@1={avg_acc * 100:.2f}% lr={lr_now:.2e} "
                    f"elapsed={elapsed / 60:.1f}min",
                    flush=True,
                )

        epoch_loss = running_loss / max(n_batches, 1)
        epoch_acc = running_acc / max(n_batches, 1)
        print(
            f"[E{epoch:03d}] DONE | loss={epoch_loss:.4f} acc@1={epoch_acc * 100:.2f}% "
            f"({n_batches} batches)",
            flush=True,
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = save_dir / f"epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "acc": epoch_acc,
                    "config": vars(args),
                },
                ckpt_path,
            )
            print(f"[E{epoch:03d}] saved {ckpt_path}", flush=True)

        # `last.pth` is always the most recent epoch — the script has no
        # held-out split so a "best by val" criterion would be misleading.
        last_path = save_dir / "last.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
                "acc": epoch_acc,
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Canonical training for V2+SimpleGraphMatcher.")
    ap.add_argument("--dataset_dir", default="data/processed_data/scene_graph_clip")
    ap.add_argument("--metadata_path", default="data/3RScan/3RScan.json")
    ap.add_argument("--save_dir", default="data/model_checkpoints/graph2graph/canonical_v2")

    # Architecture
    ap.add_argument("--node_input_dim", type=int, default=518)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Training (paper §4)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)

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
