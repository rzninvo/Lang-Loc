"""Inject ``scene_clip_emb`` into every JSON in ``scene_graph_clip/``.

Discovery: ``DualSceneGraphDataset.__getitem__`` reads
``scene_data['scene_clip_emb']`` and defaults to ``[0.0] * 512`` when
missing. None of the 4,675 JSONs we just built has this field, so the
model has been training with all-zero scene-CLIP for every scene. Eval
then passes a real scene-CLIP → out-of-distribution input for
``final_proj``.

Fix: compute ``scene_clip_emb = phi(create_scene_description(nodes, edges))``
once per JSON using the canonical ``create_scene_description`` from
``scripts/retrieval/utils.py`` (base labels + up to 5 relations,
deterministic ordering). Matches paper Eq. 2 modulo the relation suffix.

Run from repo root:
    python -m scripts.retrieval.inject_scene_clip_emb \
        --num_workers 8 --device cuda
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path


def _worker(
    job_q: mp.Queue,
    result_q: mp.Queue,
    dataset_dir: str,
    device: str,
    clip_model_name: str,
) -> None:
    """Loads CLIP once, drains the queue, writes scene_clip_emb in place."""
    import clip
    import torch
    from scripts.retrieval.utils import create_scene_description

    clip_model, _ = clip.load(clip_model_name, device=device)

    @torch.no_grad()
    def encode(text: str):
        tokens = clip.tokenize([text], truncate=True).to(device)
        emb = clip_model.encode_text(tokens)[0]
        emb = emb / emb.norm()
        return emb.cpu().tolist()

    base = Path(dataset_dir)
    while True:
        try:
            fname = job_q.get(timeout=2)
        except queue.Empty:
            break
        if fname is None:
            break

        path = base / fname
        try:
            with open(path) as f:
                d = json.load(f)
            if "scene_clip_emb" in d and isinstance(d["scene_clip_emb"], list) and len(d["scene_clip_emb"]) == 512:
                # Already populated — leave alone.
                result_q.put(("skipped", fname))
                continue

            nodes = d.get("nodes", {})
            edges = d.get("edges_text", [])
            if not nodes:
                # No node info → store zero (will read like the legacy default).
                d["scene_clip_emb"] = [0.0] * 512
                d["scene_description"] = ""
            else:
                desc = create_scene_description(nodes, edges, max_objects=10, max_relations=5)
                d["scene_clip_emb"] = encode(desc)
                d["scene_description"] = desc

            with open(path, "w") as f:
                json.dump(d, f)
            result_q.put(("ok", fname))
        except Exception as exc:
            traceback.print_exc()
            result_q.put(("fail", f"{fname}: {exc}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="data/processed_data/scene_graph_clip")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--clip_model", default="ViT-B/32")
    args = ap.parse_args()

    if args.device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"
    else:
        device = args.device

    base = Path(args.dataset_dir)
    files = sorted(
        f.name for f in base.glob("*.json") if f.name != "metadata.json"
    )
    total = len(files)
    print(f"[INJECT] {total} JSONs to process | workers={args.num_workers} | device={device}", flush=True)

    ctx = mp.get_context("spawn")
    job_q: mp.Queue = ctx.Queue()
    result_q: mp.Queue = ctx.Queue()
    for f in files:
        job_q.put(f)
    for _ in range(args.num_workers):
        job_q.put(None)

    workers = [
        ctx.Process(
            target=_worker,
            args=(job_q, result_q, str(base), device, args.clip_model),
            daemon=True,
        )
        for _ in range(args.num_workers)
    ]
    for w in workers:
        w.start()

    ok = skipped = fail = 0
    completed = 0
    t0 = time.time()
    while completed < total:
        try:
            status, payload = result_q.get(timeout=120)
        except queue.Empty:
            alive = sum(1 for w in workers if w.is_alive())
            if alive == 0:
                print(f"[INJECT] all workers exited early; completed={completed}/{total}", flush=True)
                break
            continue
        completed += 1
        if status == "ok":
            ok += 1
        elif status == "skipped":
            skipped += 1
        elif status == "fail":
            fail += 1
            print(f"[FAIL] {payload}", flush=True)

        if completed % 200 == 0 or completed == total:
            elapsed = time.time() - t0
            rate = completed / max(elapsed, 1e-3)
            eta = (total - completed) / max(rate, 1e-3)
            print(
                f"[{completed}/{total}] ok={ok} skipped={skipped} fail={fail} | {rate:.1f}/s | ETA {eta/60:.1f} min",
                flush=True,
            )

    for w in workers:
        w.join(timeout=10)

    elapsed = time.time() - t0
    print(f"[INJECT] DONE in {elapsed/60:.1f} min — ok={ok} skipped={skipped} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
