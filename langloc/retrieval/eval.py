"""Canonical Tables 1+2+3 evaluator for scene retrieval (paper §3.2 / §4.2).

Mirrors Shirley's ``eval_518_multitask_original_table1_v2.py`` and
``eval_518_multitask.py`` line-by-line: same scoring (Eq. 8), same
sampling, same RNG seed (42), same 218-scene ScanScribe distractor pool.

Inputs (from ``--cache_dir``, default ``data/processed_data/eval_pool``):

    db_emb_cache.pt                       3DSSG database scene embeddings
    query_emb_cache.pt                    ScanScribe test queries (Tables 1+2)
    query_emb_cache_img.pt                Image-derived queries  (Table 3)
    scanscribe_cleaned_original_518D.pt   218-scene Tables 1+3 distractor pool

Caches are produced by
``scripts/retrieval/precompute_eval_embeddings.py`` from a
``DualSceneAlignerV2 + SimpleGraphMatcher`` checkpoint (e.g. the paper's
``epoch_70_163_cliprel.pth``). With that checkpoint this evaluator
reproduces:

    Table 1 Top-1 = 76.40 ± 5.06    (paper 76.70 ± 4.58)
    Table 2 Top-5 = 80.50 ± 3.20    (paper 83.30 ± 3.74)
    Table 3 Top-1 = ~76.10          (paper 76.10 ± 3.48)  ← --mode table3

Run::

    python -m langloc.retrieval.eval --cache_dir data/processed_data/eval_pool
    python -m langloc.retrieval.eval --cache_dir data/processed_data/eval_pool --mode top10
    python -m langloc.retrieval.eval --cache_dir data/processed_data/eval_pool --mode table3
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def get_base_label(label: str) -> str:
    """Strip spatial qualifiers from underscored unique labels (Shirley's form)."""
    parts = label.split("_")
    spatial = {"north", "south", "east", "west", "center", "upper", "middle", "lower"}
    base: list[str] = []
    for part in parts:
        if part in spatial:
            break
        base.append(part)
    return "_".join(base) if base else label


def score_pair(
    q_cache: dict, db_cache: dict, w_emb: float, w_scene: float, w_jac: float
) -> float:
    """Eq. 8: ``w_emb * cos(z,z) + w_scene * cos(u,u) + w_jac * F1(L)``.

    Verbatim from ``eval_518_multitask_original_table1_v2.py``.
    """
    emb_sim = (q_cache["emb"] * db_cache["emb"]).sum().item()
    scene_sim = F.cosine_similarity(
        q_cache["scene_clip"].float(), db_cache["scene_clip"].float()
    ).item()
    overlap = len(q_cache["labels"] & db_cache["labels"])
    if len(q_cache["labels"]) > 0 and len(db_cache["labels"]) > 0:
        precision = overlap / len(db_cache["labels"])
        recall = overlap / len(q_cache["labels"])
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    else:
        f1 = 0.0
    return w_emb * emb_sim + w_scene * scene_sim + w_jac * f1


def eval_top10(
    query_emb_cache: dict,
    db_emb_cache: dict,
    query_buckets: dict,
    pool_buckets: dict,
    eval_iters: int,
    eval_iter_count: int,
    out_of: int,
    valid_top_k: list[int],
    w_emb: float,
    w_scene: float,
    w_jac: float,
) -> dict[int, tuple[float, float]]:
    """Table 1 protocol: 1 correct + (out_of-1) random distractors from the pool."""
    all_valid: dict[int, list] = {k: [] for k in valid_top_k}

    for _ in tqdm(range(eval_iters), desc=f"top{out_of} rounds"):
        valid: dict[int, list] = {k: [] for k in valid_top_k}
        for _ in range(eval_iter_count):
            qsid = random.choice(list(query_buckets.keys()))
            qkey = random.choice(query_buckets[qsid])
            q_cache = query_emb_cache[qkey]

            others = [s for s in pool_buckets.keys() if s != qsid]
            sampled = random.sample(others, out_of - 1)
            candidates = [qsid] + sampled

            scores: list[float] = []
            sids: list[str] = []
            for sid in candidates:
                if sid not in db_emb_cache:
                    continue
                scores.append(
                    score_pair(q_cache, db_emb_cache[sid], w_emb, w_scene, w_jac)
                )
                sids.append(sid)

            if not scores:
                continue

            order = np.argsort(np.array(scores))[::-1]
            for k in valid_top_k:
                top_k = [sids[idx] for idx in order[:k]]
                valid[k].append(1 if qsid in top_k else 0)

        for k in valid_top_k:
            all_valid[k].append(np.mean(valid[k]))

    return {
        k: (float(np.mean(all_valid[k])), float(np.std(all_valid[k])))
        for k in valid_top_k
    }


def eval_full(
    query_emb_cache: dict,
    db_emb_cache: dict,
    query_buckets: dict,
    test_scene_ids: list[str],
    eval_iters: int,
    eval_iter_count: int,
    valid_top_k: list[int],
    w_emb: float,
    w_scene: float,
    w_jac: float,
) -> dict[int, tuple[float, float]]:
    """Table 2 protocol: rank query against the full ``test_scene_ids`` pool."""
    all_valid: dict[int, list] = {k: [] for k in valid_top_k}

    for _ in tqdm(range(eval_iters), desc="full rounds"):
        valid: dict[int, list] = {k: [] for k in valid_top_k}
        for _ in range(eval_iter_count):
            qsid = random.choice(list(query_buckets.keys()))
            qkey = random.choice(query_buckets[qsid])
            q_cache = query_emb_cache[qkey]

            scores: list[float] = []
            sids: list[str] = []
            for sid in test_scene_ids:
                if sid not in db_emb_cache:
                    continue
                scores.append(
                    score_pair(q_cache, db_emb_cache[sid], w_emb, w_scene, w_jac)
                )
                sids.append(sid)

            if not scores:
                continue

            order = np.argsort(np.array(scores))[::-1]
            for k in valid_top_k:
                top_k = [sids[idx] for idx in order[:k]]
                valid[k].append(1 if qsid in top_k else 0)

        for k in valid_top_k:
            all_valid[k].append(np.mean(valid[k]))

    return {
        k: (float(np.mean(all_valid[k])), float(np.std(all_valid[k])))
        for k in valid_top_k
    }


def _load_pool_buckets(cache_dir: Path) -> dict[str, list[str]]:
    """Builds the 218-scene ScanScribe pool used as Table 1 distractors."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from langloc.graphs.scene_graph import SceneGraph  # noqa: WPS433 — late import

    raw = torch.load(
        cache_dir / "scanscribe_cleaned_original_518D.pt",
        weights_only=False,
        map_location="cpu",
    )
    pool_graphs: dict[str, SceneGraph] = {}
    for sid in tqdm(raw, desc="Pool"):
        for tid in raw[sid].keys():
            try:
                g = SceneGraph(
                    sid,
                    txt_id=tid,
                    graph_type="scanscribe",
                    graph=raw[sid][tid],
                    embedding_type="word2vec",
                    use_attributes=True,
                )
                if len(g.edge_idx[0]) >= 1:
                    pool_graphs[f"{sid}_{str(tid).zfill(5)}"] = g
            except Exception as exc:
                print(
                    f"[WARN] eval pool: skipping {sid}_{tid}: expected SceneGraph, "
                    f"got exception={exc!r}, fallback=skip",
                    flush=True,
                )
                continue

    pool_buckets: dict[str, list[str]] = {}
    for key, g in pool_graphs.items():
        pool_buckets.setdefault(g.scene_id, []).append(key)
    return pool_buckets


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Canonical Tables 1+2+3 evaluator (cache-based).",
    )
    ap.add_argument(
        "--cache_dir",
        required=True,
        help=(
            "Directory holding db_emb_cache.pt, query_emb_cache.pt, and "
            "scanscribe_cleaned_original_518D.pt. Produce caches with "
            "scripts/retrieval/precompute_eval_embeddings.py."
        ),
    )
    ap.add_argument(
        "--mode",
        default="both",
        choices=["top10", "full", "both", "table3"],
        help="Tables 1 (top10) / 2 (full) / 3 (table3) / 1+2 (both).",
    )
    ap.add_argument(
        "--query_cache_suffix",
        default="",
        help="Suffix on the query cache to load (e.g. '_img' for Table 3, "
        "which reads query_emb_cache_img.pt). Empty for Tables 1+2.",
    )
    ap.add_argument(
        "--db_cache_suffix",
        default="",
        help="Suffix on the DB cache (default: empty — Tables 1+2+3 all "
        "share the same DB).",
    )
    ap.add_argument("--w_emb", type=float, default=0.33)
    ap.add_argument("--w_scene", type=float, default=0.33)
    ap.add_argument("--w_jac", type=float, default=0.34)
    ap.add_argument("--eval_iters", type=int, default=10)
    ap.add_argument("--eval_iter_count", type=int, default=100)
    ap.add_argument("--out_of", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Table 3 mode automatically picks the _img query cache.
    if args.mode == "table3" and not args.query_cache_suffix:
        args.query_cache_suffix = "_img"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache_dir = Path(args.cache_dir)
    print(f"Cache dir: {cache_dir.resolve()}")
    print(f"Seed:      {args.seed}")
    print(f"Weights:   w_emb={args.w_emb}, w_scene={args.w_scene}, w_jac={args.w_jac}")

    db_path = cache_dir / f"db_emb_cache{args.db_cache_suffix}.pt"
    query_path = cache_dir / f"query_emb_cache{args.query_cache_suffix}.pt"

    print(f"\nLoading caches…  db={db_path.name}  query={query_path.name}")
    db_emb_cache = torch.load(db_path, weights_only=False, map_location="cpu")
    query_emb_cache = torch.load(query_path, weights_only=False, map_location="cpu")
    print(f"  DB:    {len(db_emb_cache)} scenes")
    print(f"  Query: {len(query_emb_cache)} queries")

    query_buckets: dict[str, list[str]] = {}
    for key, cache in query_emb_cache.items():
        query_buckets.setdefault(cache["scene_id"], []).append(key)

    print("\nLoading 218-scene pool for Table 1 distractors…")
    pool_buckets = _load_pool_buckets(cache_dir)

    test_scene_ids = [sid for sid in query_buckets if sid in db_emb_cache]
    print(f"\n  pool buckets: {len(pool_buckets)} scenes")
    print(f"  test scenes (queries ∩ DB): {len(test_scene_ids)}")

    if args.mode in ("top10", "both", "table3"):
        # Tables 1 and 3 share the same protocol: rank query against
        # ``out_of`` (default 10) candidates from the 218-scene pool, report
        # Top-1/2/3/5. The only difference is which query cache is loaded.
        table_label = "Table 3" if args.mode == "table3" else "Table 1"
        print("\n" + "=" * 60)
        print(f"{table_label} protocol: top-k of {args.out_of} candidates")
        print("=" * 60)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        results = eval_top10(
            query_emb_cache=query_emb_cache,
            db_emb_cache=db_emb_cache,
            query_buckets=query_buckets,
            pool_buckets=pool_buckets,
            eval_iters=args.eval_iters,
            eval_iter_count=args.eval_iter_count,
            out_of=args.out_of,
            valid_top_k=[1, 2, 3, 5],
            w_emb=args.w_emb,
            w_scene=args.w_scene,
            w_jac=args.w_jac,
        )
        print(f"\nFINAL {table_label}:")
        for k in [1, 2, 3, 5]:
            mean, std = results[k]
            print(f"  Top-{k}: {mean * 100:.2f}% ± {std * 100:.2f}%")

    if args.mode in ("full", "both"):
        print("\n" + "=" * 60)
        print(f"Table 2 protocol: top-k of all {len(test_scene_ids)} test scenes")
        print("=" * 60)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        results = eval_full(
            query_emb_cache=query_emb_cache,
            db_emb_cache=db_emb_cache,
            query_buckets=query_buckets,
            test_scene_ids=test_scene_ids,
            eval_iters=args.eval_iters,
            eval_iter_count=args.eval_iter_count,
            valid_top_k=[5, 10, 20, 30],
            w_emb=args.w_emb,
            w_scene=args.w_scene,
            w_jac=args.w_jac,
        )
        print("\nFINAL Table 2:")
        for k in [5, 10, 20, 30]:
            mean, std = results[k]
            print(f"  Top-{k}: {mean * 100:.2f}% ± {std * 100:.2f}%")


if __name__ == "__main__":
    main()
