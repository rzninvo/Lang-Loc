"""Canonical Tables 1+2+3 evaluator for scene retrieval (paper §3.2 / §4.2).

Eq. 8 scoring with weights (0.33, 0.33, 0.34), seed=42, 10 outer × 100
inner sampling rounds. The 218-scene ScanScribe distractor pool is shared
across all three tables; Table 2 reports against the full test pool.

Table 3 uses the corrected fair-comparison protocol — queries come from
``scanscribe_text_graphs_from_image_desc_node_edge_features.pt`` (the
LLM-from-image graphs), matching the protocol the CLIP2CLIP and Text2SGM
baselines use. The published 76.10% in the paper was inadvertently
produced on the canonical ScanScribe text test set; this evaluator
reports the corrected number. See
``docs/reports/2026-05-05/19_table3_corrected_for_rebuttal.md``.

Inputs (from ``--cache_dir``, default ``data/processed_data/eval_pool``):

    db_emb_cache.pt                       3DSSG database scene embeddings
    query_emb_cache.pt                    ScanScribe text test (Tables 1+2)
    query_emb_cache_img.pt                LLM-from-image queries (Table 3)
    scanscribe_cleaned_original_518D.pt   218-scene distractor pool

Caches are produced by
``scripts/retrieval/precompute_eval_embeddings.py`` from a
``DualSceneAlignerV2 + SimpleGraphMatcher`` checkpoint. With the
published paper checkpoint, this evaluator reports:

    Table 1 Top-1 = 76.40 ± 5.06    (paper 76.70 ± 4.58)  ← matches paper
    Table 2 Top-5 = 80.20 ± 3.40    (paper 83.30 ± 3.74)  ← within noise
    Table 3 Top-1 = 62.10 ± 5.63    (paper-as-published claimed 76.10;
                                      corrected here — see report 19)

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
    """Strip spatial qualifiers (north/south/upper/etc.) from a node label."""
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


def _load_pool_buckets(cache_dir: Path, pool_filename: str) -> dict[str, list[str]]:
    """Build the ScanScribe distractor pool from ``cache_dir/<pool_filename>``.

    All three Tables share the 218-scene ``scanscribe_cleaned_original_518D.pt``
    pool by default. The 55-scene LLM-from-image pool (used by some Table 3
    variants) is also a valid choice but lands within a few pp on the same
    queries.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from langloc.graphs.scene_graph import SceneGraph  # noqa: WPS433 — late import

    raw = torch.load(
        cache_dir / pool_filename, weights_only=False, map_location="cpu",
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
        help=(
            "Tables 1 (top10) / 2 (full) / 3 (table3) / 1+2 (both). "
            "Table 3 is the corrected, fair-comparison protocol — queries "
            "from `scanscribe_text_graphs_from_image_desc_node_edge_features.pt` "
            "(LLM-from-image graphs, 55 scenes × 1 query each), matching "
            "whereami Table 4's protocol used for the CLIP2CLIP / Text2SGM "
            "baselines. See docs/reports/2026-05-05/19_table3_corrected_for_rebuttal.md "
            "for the data-mismatch fix and rebuttal context."
        ),
    )
    ap.add_argument(
        "--query_cache_suffix",
        default="",
        help="Suffix on the query cache to load (e.g. '_img' for Table 3). "
        "Empty for Tables 1+2.",
    )
    ap.add_argument(
        "--db_cache_suffix",
        default="",
        help="Suffix on the DB cache (default: empty — Tables 1+2+3 all "
        "share the same 3DSSG DB).",
    )
    ap.add_argument("--w_emb", type=float, default=0.33)
    ap.add_argument("--w_scene", type=float, default=0.33)
    ap.add_argument("--w_jac", type=float, default=0.34)
    ap.add_argument("--eval_iters", type=int, default=10)
    ap.add_argument("--eval_iter_count", type=int, default=100)
    ap.add_argument("--out_of", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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

    # Pool selection. The 218-scene cleaned-original ScanScribe pool is used
    # by both Tables 1 and 3 — for Tables 1+2 it matches the published
    # protocol; for Table 3 it gives the most-permissive, paper-protocol-
    # compatible distractor pool for the corrected (LLM-from-image)
    # comparison. Either pool size lands within a few pp of each other on
    # the same query distribution.
    pool_filename = "scanscribe_cleaned_original_518D.pt"
    label = "Table 3 (FAIR — LLM-from-image queries)" if args.mode == "table3" else "Tables 1+2"
    print(f"\nLoading 218-scene pool for {label} distractors ({pool_filename})…")
    pool_buckets = _load_pool_buckets(cache_dir, pool_filename)

    test_scene_ids = [sid for sid in query_buckets if sid in db_emb_cache]
    print(f"\n  pool buckets: {len(pool_buckets)} scenes")
    print(f"  test scenes (queries ∩ DB): {len(test_scene_ids)}")

    if args.mode in ("top10", "both", "table3"):
        if args.mode == "table3":
            table_label = "Table 3 (corrected — LLM-from-image queries, fair comparison vs Whereami baselines)"
        else:
            table_label = "Table 1"
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
