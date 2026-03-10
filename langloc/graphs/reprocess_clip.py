#!/usr/bin/env python3
"""Reprocess existing .pt graph files with CLIP embeddings.

Adds ``label_clip``, ``attributes_clip``, and ``relation_clip`` keys to each
scene in an existing processed graph ``.pt`` file.  Existing keys (e.g.
word2vec-based embeddings) are preserved.

Usage::

    python -m langloc.graphs.reprocess_clip \
        --input_pt  data/processed/3dssg/3dssg_graphs_processed_edgelists_relationembed.pt \
        --output_pt data/processed/3dssg/clip_full_3dssg_graphs.pt \
        --batch_size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from langloc.graphs.create_text_embeddings import (
    create_embedding_clip,
    create_embeddings_clip_batch,
)


def _to_list(tensor_or_list: object) -> object:
    """Convert a tensor to a Python list for serialisation."""
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list.tolist()
    if hasattr(tensor_or_list, "tolist") and not isinstance(tensor_or_list, str):
        return tensor_or_list.tolist()
    return tensor_or_list


def _batch_embed_texts(
    texts: List[str],
    batch_size: int,
    cache: Dict[str, List[float]],
) -> List[List[float]]:
    """Embed a list of texts with CLIP, using a cache to avoid duplicates."""
    results: list[list[float] | None] = [None] * len(texts)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, t in enumerate(texts):
        key = t.strip().lower()
        if key in cache:
            results[i] = cache[key]
        else:
            uncached_indices.append(i)
            uncached_texts.append(t)

    for start in range(0, len(uncached_texts), batch_size):
        batch = uncached_texts[start : start + batch_size]
        embeddings = create_embeddings_clip_batch(batch)  # (B, 512)
        for j, emb in enumerate(embeddings):
            idx = uncached_indices[start + j]
            emb_list = _to_list(emb)
            if isinstance(emb_list, str):
                emb_list = create_embedding_clip(batch[j]).tolist()
            results[idx] = emb_list
            key = uncached_texts[start + j].strip().lower()
            cache[key] = emb_list

    return results  # type: ignore[return-value]


def reprocess(input_pt: Path, output_pt: Path, batch_size: int) -> None:
    """Load a .pt graph file, add CLIP embeddings, and save the result.

    Args:
        input_pt: Path to the existing ``.pt`` graph file.
        output_pt: Destination path for the augmented file.
        batch_size: Number of texts to embed in one forward pass.
    """
    print(f"Loading {input_pt} ...")
    all_scenes = torch.load(input_pt, map_location="cpu", weights_only=False)
    print(f"  {len(all_scenes)} scenes loaded.")

    cache: Dict[str, List[float]] = {}

    # Pass 1: Node labels & attributes
    print("Pass 1/2: Embedding node labels and attributes with CLIP ...")
    for scene_id in tqdm(all_scenes, desc="Nodes"):
        objects = all_scenes[scene_id].get("objects", {})
        obj_ids = list(objects.keys())
        label_texts = [objects[oid]["label"] for oid in obj_ids]
        label_embs = _batch_embed_texts(label_texts, batch_size, cache)

        for oid, label_emb in zip(obj_ids, label_embs):
            objects[oid]["label_clip"] = label_emb

            raw_attrs = objects[oid].get("attributes", {})
            attributes_clip: Dict[str, List[List[float]]] = {}
            for attr_type, attr_vals in raw_attrs.items():
                if attr_vals:
                    attr_embs = _batch_embed_texts(attr_vals, batch_size, cache)
                    attributes_clip[attr_type] = attr_embs
                else:
                    attributes_clip[attr_type] = []
            objects[oid]["attributes_clip"] = attributes_clip

    # Pass 2: Edge relations
    print("Pass 2/2: Embedding edge relations with CLIP ...")
    for scene_id in tqdm(all_scenes, desc="Edges"):
        edge_lists = all_scenes[scene_id].get("edge_lists", {})
        relations = edge_lists.get("relation", [])
        if not relations:
            edge_lists["relation_clip"] = []
            continue
        rel_embs = _batch_embed_texts(relations, batch_size, cache)
        edge_lists["relation_clip"] = rel_embs

    # Save
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_pt} ...")
    torch.save(all_scenes, str(output_pt))

    # Quick verification
    sample_sid = next(iter(all_scenes))
    sample_obj = next(iter(all_scenes[sample_sid]["objects"].values()))
    dim = len(sample_obj["label_clip"])
    print(f"  Verification: label_clip dim = {dim}")
    edge_lists = all_scenes[sample_sid].get("edge_lists", {})
    if edge_lists.get("relation_clip"):
        rdim = len(edge_lists["relation_clip"][0])
        print(f"  Verification: relation_clip dim = {rdim}")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reprocess graph .pt files with CLIP embeddings "
        "(label_clip, relation_clip, attributes_clip)."
    )
    parser.add_argument(
        "--input_pt", type=Path, required=True,
        help="Path to existing .pt file with edge_lists.",
    )
    parser.add_argument(
        "--output_pt", type=Path, required=True,
        help="Output .pt file path.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for CLIP embedding (default: 64).",
    )
    args = parser.parse_args()
    reprocess(args.input_pt, args.output_pt, args.batch_size)


if __name__ == "__main__":
    main()
