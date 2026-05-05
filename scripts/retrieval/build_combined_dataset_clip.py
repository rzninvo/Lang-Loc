"""Build ``combined_dataset_clip/`` — the paper's training dataset.

Combines:

  * **3DSSG side**: ``scene_graphs_unique/*.json`` (one JSON per unique 3DSSG
    scene, already in 518D format with ``clip_text_emb`` per node and 7-vocab
    spatial relations in ``edges_text``). 100 files in the released artefact.

  * **ScanScribe side**: read ``scanscribe_graphs_train_final_no_graph_min.pt``,
    map each ``scan_id`` → ``reference_id`` via ``3RScan.json``, and emit one
    JSON per ``(reference_id, text_id)`` paraphrase. Each paraphrase becomes a
    file ``{reference_id}_text_{text_id}.json``.

The expected output is ~3,400 JSON files in ``--out_dir`` (~100 3DSSG +
~3,356 ScanScribe paraphrases). Exact count depends on which scan_ids are
present in 3RScan.json and how many paraphrases each has.

The output is consumed by ``langloc.retrieval.train`` as ``--dataset_dir``.

Usage::

    python -m scripts.retrieval.build_combined_dataset_clip \\
        --scanscribe_path data/processed_data/eval_pool/scanscribe_graphs_train_final_no_graph_min.pt \\
        --scene_graphs_dir data/processed_data/eval_pool/scene_graphs_unique \\
        --metadata_path data/3RScan/3RScan.json \\
        --out_dir data/processed_data/combined_dataset_clip \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import clip
import torch
from tqdm import tqdm


def get_clip_embedding(text: str, clip_model, device: torch.device) -> list[float]:
    """Return an L2-normalized CLIP text embedding as a Python list."""
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy().astype("float32").tolist()


def convert_scanscribe(graph: dict, scene_id: str, text_id, clip_model, device) -> dict:
    """Convert a ScanScribe text graph into the 518D training-JSON schema.

    Text graphs have no spatial information: centroid is zero, mean_color is
    grey (128/128/128 — the dataset class divides by 255 at load time), and
    radius is 0.4. ``base_label`` falls back to ``label``.
    """
    nodes_dict: dict[str, dict] = {}
    for node in graph["nodes"]:
        label = node["label"]
        nodes_dict[str(node["id"])] = {
            "label": label,
            "base_label": label,
            "centroid": [0.0, 0.0, 0.0],
            "mean_color": [128.0, 128.0, 128.0],
            "radius": 0.4,
            "clip_text_emb": get_clip_embedding(label, clip_model, device),
        }

    edges_text: list[dict] = []
    for edge in graph.get("edges", []) or []:
        edges_text.append({
            "subject": str(edge["source"]),
            "object": str(edge["target"]),
            "relation": edge["relationship"],
        })

    return {
        "scene_id": scene_id,
        "text_id": text_id,
        "nodes": nodes_dict,
        "edges_text": edges_text,
        "source": "scanscribe",
    }


def load_scan_to_reference(metadata_path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Build scan_id → reference_id and reference_id → list[scan_id] from 3RScan.json."""
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    scan_to_ref: dict[str, str] = {}
    ref_to_scans: dict[str, list[str]] = {}
    for entry in meta:
        ref_id = entry["reference"]
        scan_to_ref[ref_id] = ref_id
        ref_to_scans.setdefault(ref_id, []).append(ref_id)
        for scan in entry.get("scans", []):
            sid = scan["reference"]
            scan_to_ref[sid] = ref_id
            ref_to_scans[ref_id].append(sid)
    return scan_to_ref, ref_to_scans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scanscribe_path", required=True,
                    help="Path to scanscribe_graphs_train_final_no_graph_min.pt")
    ap.add_argument("--scene_graphs_dir", required=True,
                    help="Path to scene_graphs_unique/ (100 3DSSG JSONs in V2 schema)")
    ap.add_argument("--metadata_path", required=True,
                    help="Path to 3RScan.json (for scan→reference mapping)")
    ap.add_argument("--out_dir", required=True,
                    help="Output dir; will contain ~3.4k JSONs + metadata.json")
    ap.add_argument("--clip_model", default="ViT-B/32")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed (canonical project seed = 42).")
    args = ap.parse_args()

    # Canonical project seed (see CLAUDE.md §0). CLIP encoding here is
    # deterministic given input; seeds are set defensively.
    from langloc.utils.seed import set_seed
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"[BUILD] device={device}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[BUILD] loading CLIP {args.clip_model}…", flush=True)
    clip_model, _ = clip.load(args.clip_model, device=device)

    # ---- 3RScan grouping ------------------------------------------------------
    print(f"[BUILD] loading 3RScan grouping from {args.metadata_path}…", flush=True)
    scan_to_ref, ref_to_scans = load_scan_to_reference(Path(args.metadata_path))
    print(f"[BUILD]   {len(scan_to_ref)} scan_ids, {len(ref_to_scans)} reference rooms",
          flush=True)

    # ---- ScanScribe → JSON ----------------------------------------------------
    print(f"[BUILD] loading ScanScribe train graphs from {args.scanscribe_path}…",
          flush=True)
    scanscribe = torch.load(args.scanscribe_path, map_location="cpu", weights_only=False)
    print(f"[BUILD]   {len(scanscribe)} scan_ids", flush=True)

    n_text = 0
    n_unmapped = 0
    scanscribe_ref_ids: set[str] = set()
    for scan_id in tqdm(list(scanscribe.keys()), desc="ScanScribe → JSON"):
        ref_id = scan_to_ref.get(scan_id)
        if ref_id is None:
            n_unmapped += 1
            continue
        scanscribe_ref_ids.add(ref_id)

        for text_id, graph in scanscribe[scan_id].items():
            doc = convert_scanscribe(graph, ref_id, text_id, clip_model, device)
            out_path = out_dir / f"{ref_id}_text_{text_id}.json"
            with open(out_path, "w") as f:
                json.dump(doc, f)
            n_text += 1

    if n_unmapped:
        print(f"[WARN] {n_unmapped} scan_ids absent from 3RScan.json — skipped",
              flush=True)

    # ---- 3DSSG side ----------------------------------------------------------
    print(f"[BUILD] copying 3DSSG side from {args.scene_graphs_dir}…", flush=True)
    n_3dssg = 0
    dssg_ref_ids: set[str] = set()
    src_dir = Path(args.scene_graphs_dir)
    for fname in tqdm(sorted(os.listdir(src_dir)), desc="3DSSG → JSON"):
        if not fname.endswith(".json"):
            continue
        with open(src_dir / fname, "r") as f:
            graph = json.load(f)
        graph["source"] = "3dssg"
        with open(out_dir / fname, "w") as f:
            json.dump(graph, f)
        scene_id = fname.replace(".json", "")
        dssg_ref_ids.add(scene_id)
        n_3dssg += 1

    overlap = scanscribe_ref_ids & dssg_ref_ids

    metadata = {
        "total_graphs": n_text + n_3dssg,
        "scanscribe_text_graphs": n_text,
        "3dssg_3d_graphs": n_3dssg,
        "overlapping_references": len(overlap),
        "overlapping_reference_ids": sorted(overlap),
        "scanscribe_unique_references": len(scanscribe_ref_ids - dssg_ref_ids),
        "3dssg_unique_references": len(dssg_ref_ids - scanscribe_ref_ids),
        "format": {
            "node_features": "centroid(3) + color(3) + CLIP(512) = 518 dims",
            "clip_model": args.clip_model,
            "scanscribe_geometry": "dummy [0,0,0] centroids (text has no spatial info)",
            "3dssg_geometry": "real centroids + colors from 3D scans",
            "id_mapping": "ScanScribe scan IDs mapped to 3RScan reference IDs",
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[BUILD] DONE — wrote {metadata['total_graphs']} graphs to {out_dir}",
          flush=True)
    print(f"        text={n_text}, 3DSSG={n_3dssg}, overlap_refs={len(overlap)}",
          flush=True)


if __name__ == "__main__":
    main()
