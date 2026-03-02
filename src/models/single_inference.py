"""Open-set single-query inference: natural language to top-k 3DSSG scene matches.

Usage::

    python -m src.models.single_inference \
        inference.query="There is a wooden chair next to a table." \
        inference.top_k=5

Requires OPENAI_API_KEY in .env or environment.
"""

import json
import os

import numpy as np
import openai
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.data_processing.scene_graph import SceneGraph
from src.models.model_graph2graph import BigGNN
from src.models.inference import compute_match_score
from src.data_processing.create_text_embeddings import (
    create_embedding, create_embedding_clip, create_embedding_nlp
)


def embed_word(word: str, embedding_type="word2vec"):
    """Embeds a single word using the specified backend.

    Args:
        word: Word or phrase to embed.
        embedding_type: One of ``'word2vec'``, ``'clip'``, or ``'ada'``.

    Returns:
        List of floats representing the embedding vector.

    Raises:
        ValueError: If ``embedding_type`` is not recognized.
    """
    if embedding_type == "word2vec":
        return create_embedding_nlp(word).tolist()
    elif embedding_type == "clip":
        return create_embedding_clip(word).tolist()
    elif embedding_type == "ada":
        return create_embedding(word)
    else:
        raise ValueError(f"Unknown embedding type {embedding_type}")


def parse_text_to_json(query_text: str, debug: bool = False) -> dict:
    """Uses GPT to extract a scene graph from a natural language description.

    Sends the query to GPT-4o-mini to parse objects, attributes, and
    relationships into a structured JSON graph.

    Args:
        query_text: Natural language scene description.
        debug: If True, prints raw LLM output and parsed JSON.

    Returns:
        Dictionary with ``'nodes'`` and ``'edges'`` lists ready for SceneGraph.

    Raises:
        ValueError: If the LLM returns invalid JSON that cannot be parsed.
    """
    client = openai.OpenAI()
    prompt = f"""
    You are a parser that converts natural language scene descriptions into a JSON graph.
    Extract:
    - objects (with id, label, attributes if any)
    - relationships (edges: source, target, relationship)

    Rules:
    - Assign each object an integer id starting at 0.
    - Each node: {{"id": int, "label": str, "attributes": [str,...]}}
    - Each edge: {{"source": int, "target": int, "relationship": str}}
    - If no attributes → "attributes": []
    - If no edges → "edges": []

    Example:
    Input: "There is a wooden chair next to a table."
    Output:
    {{
    "nodes": [
        {{"id": 0, "label": "chair", "attributes": ["wooden"]}},
        {{"id": 1, "label": "table", "attributes": []}}
    ],
    "edges": [
        {{"source": 0, "target": 1, "relationship": "next to"}}
    ]
    }}

    Now process:
    "{query_text}"
    Only output valid JSON, nothing else.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a JSON scene graph extractor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    if debug:
        print("\n[DEBUG] Raw LLM output:\n", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            raise ValueError(f"LLM returned invalid JSON:\n{raw}")

    if debug:
        print("\n[DEBUG] Parsed JSON graph:")
        print(json.dumps(parsed, indent=2))

    return parsed


def text_to_scenegraph(query_text: str,
                       embedding_type="word2vec",
                       use_attributes=True,
                       scene_id="query_0001", debug: bool = False):
    """Converts a natural language query into a SceneGraph.

    Parses the text with GPT, embeds all node labels, attributes, and edge
    relationships, then constructs a SceneGraph.

    Args:
        query_text: Natural language scene description.
        embedding_type: Embedding backend (``'word2vec'``, ``'clip'``, or ``'ada'``).
        use_attributes: Whether to include attribute embeddings.
        scene_id: Scene ID to assign to the resulting graph.
        debug: If True, enables debug output during parsing.

    Returns:
        A SceneGraph constructed from the parsed and embedded text.
    """
    parsed = parse_text_to_json(query_text, debug)

    for node in parsed["nodes"]:
        node["label_" + embedding_type] = embed_word(node["label"], embedding_type)
        node["attributes_" + embedding_type] = {
            "all": [embed_word(a, embedding_type) for a in node["attributes"]]
        }

    for edge in parsed["edges"]:
        edge["relation_" + embedding_type] = embed_word(edge["relationship"], embedding_type)

    return SceneGraph(scene_id,
                      graph_type="scanscribe",
                      graph=parsed,
                      embedding_type=embedding_type,
                      use_attributes=use_attributes)


def run_single_inference(cfg: DictConfig) -> None:
    """Runs single-query text-to-scene retrieval against the 3DSSG database.

    Args:
        cfg: Merged Hydra configuration.
    """
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.inference.query is None:
        raise ValueError("inference.query is required. Set via CLI: inference.query='...'")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env file.")

    # Load 3DSSG database
    g3d_raw = torch.load(cfg.paths.graphs_3dssg,
                         map_location="cpu", weights_only=False)
    database_3dssg = {
        sid: SceneGraph(sid, graph_type="3dssg", graph=g,
                        max_dist=cfg.graph.max_dist,
                        embedding_type=cfg.graph.embedding_type,
                        use_attributes=cfg.graph.use_attributes)
        for sid, g in g3d_raw.items()
    }

    # Load model
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    if cfg.eval.model_name is None:
        raise ValueError("eval.model_name is required. Set via CLI: eval.model_name=my_model")
    ckpt_path = ckpt_dir / f"{cfg.eval.model_name}.pt"

    model = BigGNN(cfg.model.N, cfg.model.heads, cfg.model.embed_dim, cfg.model.dropout).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()

    # Convert query text to SceneGraph
    query_text = cfg.inference.query
    debug = cfg.inference.debug
    query_graph = text_to_scenegraph(query_text,
                                     embedding_type=cfg.graph.embedding_type,
                                     use_attributes=cfg.graph.use_attributes,
                                     scene_id="query_0001", debug=debug)

    # Score against database
    scores = {}
    iterator = database_3dssg.items()

    if debug:
        iterator = tqdm(iterator, total=len(database_3dssg), desc="Scoring scenes")

    for sid, sg in iterator:
        scores[sid] = compute_match_score(model, query_graph, sg, device,
                                              cfg.inference.score_blend_weight,
                                              cfg.graph.dbscan_eps,
                                              cfg.graph.dbscan_min_samples)

    best = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:cfg.inference.top_k]

    print(f"\nQuery: {query_text}")
    print("Top matches:")
    for rank, (sid, sc) in enumerate(best, 1):
        print(f"  {rank:>2}. {sid:<18}  score={sc:5.3f}")

    if debug:
        print("\n[DEBUG] Finished scoring all scenes.")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entry point for single-query inference."""
    run_single_inference(cfg)


if __name__ == "__main__":
    main()
