"""Shared utility functions: text parsing, spaCy embeddings, and PyG graph helpers."""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=r"\[W095\]", category=UserWarning)

import os
import numpy as np
import torch
import re
import json
import math

SPACY_EMBED_DIM = 300

# Lazy-loaded spaCy model
_nlp = None

def _get_nlp() -> object:
    """Return the lazily-loaded spaCy ``en_core_web_lg`` model."""
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_lg")
    return _nlp


def load_text_dataset(filename: str) -> tuple[set, dict] | None:
    """Loads a ScanScribe text dataset from the bundled scripts directory.

    Args:
        filename: JSON filename (e.g. ``'scanscribe.json'`` or
            ``'scanscribe_*.json'``).

    Returns:
        Tuple of (scan_ids, dict_of_texts) where dict_of_texts maps
        scene IDs to lists of text descriptions.
    """
    _scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'hugging_face')
    if filename[0:11] == "scanscribe_":
        with open(os.path.join(_scripts_dir, filename), "r") as f:
            scanscribe = json.load(f)

        scan_ids = scanscribe.keys()
        dict_of_texts = scanscribe
    elif filename == "scanscribe.json":
        with open(os.path.join(_scripts_dir, filename), "r") as f:
            scanscribe = json.load(f)
        scan_ids = set()
        dict_of_texts = {}
        for s in scanscribe:
            scan_id = s['scan_id']
            scan_ids.add(scan_id)
            if scan_id not in dict_of_texts:
                dict_of_texts[scan_id] = []
            dict_of_texts[scan_id].append(s['sentence'])
    else:
        print("Invalid filename")
        return
    return scan_ids, dict_of_texts


def verify_subgraph(text_graph: object, subgraph: object, og_graph: object, output: dict, clusters: dict) -> None:
    """Prints debug info about matched nodes between a text graph and scene subgraph.

    Args:
        text_graph: Query text scene graph.
        subgraph: Extracted subgraph from the scene.
        og_graph: Original full scene graph.
        output: Match output dict with ``'matches0'`` key.
        clusters: Dictionary mapping cluster IDs to lists of node indices.
    """
    print("verifying subgraph--------------------------")
    for idx, val in enumerate(output['matches0'][0]):
        matched_list = []
        if (val != -1):
            for cluster in clusters:
                if val in clusters[cluster]: matched_list = clusters[cluster]
            matched = [og_graph.get_nodes()[n].label for n in matched_list]
            print(text_graph.get_nodes()[idx].label, " --> ", end="")
            print(matched)
        else:
            print(text_graph.get_nodes()[idx].label, " --> ")
    for n in subgraph.get_nodes():
        print(n.label, end=" ")
    print("--------------------------")


def txt_to_json(text: str) -> dict | list:
    """Cleans raw text and parses it as JSON.

    Removes newlines, unescapes quotes, and collapses whitespace before parsing.

    Args:
        text: Raw text string to parse.

    Returns:
        Parsed JSON object (dict or list).
    """
    text = text.replace('\n', '')
    text = text.replace('\"', '"')
    text = re.sub(' +', ' ', text)
    json_data = json.loads(text)
    return json_data


def noun_in_list_of_nouns(noun: str, nouns: list[str], threshold: float = 0.5) -> tuple[str | None, bool]:
    """Finds the most similar noun in a list using spaCy similarity.

    Args:
        noun: Query noun string.
        nouns: List of candidate noun strings.
        threshold: Minimum similarity to consider a match.

    Returns:
        Tuple of (best_match_noun, is_above_threshold).
    """
    max_sim = 0
    max_sim_noun = None
    for n in nouns:
        sim = _get_nlp()(noun).similarity(_get_nlp()(n))
        if sim > max_sim:
            max_sim = sim
            max_sim_noun = n
    return max_sim_noun, max_sim > threshold


def vectorize_word(word: str) -> np.ndarray:
    """Converts a word to its spaCy word2vec embedding.

    Args:
        word: Input word string. Empty string returns a zero vector.

    Returns:
        Numpy array of shape ``(SPACY_EMBED_DIM,)``.
    """
    if word == "":
        return np.zeros(SPACY_EMBED_DIM)
    return _get_nlp()(word)[0].vector


def recover_word(vector: np.ndarray, top_n: int = 3) -> list[str]:
    """Recovers the closest words to a given embedding vector.

    Args:
        vector: Embedding vector of length ``SPACY_EMBED_DIM``.
        top_n: Number of nearest words to return.

    Returns:
        List of the top_n closest word strings.
    """
    assert len(vector) == SPACY_EMBED_DIM, \
        f"Vector must have length {SPACY_EMBED_DIM}, got {len(vector)}"
    ms = _get_nlp().vocab.vectors.most_similar(
        np.asarray([vector]), n=top_n
    )
    words = [_get_nlp().vocab.strings[w] for w in ms[0][0]]
    return words


def print_closest_words(out: np.ndarray, x: np.ndarray, first_n: int = 5) -> None:
    """Prints the closest recovered words for each row of two embedding matrices.

    Args:
        out: Output embedding matrix.
        x: Input embedding matrix (same shape as ``out``).
        first_n: Number of rows to print.
    """
    assert out.shape == x.shape, "out and x must have the same shape"
    if len(out.shape) != 2:
        out = out.reshape(-1, SPACY_EMBED_DIM)
        x = x.reshape(-1, SPACY_EMBED_DIM)
    for i in range(min(out.shape[0], first_n)):
        x_word = recover_word(x[i])
        out_word = recover_word(out[i])
        print("Closest words to " + str(x_word) + ": " + str(out_word))


def print_word_distances(word1: str, word2: str) -> None:
    """Prints the L2 distance between two words' spaCy embeddings.

    Args:
        word1: First word string.
        word2: Second word string.
    """
    word1_vec = _get_nlp()(word1)[0].vector
    word2_vec = _get_nlp()(word2)[0].vector
    print("Distance between " + word1 + " and " + word2 + ": " + str(np.linalg.norm(word1_vec - word2_vec)))


def print_word_similarity(word1: str, word2: str) -> None:
    """Prints the spaCy similarity score between two words.

    Args:
        word1: First word string.
        word2: Second word string.
    """
    print("Similarity between " + word1 + " and " + word2 + ": " + str(_get_nlp()(word1).similarity(_get_nlp()(word2))))


def word_similarity(word1: str, word2: str) -> float:
    """Computes the spaCy similarity between two words.

    Args:
        word1: First word string.
        word2: Second word string.

    Returns:
        Similarity score in [-1, 1].
    """
    return _get_nlp()(word1).similarity(_get_nlp()(word2))


def make_cross_graph(x_1_dim: tuple, x_2_dim: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds a fully-connected bipartite cross-graph between two node sets.

    Creates edges from every node in graph 1 to every node in graph 2,
    with zero-valued edge attributes.

    Args:
        x_1_dim: Shape tuple ``(num_nodes_1, ...)``.
        x_2_dim: Shape tuple ``(num_nodes_2, ...)``.

    Returns:
        Tuple of (edge_index, edge_attr) tensors for the cross-graph.
    """
    x_1_dim = x_1_dim[0]
    x_2_dim = x_2_dim[0]

    edge_index_cross = torch.tensor([[], []], dtype=torch.long)
    edge_attr_cross = torch.tensor([], dtype=torch.float)

    for i in range(x_1_dim):
        for j in range(x_2_dim):
            edge_index_cross = torch.cat((edge_index_cross, torch.tensor([[i], [x_1_dim + j]], dtype=torch.long)), dim=1)
            edge_attr_cross = torch.cat((edge_attr_cross, torch.zeros((1, SPACY_EMBED_DIM), dtype=torch.float)), dim=0)

    assert edge_index_cross.shape[1] == x_1_dim * x_2_dim, \
        "Cross-graph edge count must equal product of node counts"
    assert edge_attr_cross.shape[0] == x_1_dim * x_2_dim, \
        "Cross-graph edge attr count must match edge count"
    return edge_index_cross, edge_attr_cross


def mask_node(x: torch.Tensor, p: float = 0.1) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Randomly masks a fraction of node features by zeroing them out.

    Args:
        x: Node feature tensor of shape ``(num_nodes, feature_dim)``.
        p: Fraction of nodes to mask.

    Returns:
        Tuple of (masked_tensor, masked_row_indices) or ``(x, None)``
        if no nodes were masked.
    """
    if p == 0:
        return x, None
    x_clone = x.clone()
    num_nodes_to_mask = math.floor(x.shape[0] * p)
    if num_nodes_to_mask == 0:
        return x, None
    rows_to_mask = torch.randperm(x.shape[0])[:num_nodes_to_mask]
    x_clone[rows_to_mask] = 0
    return x_clone, rows_to_mask


def accuracy_score(y_pred: np.ndarray, y_true: np.ndarray, top_n: int = 3, thresh: float = 0.8) -> float:
    """Computes word-recovery accuracy between predicted and true embeddings.

    For each row, recovers the top-n closest words and checks if the true
    word is among them (exact match or above similarity threshold).

    Args:
        y_pred: Predicted embedding tensor.
        y_true: Ground-truth embedding tensor.
        top_n: Number of candidate words to recover per row.
        thresh: Similarity threshold for fuzzy matching.

    Returns:
        Accuracy as a float in [0, 1].
    """
    assert y_pred.shape[0] == y_true.shape[0], \
        "Prediction and ground-truth must have the same number of rows"
    if len(y_pred.shape) != 2:
        y_pred = y_pred.reshape(-1, SPACY_EMBED_DIM)
        y_true = y_true.reshape(-1, SPACY_EMBED_DIM)

    count_correct = 0
    for i in range(y_pred.shape[0]):
        y_pred_word = recover_word(y_pred[i], top_n=top_n)
        y_true_word = recover_word(y_true[i], top_n=top_n)

        if y_true_word[0] in y_pred_word:
            count_correct += 1
        else:
            for y in y_pred_word:
                if _get_nlp()(y_true_word[0]).similarity(_get_nlp()(y)) >= thresh:
                    count_correct += 1
                    break

    return count_correct / y_pred.shape[0]
