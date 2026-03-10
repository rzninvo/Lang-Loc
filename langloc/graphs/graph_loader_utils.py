"""Shared utilities for graph loading: bounding-box geometry, embedding caching,
and edge validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", message=r"\[W095\]", category=UserWarning)

import torch

from langloc.utils.utils import _get_nlp
from langloc.graphs.create_text_embeddings import create_embedding, create_embedding_clip


def plot_relation(obj1: dict, obj2: dict, ax: plt.Axes, distance: float) -> None:
    """Draws a line between two object centroids on a 3-D axis.

    Args:
        obj1: First object dict with ``obb.centroid``.
        obj2: Second object dict with ``obb.centroid``.
        ax: Matplotlib 3-D axis.
        distance: Line width (thicker = closer).
    """
    centroid1 = np.array(obj1['obb']['centroid'])
    centroid2 = np.array(obj2['obb']['centroid'])
    ax.plot([centroid1[0], centroid2[0]], [centroid1[1], centroid2[1]], [centroid1[2], centroid2[2]], c='b', linewidth=distance)


def draw_obb(corners: np.ndarray, ax: plt.Axes) -> None:
    """Draws the 12 edges of an oriented bounding box on a 3-D axis.

    Args:
        corners: ``(8, 4)`` array of homogeneous corner coordinates.
        ax: Matplotlib 3-D axis.
    """
    ax.plot([corners[0, 0], corners[1, 0]], [corners[0, 1], corners[1, 1]], [corners[0, 2], corners[1, 2]], c='g')
    ax.plot([corners[0, 0], corners[2, 0]], [corners[0, 1], corners[2, 1]], [corners[0, 2], corners[2, 2]], c='g')
    ax.plot([corners[0, 0], corners[3, 0]], [corners[0, 1], corners[3, 1]], [corners[0, 2], corners[3, 2]], c='g')
    ax.plot([corners[1, 0], corners[4, 0]], [corners[1, 1], corners[4, 1]], [corners[1, 2], corners[4, 2]], c='g')
    ax.plot([corners[1, 0], corners[5, 0]], [corners[1, 1], corners[5, 1]], [corners[1, 2], corners[5, 2]], c='g')
    ax.plot([corners[2, 0], corners[4, 0]], [corners[2, 1], corners[4, 1]], [corners[2, 2], corners[4, 2]], c='g')
    ax.plot([corners[2, 0], corners[6, 0]], [corners[2, 1], corners[6, 1]], [corners[2, 2], corners[6, 2]], c='g')
    ax.plot([corners[3, 0], corners[5, 0]], [corners[3, 1], corners[5, 1]], [corners[3, 2], corners[5, 2]], c='g')
    ax.plot([corners[3, 0], corners[6, 0]], [corners[3, 1], corners[6, 1]], [corners[3, 2], corners[6, 2]], c='g')
    ax.plot([corners[4, 0], corners[7, 0]], [corners[4, 1], corners[7, 1]], [corners[4, 2], corners[7, 2]], c='g')
    ax.plot([corners[5, 0], corners[7, 0]], [corners[5, 1], corners[7, 1]], [corners[5, 2], corners[7, 2]], c='g')
    ax.plot([corners[6, 0], corners[7, 0]], [corners[6, 1], corners[7, 1]], [corners[6, 2], corners[7, 2]], c='g')


def bounding_box(obj: dict, ax: plt.Axes | None = None, plot: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
    """Computes or visualises an oriented bounding box from a 3DSSG object.

    Args:
        obj: Object dict with ``obb`` containing ``normalizedAxes``,
            ``centroid``, and ``axesLengths``.
        ax: Matplotlib 3-D axis (required if ``plot=True``).
        plot: If True, draws the OBB on ``ax`` instead of returning corners.

    Returns:
        Tuple of (bb_min, bb_max) each as 3-element arrays, or None if
        ``plot=True``.
    """
    mat44 = np.eye(4)
    mat44[:3, :3] = np.array(obj['obb']['normalizedAxes']).reshape(3, 3).transpose()
    mat44[:3, 3] = obj['obb']['centroid']

    X, Y, Z = obj['obb']['axesLengths']
    corners = [
        [0, 0, 0, 1],
        [X, 0, 0, 1],
        [0, Y, 0, 1],
        [0, 0, Z, 1],
        [X, Y, 0, 1],
        [X, 0, Z, 1],
        [0, Y, Z, 1],
        [X, Y, Z, 1]
    ]
    corners = np.array(corners) - np.array([X / 2, Y / 2, Z / 2, 0])

    if plot:
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_xlim3d(-100, 100)
        ax.set_ylim3d(-100, 100)
        ax.set_zlim3d(-100, 100)

        corners = np.array([mat44 @ c for c in corners])
        ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c='r', marker='o')
        draw_obb(corners, ax)
    else:
        corners = np.array([mat44 @ c for c in corners])

        bb_min = np.min(corners, axis=0)
        bb_max = np.max(corners, axis=0)
        return bb_min[:3], bb_max[:3]


def get_obj_distance(obj1: str, obj2: str, objs: dict) -> float:
    """Computes the minimum distance between two objects' bounding boxes.

    Args:
        obj1: Key into ``objs`` for the first object.
        obj2: Key into ``objs`` for the second object.
        objs: Dictionary mapping object keys to object dicts.

    Returns:
        Euclidean distance between the nearest faces of the two OBBs.
    """
    obj1 = objs[obj1]
    obj2 = objs[obj2]

    A_min, A_max = bounding_box(obj1)
    B_min, B_max = bounding_box(obj2)

    u = np.array([max(0, x) for x in A_min - B_max])
    v = np.array([max(0, x) for x in B_min - A_max])
    dist = np.sqrt(np.sum(u * u) + np.sum(v * v))
    return dist


def get_clip(desc: str, cache: dict) -> tuple[torch.Tensor, dict]:
    """Returns a cached CLIP embedding, computing it if missing.

    Args:
        desc: Text description to embed.
        cache: Dictionary of previously computed embeddings.

    Returns:
        Tuple of (embedding_tensor, updated_cache).
    """
    if desc in cache:
        return cache[desc], cache
    else:
        emb = create_embedding_clip(desc)
        cache[desc] = emb
        return emb, cache


def get_ada(desc: str, hash: dict) -> tuple[list[float], dict]:
    """Returns a cached Ada embedding, computing it if missing.

    Args:
        desc: Text description to embed.
        hash: Dictionary of previously computed embeddings.

    Returns:
        Tuple of (embedding, updated_hash).
    """
    if desc in hash:
        return hash[desc], hash
    else:
        hash[desc] = create_embedding(desc)
    return hash[desc], hash


def get_word2vec(desc: str, hash: dict) -> tuple[np.ndarray, dict]:
    """Returns a cached spaCy word2vec embedding, computing it if missing.

    Args:
        desc: Text description to embed. Returns zero vector if empty.
        hash: Dictionary of previously computed embeddings.

    Returns:
        Tuple of (300-dim numpy array, updated_hash). Returns a zero
        vector paired with the unchanged hash if ``desc`` is empty.
    """
    if desc == "":
        return np.zeros(300), hash
    if desc in hash:
        return hash[desc], hash
    else:
        hash[desc] = _get_nlp()(desc)[0].vector
    return hash[desc], hash


def check_and_remove_invalid_edges(all_scenes: dict) -> dict:
    """Removes edges whose source or target cannot be parsed as integers.

    Args:
        all_scenes: Nested dict of ``{scene_id: {txt_id: {edges: [...]}}}``.

    Returns:
        The mutated ``all_scenes`` dict with invalid edges removed.
    """
    for scene_id in tqdm(all_scenes):
        for txt_id in all_scenes[scene_id]:
            valid_edges = []
            for edge in all_scenes[scene_id][txt_id]['edges']:
                try:
                    int(edge['source'])
                    int(edge['target'])
                    valid_edges.append(edge)
                except (ValueError, TypeError):
                    print(f'Error in scene {scene_id}, txt {txt_id}, '
                          f'source {edge["source"]}, target {edge["target"]}')
                    print("Removing edge")
            all_scenes[scene_id][txt_id]['edges'] = valid_edges
    return all_scenes
