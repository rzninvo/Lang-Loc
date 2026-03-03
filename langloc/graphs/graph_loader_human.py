"""Loads and processes human-annotated scene graph JSON files."""

import os
import json
import torch
from tqdm import tqdm

from langloc.graphs.graph_loader_utils import get_ada, get_word2vec


def process_scenes_to_dict(dir_to_scenes):
    """Loads all JSON scene files from a directory into a dictionary.

    Args:
        dir_to_scenes: Directory containing ``<scene_id>.json`` files.

    Returns:
        Dictionary mapping scene IDs (filename without extension) to
        parsed scene dicts.
    """
    ids = os.listdir(dir_to_scenes)
    scenes = {}
    for id in tqdm(ids):
        try:
            scene = {}
            filename = id
            with open(os.path.join(dir_to_scenes, filename)) as f:
                scene = json.load(f)
        except Exception as e:
            print(f'Error processing scene {id}: {e}')

        scenes[id.split('.')[0]] = scene
    return scenes


def add_edge_features(all_scenes):
    """Adds Ada and word2vec embeddings to all edges.

    Args:
        all_scenes: Dictionary of scene dicts (modified in place).

    Returns:
        The mutated ``all_scenes`` dict.
    """
    hada = {}
    hw2v = {}
    for scene_id in tqdm(all_scenes):
        for edge in all_scenes[scene_id]['edges']:
            edge['relation_ada'], hada = get_ada(edge['relationship'], hada)
            edge['relation_word2vec'], hw2v = get_word2vec(edge['relationship'], hw2v)
    return all_scenes


def add_node_features(all_scenes):
    """Adds Ada and word2vec embeddings to all nodes and their attributes.

    Args:
        all_scenes: Dictionary of scene dicts (modified in place).

    Returns:
        The mutated ``all_scenes`` dict.
    """
    hada = {}
    hw2v = {}
    for scene_id in tqdm(all_scenes):
        for node in all_scenes[scene_id]['nodes']:
            node['label_ada'], hada = get_ada(node['label'], hada)
            node['label_word2vec'], hw2v = get_word2vec(node['label'], hw2v)
            attributes_ada = {'all': []}
            attributes_word2vec = {'all': []}
            for attribute in node['attributes']:
                attr_ada, hada = get_ada(attribute, hada)
                attr_word2vec, hw2v = get_word2vec(attribute, hw2v)
                attributes_ada['all'].append(attr_ada)
                attributes_word2vec['all'].append(attr_word2vec)
            node['attributes_ada'] = attributes_ada
            node['attributes_word2vec'] = attributes_word2vec
    return all_scenes


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with human annotation JSONs')
    parser.add_argument('--output', type=str, required=True, help='Path to save processed .pt file')
    cli_args = parser.parse_args()

    all_scenes = process_scenes_to_dict(cli_args.input_dir)
    all_scenes = add_node_features(all_scenes)
    all_scenes = add_edge_features(all_scenes)
    torch.save(all_scenes, cli_args.output)

    print(f'keys: {all_scenes.keys()}')
    print(f'len of keys: {len(all_scenes.keys())}')
