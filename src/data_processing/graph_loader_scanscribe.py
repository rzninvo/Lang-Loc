"""ScanScribe scene graph loader: parses JSON descriptions into enriched scene dicts."""

import os
import json
import torch
from tqdm import tqdm

from src.data_processing.graph_loader_utils import get_ada, get_word2vec, check_and_remove_invalid_edges
from src.utils.utils import txt_to_json


def process_scenes_to_dict(dir_to_scenes):
    """Reads all ScanScribe JSON scene files into a nested dictionary.

    Args:
        dir_to_scenes: Path to directory containing per-scene subdirectories,
            each with numbered ``.json`` files.

    Returns:
        Dictionary mapping scene IDs to dicts of ``{text_id: parsed_json}``.
    """
    ids = os.listdir(dir_to_scenes)
    scenes = {}
    for id in tqdm(ids):
        try:
            scene_dir = sorted([int(x[:-5]) for x in os.listdir(os.path.join(dir_to_scenes, id))])
            scene = {}
            for file in scene_dir:
                filename = str(file) + '.json'
                with open(os.path.join(dir_to_scenes, id, filename)) as f:
                    data = f.read()
                    data = txt_to_json(data)
                    data = json.loads(data)
                    scene[file] = data
        except Exception as e:
            print(f'Error processing scene {id}: {e}')

        scenes[id] = scene
    return scenes
                
def add_edge_features(all_scenes):
    """Adds Ada and word2vec embeddings for edge relationships in all scenes.

    Args:
        all_scenes: Nested dict of ``{scene_id: {text_id: graph_dict}}``.

    Returns:
        The same dictionary with ``'relation_ada'`` and ``'relation_word2vec'``
        fields added to each edge.
    """
    hada = {}
    hw2v = {}
    for scene_id in tqdm(all_scenes):
        for txt_id in all_scenes[scene_id]:
            for edge in all_scenes[scene_id][txt_id]['edges']:
                edge['relation_ada'], hada = get_ada(edge['relationship'], hada)
                edge['relation_word2vec'], hw2v = get_word2vec(edge['relationship'], hw2v)
                # NOTE: assumed there are no attributes for edges
    return all_scenes

def add_node_features(all_scenes):
    """Adds Ada and word2vec embeddings for node labels and attributes in all scenes.

    Args:
        all_scenes: Nested dict of ``{scene_id: {text_id: graph_dict}}``.

    Returns:
        The same dictionary with ``'label_ada'``, ``'label_word2vec'``,
        ``'attributes_ada'``, and ``'attributes_word2vec'`` fields added
        to each node.
    """
    hada = {}
    hw2v = {}
    for scene_id in tqdm(all_scenes):
        for txt_id in all_scenes[scene_id]:
            for node in all_scenes[scene_id][txt_id]['nodes']:
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
    parser.add_argument('--input', type=str, required=True, help='Path to input .pt file to inspect')
    cli_args = parser.parse_args()

    all_scenes = torch.load(cli_args.input, weights_only=False)
    sample_key = list(all_scenes.keys())[0]
    print(f'Sample scene: {sample_key}')
    print(f'Total scenes: {len(all_scenes)}')