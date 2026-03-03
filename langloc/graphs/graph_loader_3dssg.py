"""Loads and processes 3DSSG scene graphs from raw JSON files into a single
serializable dictionary with node features, edge lists, and embeddings.
"""

import os
import json
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from langloc.graphs.graph_loader_utils import get_obj_distance, bounding_box, plot_relation, get_ada, get_clip, get_word2vec

dist_thr = 1.0

_scans_dict = None
_relationships_dict = None


def _load_3dssg_data():
    """Lazily loads and caches 3DSSG objects.json and relationships.json.

    Returns:
        Tuple of (scans_dict, relationships_dict) keyed by scan ID.
    """
    global _scans_dict, _relationships_dict
    if _scans_dict is not None:
        return _scans_dict, _relationships_dict

    datasets_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    objects_path = os.path.join(datasets_dir, '3DSSG', 'objects.json')
    with open(objects_path, 'r') as f:
        objects = json.load(f)
    _scans_dict = {}
    for s in objects['scans']:
        _scans_dict[s['scan']] = s

    relationships_path = os.path.join(datasets_dir, '3DSSG', 'relationships.json')
    with open(relationships_path, 'r') as f:
        relationships = json.load(f)
    _relationships_dict = {}
    for r in relationships['scans']:
        _relationships_dict[r['scan']] = r

    return _scans_dict, _relationships_dict


def process_scenes(dir_to_scenes, plot=False, dist_thr=1.0):
    """Processes all 3RScan scenes into graph dictionaries.

    Args:
        dir_to_scenes: Parent directory containing ``3RScan/<scan_id>/``.
        plot: If True, visualise bounding boxes and relations.
        dist_thr: Maximum edge distance threshold.

    Returns:
        Dictionary mapping scan IDs to scene dicts with ``objects`` and
        ``relationships`` keys.
    """
    scans_dict, relationships_dict = _load_3dssg_data()
    ids = os.listdir(os.path.join(dir_to_scenes, '3RScan'))
    scenes = {}
    for id in tqdm(ids):
        try:
            scene = {}
            scene['objects'], scene['relationships'] = process_objects_and_relationships(dir_to_scenes, id, plot, dist_thr)
            assert id not in scenes, f"Duplicate scene ID: {id}"
            scenes[id] = scene
        except Exception as e:
            print(f'Error processing scene {id}: {e}')

    assert len(scenes) == len(relationships_dict), "Scene count mismatch"
    return scenes


def process_objects_and_relationships(dir_to_objects, scene_id, plot=False, dist_thr=1.0):
    """Processes objects and relationships for a single 3RScan scene.

    Args:
        dir_to_objects: Parent directory containing ``3RScan/<scene_id>/``.
        scene_id: The 3RScan scene identifier.
        plot: If True, visualise bounding boxes and relations.
        dist_thr: Maximum edge distance threshold for plotting.

    Returns:
        Tuple of (objects_in_scan, graph_adj) where objects_in_scan is a dict
        of object dicts and graph_adj maps object IDs to adjacency info.
    """
    scans_dict, relationships_dict = _load_3dssg_data()
    segmentations_path = os.path.join(dir_to_objects, '3RScan', scene_id, 'semseg.v2.json')
    with open(segmentations_path, 'r') as f:
        segmentations = json.load(f)
    segmentations = segmentations['segGroups']

    objects_in_scan = scans_dict[scene_id]['objects']
    assert len(objects_in_scan) == len(segmentations)

    objects_in_scan = pd.DataFrame(objects_in_scan)
    objects_in_scan = objects_in_scan.drop(columns=['nyu40', 'ply_color', 'eigen13', 'rio27', 'affordances', 'state_affordances', 'symmetry'], errors='ignore')
    objects_in_scan = objects_in_scan.set_index('id', drop=False).to_dict('index')

    segmentations_dict = {}
    for seg in segmentations:
        segmentations_dict[str(seg['id'])] = seg

    graph_adj = {}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for obj_id in objects_in_scan.keys():
        obj = objects_in_scan[obj_id]
        assert obj['label'] == segmentations_dict[obj_id]['label']
        obj['obb'] = segmentations_dict[obj['id']]['obb']

        if plot:
            bounding_box(obj, ax, plot=True)

        graph_adj[obj_id] = {'label': obj['label'], 'adj_to': []}

    assert scene_id in relationships_dict, f"Scene {scene_id} not in relationships"
    relationship = relationships_dict[scene_id]

    for r in relationship['relationships']:
        assert str(r[0]) in graph_adj, f"Object {r[0]} not found"
        assert str(r[1]) in graph_adj, f"Object {r[1]} not found"

        distance = get_obj_distance(str(r[0]), str(r[1]), objects_in_scan)
        if plot and distance < dist_thr:
            plot_relation(objects_in_scan[str(r[0])], objects_in_scan[str(r[1])], ax, distance)

        graph_adj[str(r[0])]['adj_to'].append({'obj_id': str(r[1]), 'relation': r[3], 'distance': distance})

    if plot:
        plt.show()

    return objects_in_scan, graph_adj


def add_edge_list(all_scenes, output_path=None):
    """Adds edge lists with embeddings to each scene in ``all_scenes``.

    Args:
        all_scenes: Dictionary of scene dicts (modified in place).
        output_path: If provided, saves the result with ``torch.save``.
    """
    _, relationships_dict = _load_3dssg_data()
    hada = {}
    hw2v = {}
    for sceneid in tqdm(all_scenes):
        relationships = relationships_dict[sceneid]['relationships']
        obj1_list = []
        obj2_list = []
        relation_list = []
        relation_word2vec_list = []
        relation_ada_list = []
        dist_list = []
        for rel in relationships:
            obj1_list.append(rel[0])
            obj2_list.append(rel[1])
            relation_list.append(rel[3])
            w2v, hw2v = get_word2vec(rel[3], hw2v)
            relation_word2vec_list.append(w2v)
            ada, hada = get_clip(rel[3], hada)
            relation_ada_list.append(ada)
            dist_list.append(get_obj_distance(str(rel[0]), str(rel[1]), all_scenes[sceneid]['objects']))
        assert len(obj1_list) == len(obj2_list) == len(relation_list) == len(dist_list)
        all_scenes[sceneid]['edge_lists'] = {}
        all_scenes[sceneid]['edge_lists']['from'] = obj1_list
        all_scenes[sceneid]['edge_lists']['to'] = obj2_list
        all_scenes[sceneid]['edge_lists']['relation'] = relation_list
        all_scenes[sceneid]['edge_lists']['relation_word2vec'] = relation_word2vec_list
        all_scenes[sceneid]['edge_lists']['relation_ada'] = relation_ada_list
        all_scenes[sceneid]['edge_lists']['distance'] = dist_list
    if output_path:
        torch.save(all_scenes, output_path)


def add_node_features(all_scenes, output_path=None):
    """Adds label and attribute embeddings to each node in ``all_scenes``.

    Args:
        all_scenes: Dictionary of scene dicts (modified in place).
        output_path: If provided, saves the result with ``torch.save``.
    """
    hada = {}
    hw2v = {}
    print(len(all_scenes))
    for scene in tqdm(all_scenes):
        objects = all_scenes[scene]['objects']
        for obj in tqdm(objects):
            label_ada, hada = get_clip(objects[obj]['label'], hada)
            objects[obj]['label_ada'] = label_ada
            label_word2vec, hw2v = get_word2vec(objects[obj]['label'], hw2v)
            objects[obj]['label_word2vec'] = label_word2vec
            attributes_word2vec = {}
            attributes_ada = {}
            for attrs in objects[obj]['attributes']:
                attributes_word2vec[attrs] = []
                attributes_ada[attrs] = []
                for attr in objects[obj]['attributes'][attrs]:
                    attr_word2vec, hw2v = get_word2vec(attr, hw2v)
                    attributes_word2vec[attrs].append(attr_word2vec)
                    attr_ada, hada = get_clip(attr, hada)
                    attributes_ada[attrs].append(attr_ada)
            objects[obj]['attributes_word2vec'] = attributes_word2vec
            objects[obj]['attributes_ada'] = attributes_ada
    if output_path:
        torch.save(all_scenes, output_path)


def check_num_edges(all_scenes):
    """Verifies that edge list counts match adjacency list counts.

    Args:
        all_scenes: Dictionary of scene dicts with both ``edge_lists``
            and ``relationships`` keys.
    """
    _, relationships_dict = _load_3dssg_data()
    num_edges = []
    adj_num_edges = []
    for scene in all_scenes:
        num_edges.append(len(all_scenes[scene]['edge_lists']['from']))
        adj_list = all_scenes[scene]['relationships']
        scene_sum = 0
        for adj in adj_list:
            adj_to = len(adj_list[adj]['adj_to'])
            scene_sum += adj_to
        adj_num_edges.append(scene_sum)

    assert len(num_edges) == len(adj_num_edges)
    assert all(num_edges[i] == adj_num_edges[i] for i in range(len(num_edges)))


def change_w2v_word2vec(all_scenes, p):
    """Renames ``attributes_w2v`` keys to ``attributes_word2vec``. Deprecated."""
    for scene_id in tqdm(all_scenes):
        for node_id in all_scenes[scene_id]['objects']:
            node = all_scenes[scene_id]['objects'][node_id]
            node['attributes_word2vec'] = node['attributes_w2v']
            del node['attributes_w2v']
    torch.save(all_scenes, p)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input .pt file')
    parser.add_argument('--output', type=str, default=None, help='Path to output .pt file')
    cli_args = parser.parse_args()

    all_scenes = torch.load(cli_args.input, weights_only=False)
    add_edge_list(all_scenes, output_path=cli_args.output)
