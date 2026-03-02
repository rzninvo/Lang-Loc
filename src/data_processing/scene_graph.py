"""Scene graph data structures for 3DSSG, ScanScribe, and human-annotated graphs."""

import torch
from tqdm import tqdm
import numpy as np

from src.data_processing.scene_graph_utils import check_valid_graph


class Node:
    """A node in a scene graph representing an object with label and attribute embeddings.

    Attributes:
        idx: Integer node ID.
        label_features: Embedding vector for the object label.
        attribute_features: List of embedding vectors for attributes.
        label: Human-readable object label string.
        attributes: List of human-readable attribute strings.
        features: Combined feature vector (label + optional mean attributes).
    """

    def __init__(self, idx, label_features, attribute_features, use_attributes, label=None, attributes=None):
        self.idx = idx
        self.label_features = label_features
        self.attribute_features = attribute_features
        self.label = label
        self.attributes = attributes
        assert type(self.attribute_features) == list, "attribute_features must be a list"
        assert type(self.label_features) in (list, np.ndarray), "label_features must be a list or ndarray"
        self.features = self.set_features(label_features, attribute_features, use_attributes=use_attributes)

    def set_features(self, labels, attributes, use_attributes):
        """Combines label and attribute embeddings into a single feature vector.

        Args:
            labels: Label embedding vector.
            attributes: List of attribute embedding vectors.
            use_attributes: Whether to incorporate attribute embeddings.

        Returns:
            Combined feature vector. If ``use_attributes`` is True, returns
            label + mean(attributes); otherwise returns label only.
        """
        if use_attributes:
            attribute_features = np.zeros(len(labels))
            if attributes is not None and len(attributes) > 0:
                attribute_features = np.mean(attributes, axis=0)
            l = labels + attribute_features
            assert len(l) == len(labels), "Combined features must match label dimension"
            return l
        else:
            return labels


class Edge:
    """An edge in a scene graph representing a relationship between two nodes.

    Attributes:
        from_idx: Source node ID.
        to_idx: Target node ID.
        features: Embedding vector for the relationship.
    """

    def __init__(self, from_idx, to_idx, features):
        self.from_idx = from_idx
        self.to_idx = to_idx
        self.features = features


class SceneGraph:
    """A scene graph containing nodes (objects) and edges (relationships).

    Supports construction from 3DSSG, ScanScribe, and human-annotated formats.

    Attributes:
        scene_id: Unique scene identifier.
        txt_id: Text description ID (for ScanScribe/human graphs).
        nodes: Dictionary mapping node IDs to Node objects.
        edge_idx: List of ``[from_ids, to_ids]``.
        edge_relations: List of human-readable relation strings.
        edge_features: List of relation embedding vectors.
    """

    def __init__(self, scene_id, txt_id=None, graph_type=None, graph=None, max_dist=None, embedding_type='ada', use_attributes=True):
        self.scene_id = scene_id
        if graph_type == '3dssg':
            self.nodes = self.extract_nodes_3dssg(graph['objects'], use_attributes, embedding_type)
            self.edge_idx, self.edge_relations, self.edge_features = self.extract_edges_3dssg(graph['edge_lists'], max_dist, embedding_type)
            assert len(self.edge_idx[0]) == len(self.edge_features), "Edge count mismatch"
            assert check_valid_graph(self.nodes, self.edge_idx), "Invalid graph: edge references missing node"
        elif graph_type == 'scanscribe' or graph_type == 'human':
            self.nodes = self.extract_nodes_scanscribe(graph['nodes'], use_attributes, embedding_type)
            self.edge_idx, self.edge_relations, self.edge_features = self.extract_edges_scanscribe(graph['edges'], embedding_type)
            assert len(self.edge_idx[0]) == len(self.edge_features), "Edge count mismatch"
            assert check_valid_graph(self.nodes, self.edge_idx), "Invalid graph: edge references missing node"
            self.txt_id = txt_id
        elif graph_type is None:
            self.nodes = None
            self.edge_idx = None
            self.edge_features = None
            self.scene_id = scene_id
            self.txt_id = txt_id

    def extract_nodes_3dssg(self, objects, use_attributes, embedding_type='ada'):
        """Extracts nodes from a 3DSSG object dictionary.

        Args:
            objects: Dictionary mapping object IDs to object dicts with label
                and attribute embeddings.
            use_attributes: Whether to include attribute embeddings in features.
            embedding_type: Embedding key suffix (``'ada'`` or ``'word2vec'``).

        Returns:
            Dictionary mapping integer node IDs to Node objects.
        """
        nodes = {}
        for objid in objects:
            obj = objects[objid]
            attributes_list = [a for attr in obj['attributes_' + embedding_type] for a in obj['attributes_' + embedding_type][attr]]
            if len(attributes_list):
                assert len(attributes_list[0]) == len(obj['label_' + embedding_type]), "Attribute and label dimensions must match"
            node = Node(int(obj['id']), obj['label_' + embedding_type], attributes_list, use_attributes=use_attributes,
                        label=obj['label'], attributes=obj['attributes'])
            nodes[int(obj['id'])] = node
        return nodes

    def extract_nodes_scanscribe(self, objects, use_attributes, embedding_type='ada'):
        """Extracts nodes from a ScanScribe/human node list.

        Args:
            objects: List of node dicts with label and attribute embeddings.
            use_attributes: Whether to include attribute embeddings in features.
            embedding_type: Embedding key suffix (``'ada'`` or ``'word2vec'``).

        Returns:
            Dictionary mapping integer node IDs to Node objects.
        """
        nodes = {}
        for obj in objects:
            attributes_list = obj['attributes_' + embedding_type]['all']
            if len(attributes_list):
                assert len(attributes_list[0]) == len(obj['label_' + embedding_type]), "Attribute and label dimensions must match"
            node = Node(int(obj['id']), obj['label_' + embedding_type], attributes_list, use_attributes=use_attributes,
                        label=obj['label'], attributes=obj['attributes'])
            nodes[int(obj['id'])] = node
        return nodes

    def extract_edges_3dssg(self, edge_lists, max_dist, embedding_type='ada'):
        """Extracts edges from a 3DSSG edge list, filtering by distance.

        Args:
            edge_lists: Dictionary with ``'from'``, ``'to'``, ``'distance'``,
                ``'relation'``, and ``'relation_<embedding_type>'`` lists.
            max_dist: Maximum distance threshold for edge inclusion.
            embedding_type: Embedding key suffix (``'ada'`` or ``'word2vec'``).

        Returns:
            Tuple of (edge_idx, edge_relations, edge_features) where edge_idx
            is ``[from_ids, to_ids]``.
        """
        edge_idx = []
        from_edge = []
        to_edge = []
        edge_attributes = []
        edge_attributes_embedding = []
        for idx, d in enumerate(edge_lists['distance']):
            if d <= max_dist:
                from_edge.append(int(edge_lists['from'][idx]))
                to_edge.append(int(edge_lists['to'][idx]))
                edge_attributes.append(edge_lists['relation'][idx])
                edge_attributes_embedding.append(edge_lists['relation_' + embedding_type][idx])
        assert len(from_edge) == len(to_edge), "Edge from/to lists must have same length"
        edge_idx.append(from_edge)
        edge_idx.append(to_edge)
        return edge_idx, edge_attributes, edge_attributes_embedding

    def extract_edges_scanscribe(self, edges, embedding_type='ada'):
        """Extracts edges from a ScanScribe/human edge list.

        Args:
            edges: List of edge dicts with ``'source'``, ``'target'``,
                ``'relationship'``, and ``'relation_<embedding_type>'`` fields.
            embedding_type: Embedding key suffix (``'ada'`` or ``'word2vec'``).

        Returns:
            Tuple of (edge_idx, edge_relations, edge_features) where edge_idx
            is ``[from_ids, to_ids]``.
        """
        edge_idx = []
        from_edge = []
        to_edge = []
        edge_attributes = []
        edge_attributes_embedding = []
        for idx in range(len(edges)):
            from_edge.append(int(edges[idx]['source']))
            to_edge.append(int(edges[idx]['target']))
            edge_attributes.append(edges[idx]['relationship'])
            edge_attributes_embedding.append(edges[idx]['relation_' + embedding_type])
        assert len(from_edge) == len(to_edge), "Edge from/to lists must have same length"
        edge_idx.append(from_edge)
        edge_idx.append(to_edge)
        return edge_idx, edge_attributes, edge_attributes_embedding

    def get_subgraph(self, node_ids, return_graph=False):
        """Extracts a subgraph containing only the specified nodes and their edges.

        Args:
            node_ids: List of node IDs to include in the subgraph.
            return_graph: If True, returns a new SceneGraph object instead of
                raw components.

        Returns:
            If ``return_graph`` is False, returns a tuple of
            ``(nodes, node_features, edge_idx, edge_features)``.
            If ``return_graph`` is True, returns a new SceneGraph instance.
        """
        assert all([id in [int(n.idx) for n in self.nodes.values()] for id in node_ids]), \
            "All requested node IDs must exist in the graph"
        if len(node_ids) == 0:
            return None
        subgraph_nodes = {}
        subgraph_node_features = []
        subgraph_edge_ids_from = []
        subgraph_edge_ids_to = []
        subgraph_edge_features = []
        for node_id in self.nodes:
            node = self.nodes[node_id]
            if int(node.idx) in node_ids:
                subgraph_nodes[int(node.idx)] = node
                subgraph_node_features.append(node.features)
        for i, (from_idx, to_idx) in enumerate(zip(self.edge_idx[0], self.edge_idx[1])):
            if int(from_idx) in node_ids and int(to_idx) in node_ids:
                subgraph_edge_ids_from.append(int(from_idx))
                subgraph_edge_ids_to.append(int(to_idx))
                subgraph_edge_features.append(self.edge_features[i])
        subgraph_edge_ids = [subgraph_edge_ids_from, subgraph_edge_ids_to]
        assert len(subgraph_edge_ids[0]) == len(subgraph_edge_ids[1]), "Edge from/to lists must have same length"
        assert len(subgraph_edge_ids[0]) == len(subgraph_edge_features), "Edge count mismatch"
        assert check_valid_graph(subgraph_nodes, subgraph_edge_ids), "Invalid subgraph: edge references missing node"
        if return_graph:
            new_graph = SceneGraph(self.scene_id,
                                   graph_type=None,
                                   graph=None)
            new_graph.nodes = subgraph_nodes
            new_graph.edge_idx = subgraph_edge_ids
            new_graph.edge_features = subgraph_edge_features
            return new_graph
        return subgraph_nodes, subgraph_node_features, subgraph_edge_ids, subgraph_edge_features

    def to_pyg(self):
        """Converts the scene graph to PyG-compatible tensors with remapped node IDs.

        Remaps original node IDs to contiguous 0-based indices for use with
        PyTorch Geometric.

        Returns:
            Tuple of (node_features, edge_idx, edge_features) with remapped
            contiguous node indices.
        """
        assert len(self.nodes) > 0, "Graph must have at least one node"
        node_ids = [int(self.nodes[node_id].idx) for node_id in self.nodes]
        edge_ids = self.edge_idx

        nodeid_map = {}
        for idx, nodeid in enumerate(node_ids):
            nodeid_map[int(nodeid)] = idx

        edge_ids_remap = []
        edge_ids_from = []
        edge_ids_to = []
        for (from_idx, to_idx) in zip(edge_ids[0], edge_ids[1]):
            edge_ids_from.append(int(nodeid_map[int(from_idx)]))
            edge_ids_to.append(int(nodeid_map[int(to_idx)]))
        edge_ids_remap.append(edge_ids_from)
        edge_ids_remap.append(edge_ids_to)

        node_features = [self.nodes[node_id].features for node_id in self.nodes]
        return node_features, edge_ids_remap, self.edge_features

    def get_node_features(self):
        """Returns the feature vectors for all nodes in the graph.

        Returns:
            List of feature vectors, one per node.
        """
        node_features = [self.nodes[node_id].features for node_id in self.nodes]
        return node_features


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default=os.environ.get('WHEREAMI_DATA_ROOT', './data'),
                        help='Root data directory')
    cli_args = parser.parse_args()
    from pathlib import Path
    graphs_dir = Path(cli_args.data_root) / 'processed_data'

    ######## 3DSSG #########
    _3dssg_scenes = torch.load(graphs_dir / '3dssg' / '3dssg_graphs_processed_edgelists_relationembed.pt', weights_only=False)
    for sceneid in tqdm(_3dssg_scenes):
        sg = SceneGraph(sceneid,
                        graph_type='3dssg',
                        graph=_3dssg_scenes[sceneid],
                        max_dist=1.0, embedding_type='ada')

    ######### ScanScribe #########
    scanscribe_scenes = torch.load(graphs_dir / 'scanscribe' / 'scanscribe_cleaned_original_node_edge_features.pt', weights_only=False)
    for scene_id in tqdm(scanscribe_scenes):
        txtids = scanscribe_scenes[scene_id].keys()
        assert len(set(txtids)) == len(txtids), "Duplicate text IDs found"
        assert len(set(txtids)) == len(range(max([int(id) for id in txtids]) + 1)), "Non-contiguous text IDs"
        for txt_id in txtids:
            sg = SceneGraph(scene_id,
                            txt_id=txt_id,
                            graph_type='scanscribe',
                            graph=scanscribe_scenes[scene_id][txt_id],
                            embedding_type='ada')
