"""Helper functions for scene graph analysis: subgraph matching via DBSCAN."""

import numpy as np
from sklearn.cluster import DBSCAN

from src.data_processing.scene_graph import SceneGraph


def combine_node_features(graph1, graph2):
    """Concatenates node features from two graphs with graph-membership indices.

    Args:
        graph1: First SceneGraph.
        graph2: Second SceneGraph.

    Returns:
        Tuple of (all_node_features, graph_index) where graph_index is 0
        for graph1 nodes and 1 for graph2 nodes.
    """
    node_features1 = graph1.get_node_features()
    node_features2 = graph2.get_node_features()
    all_node_features = np.concatenate((node_features1, node_features2), axis=0)
    all_node_graph_index = np.concatenate((np.zeros(len(node_features1)), np.ones(len(node_features2))), axis=0)
    return all_node_features, all_node_graph_index


def get_matching_subgraph(graph1, graph2, dbscan_eps=0.5, dbscan_min_samples=1):
    """Extracts matching subgraphs from two scene graphs using DBSCAN clustering.

    Clusters all nodes from both graphs by feature similarity, then keeps
    only clusters containing nodes from both graphs to form subgraphs.

    Args:
        graph1: First SceneGraph (e.g. text query graph).
        graph2: Second SceneGraph (e.g. 3DSSG scene graph).
        dbscan_eps: DBSCAN epsilon parameter for cosine distance.
        dbscan_min_samples: DBSCAN minimum samples per cluster.

    Returns:
        Tuple of (subgraph1, subgraph2) as SceneGraph objects, or None
        if no matching nodes were found for a graph.
    """
    all_node_features, all_node_graph_index = combine_node_features(graph1, graph2)
    combined_node_idx = np.concatenate(([n1 for n1 in graph1.nodes], [n2 for n2 in graph2.nodes]), axis=0)
    assert all([i == graph1.nodes[i].idx for i in graph1.nodes]), \
        "graph1 node keys must equal node idx values"
    assert all([i == graph2.nodes[i].idx for i in graph2.nodes]), \
        "graph2 node keys must equal node idx values"
    idx_mapping = {}
    for i, idx in enumerate(combined_node_idx):
        idx_mapping[i] = idx

    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='cosine').fit(all_node_features)
    clusters = {}
    for i, cluster in enumerate(clustering.labels_):
        if cluster in clusters:
            clusters[cluster].append(i)
        else:
            clusters[cluster] = [i]

    graph1_keep_nodes = []
    graph2_keep_nodes = []
    for cluster in clusters:
        indices = clusters[cluster]
        graphs = [int(all_node_graph_index[i]) for i in indices]
        if 0 in graphs and 1 in graphs:
            graph1_keep_nodes.extend([idx_mapping[i] for i in indices if int(all_node_graph_index[i]) == 0])
            graph2_keep_nodes.extend([idx_mapping[i] for i in indices if int(all_node_graph_index[i]) == 1])

    assert isinstance(graph1, SceneGraph), "graph1 must be a SceneGraph"
    assert isinstance(graph2, SceneGraph), "graph2 must be a SceneGraph"
    graph1_keep_nodes = list(set(graph1_keep_nodes))
    graph2_keep_nodes = list(set(graph2_keep_nodes))
    subgraph1 = graph1.get_subgraph(graph1_keep_nodes, return_graph=True)
    subgraph2 = graph2.get_subgraph(graph2_keep_nodes, return_graph=True)

    return subgraph1, subgraph2


