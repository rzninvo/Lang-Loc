import numpy as np
from sklearn.cluster import DBSCAN

from langloc.retrieval.scene_graph import SceneGraph


def combine_node_features(graph1, graph2):
    node_features1 = graph1.get_node_features()
    node_features2 = graph2.get_node_features()
    all_node_features = np.concatenate((node_features1, node_features2), axis=0)
    all_node_graph_index = np.concatenate((np.zeros(len(node_features1)), np.ones(len(node_features2))), axis=0) # graph1 is 0, graph2 is 1
    return all_node_features, all_node_graph_index

def get_matching_subgraph(graph1, graph2):
    # Cluster the nodes in both graphs with dbscan
    all_node_features, all_node_graph_index = combine_node_features(graph1, graph2)
    combined_node_idx = np.concatenate(([n1 for n1 in graph1.nodes], [n2 for n2 in graph2.nodes]), axis=0)
    assert(all([i == graph1.nodes[i].idx for i in graph1.nodes])) # key equals the idx
    assert(all([i == graph2.nodes[i].idx for i in graph2.nodes]))
    idx_mapping = {}
    for i, idx in enumerate(combined_node_idx):
        idx_mapping[i] = idx

    # Track the indices of the nodes that are matched, after combining into all_node_features
    clustering = DBSCAN(eps=0.5, min_samples=1, metric='cosine').fit(all_node_features) # default 0.05
    clusters = {}
    for i, cluster in enumerate(clustering.labels_):
        if cluster in clusters:
            clusters[cluster].append(i)
        else:
            clusters[cluster] = [i]

    # Process the clusters so that only clusters with nodes from both graphs remain
    graph1_keep_nodes = []
    graph2_keep_nodes = []
    for cluster in clusters:
        indices = clusters[cluster]
        graphs = [int(all_node_graph_index[i]) for i in indices]
        if 0 in graphs and 1 in graphs:
            graph1_keep_nodes.extend([idx_mapping[i] for i in indices if int(all_node_graph_index[i]) == 0])
            graph2_keep_nodes.extend([idx_mapping[i] for i in indices if int(all_node_graph_index[i]) == 1])

    # Get the subgraph
    assert(type(graph1) == SceneGraph)
    assert(type(graph2) == SceneGraph)        
    graph1_keep_nodes = list(set(graph1_keep_nodes))
    graph2_keep_nodes = list(set(graph2_keep_nodes))
    subgraph1 = graph1.get_subgraph(graph1_keep_nodes, return_graph=True)
    subgraph2 = graph2.get_subgraph(graph2_keep_nodes, return_graph=True)

    return subgraph1, subgraph2
