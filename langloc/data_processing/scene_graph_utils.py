"""Validation utilities for scene graph data structures."""


def check_valid_graph(nodes, edge_idx):
    """Checks that every node referenced by an edge exists in the node set.

    Args:
        nodes: Dictionary mapping node IDs to Node objects.
        edge_idx: List of two lists ``[from_ids, to_ids]``, where each is
            a list of integer node IDs.

    Returns:
        True if all edge endpoints are present in nodes, False otherwise.
    """
    assert type(nodes) == dict, "nodes must be a dict"
    assert type(edge_idx) == list, "edge_idx must be a list"
    nodeids = {int(node_id) for node_id in nodes}
    edgeids = set()
    for from_idx, to_idx in zip(edge_idx[0], edge_idx[1]):
        edgeids.add(int(from_idx))
        edgeids.add(int(to_idx))

    return edgeids.issubset(nodeids)
