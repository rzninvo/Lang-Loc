"""Graph-to-graph matching model using Transformer convolutions with cross-attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, TransformerConv

from src.utils.utils import make_cross_graph


class SimpleTConv(MessagePassing):
    """Single Transformer convolution layer with additive aggregation.

    Applies a TransformerConv followed by LeakyReLU activation.

    Args:
        in_n: Input node feature dimension.
        in_e: Input edge feature dimension.
        out_n: Output node feature dimension.
        heads: Number of attention heads.
        dropout: Dropout probability for attention weights.
    """

    def __init__(self, in_n, in_e, out_n, heads, dropout=0.5):
        super().__init__(aggr='add')
        self.TConv = TransformerConv(in_n, out_n, concat=False, heads=heads, dropout=dropout, edge_dim=in_e)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_attr):
        """Applies Transformer convolution and activation.

        Args:
            x: Node feature tensor of shape ``(num_nodes, in_n)``.
            edge_index: Edge index tensor of shape ``(2, num_edges)``.
            edge_attr: Edge attribute tensor of shape ``(num_edges, in_e)``.

        Returns:
            Updated node features of shape ``(num_nodes, out_n)``.
        """
        x = self.TConv(x, edge_index, edge_attr)
        x = self.act(x)
        return x


class BigGNN(nn.Module):
    """Graph neural network for text-to-scene-graph matching.

    Uses N layers of self-attention (TransformerConv) on each graph followed by
    cross-attention between the two graphs, then mean-pools and feeds concatenated
    embeddings through an MLP to produce a matching score.

    Args:
        N: Number of self-attention + cross-attention layer pairs.
        heads: Number of attention heads per TransformerConv.
        embed_dim: Node and edge embedding dimension.
        dropout: Dropout probability for attention weights.
    """

    def __init__(self, N, heads, embed_dim=300, dropout=0.5):
        super().__init__()
        self.N = N
        in_n, in_e, out_n = embed_dim, embed_dim, embed_dim
        self.TSALayers = nn.ModuleList([SimpleTConv(in_n, in_e, out_n, heads, dropout) for _ in range(N)])
        self.GSALayers = nn.ModuleList([SimpleTConv(in_n, in_e, out_n, heads, dropout) for _ in range(N)])
        self.TCALayers = nn.ModuleList([SimpleTConv(in_n, in_e, out_n, heads, dropout) for _ in range(N)])
        self.GCALayers = nn.ModuleList([SimpleTConv(in_n, in_e, out_n, heads, dropout) for _ in range(N)])

        self.SceneText_MLP = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_1, x_2,
                      edge_idx_1, edge_idx_2,
                      edge_attr_1, edge_attr_2):
        """Computes a matching score between two graphs.

        Args:
            x_1: Text graph node features, shape ``(N1, embed_dim)``.
            x_2: Scene graph node features, shape ``(N2, embed_dim)``.
            edge_idx_1: Text graph edge index, shape ``(2, E1)``.
            edge_idx_2: Scene graph edge index, shape ``(2, E2)``.
            edge_attr_1: Text graph edge features, shape ``(E1, embed_dim)``.
            edge_attr_2: Scene graph edge features, shape ``(E2, embed_dim)``.

        Returns:
            Tuple of (x_1_pooled, x_2_pooled, matching_score) where pooled
            tensors have shape ``(embed_dim,)`` and matching_score is a scalar
            in ``[0, 1]``.
        """
        for i in range(self.N):
            # Self-attention
            x_1 = self.TSALayers[i](x_1, edge_idx_1, edge_attr_1)
            x_2 = self.GSALayers[i](x_2, edge_idx_2, edge_attr_2)

            # Cross-attention: concatenate graphs, apply cross-conv, then slice
            # back to original sizes
            len_x_1 = x_1.shape[0]
            len_x_2 = x_2.shape[0]
            edge_index_1_cross, edge_attr_1_cross = make_cross_graph(x_1.shape, x_2.shape)
            edge_index_2_cross, edge_attr_2_cross = make_cross_graph(x_2.shape, x_1.shape)
            x_1_cross = torch.cat((x_1, x_2), dim=0)
            x_2_cross = torch.cat((x_2, x_1), dim=0)
            dev = x_1.device
            x_1_cross = self.TCALayers[i](x_1_cross.to(dev), edge_index_1_cross.to(dev), edge_attr_1_cross.to(dev))
            x_2_cross = self.GCALayers[i](x_2_cross.to(dev), edge_index_2_cross.to(dev), edge_attr_2_cross.to(dev))
            x_1 = x_1_cross[:len_x_1]
            x_2 = x_2_cross[:len_x_2]

        # Mean pooling
        x_1_pooled = torch.mean(x_1, dim=0)
        x_2_pooled = torch.mean(x_2, dim=0)

        # Concatenate and feed into matching MLP
        x_concat = torch.cat((x_1_pooled, x_2_pooled), dim=0)
        out_matching = self.SceneText_MLP(x_concat)
        return x_1_pooled, x_2_pooled, out_matching
