"""Edge-aware GATv2 layers with residual connections for scene graph encoding."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class EdgeGATLayer(nn.Module):
    """Single GATv2 layer with edge features and a residual connection.

    Args:
        in_dim: Input node feature dimension.
        out_dim: Output node feature dimension (before head concatenation).
        heads: Number of attention heads.
        edge_dim: Edge feature dimension (8 for geometric, 512 for CLIP text).
        dropout: Dropout probability for attention weights.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int, edge_dim: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.gat = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False
        )

        self.norm = nn.Identity()
        self.act = nn.GELU()

        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Applies GATv2 convolution with residual connection and activation.

        Args:
            x: Node feature tensor of shape ``(num_nodes, in_dim)``.
            edge_index: Edge index tensor of shape ``(2, num_edges)``.
            edge_attr: Edge attribute tensor of shape ``(num_edges, edge_dim)``.

        Returns:
            Updated node features of shape ``(num_nodes, out_dim)``.
        """
        out = self.gat(x, edge_index, edge_attr)
        out = out + self.res_proj(x)
        out = self.act(out)

        return out


class MultiGAT_Edge(nn.Module):
    """Multi-layer edge-aware GATv2 stack.

    Args:
        n_units: List of hidden dimensions per layer (length = num_layers + 1).
        n_heads: List of attention head counts per layer (length = num_layers).
        edge_dim: Edge feature dimension passed to each layer.
        dropout: Dropout probability for attention weights.
    """

    def __init__(self, n_units: list[int], n_heads: list[int], edge_dim: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(len(n_units) - 1):

            in_dim  = n_units[i]
            out_dim = n_units[i+1]

            heads = n_heads[i]

            self.layers.append(
                EdgeGATLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout
                )
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Passes node features through all GATv2 layers sequentially.

        Args:
            x: Node feature tensor of shape ``(num_nodes, in_dim)``.
            edge_index: Edge index tensor of shape ``(2, num_edges)``.
            edge_attr: Edge attribute tensor of shape ``(num_edges, edge_dim)``.

        Returns:
            Updated node features of shape ``(num_nodes, out_dim)`` after all layers.
        """
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x