import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class EdgeGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads, edge_dim, dropout=0.0):
        super().__init__()

        self.gat = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim // heads,
            heads=heads,
            edge_dim=edge_dim,  # Use original edge_dim (8 for geom, 512 for text)
            dropout=dropout,
            add_self_loops=False
        )

        self.norm = nn.Identity()
        self.act = nn.GELU()

        # Residual requires matching dims
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index, edge_attr):

        # print("\n=== EdgeGATLayer DEBUG ===")
        # print("Input x:", x.shape)
        # print("Edge index:", None if edge_index is None else edge_index.shape)
        # print("Edge attr:", None if edge_attr is None else edge_attr.shape)

        out = self.gat(x, edge_index, edge_attr)
        # print("GAT output:", out.shape)
        out = out + self.res_proj(x)         # residual connection
        # out = self.norm(out)
        out = self.act(out)
        # print("Output after norm and activation:", out.shape)

        return out

class MultiGAT_Edge(nn.Module):
    def __init__(self, n_units, n_heads, edge_dim, dropout=0.0):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(len(n_units) - 1):

            in_dim  = n_units[i]
            out_dim = n_units[i+1]      # ← full dim, not divided!!

            heads = n_heads[i]

            self.layers.append(
                EdgeGATLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,    # ← full output dimension
                    heads=heads,        # GAT divides internally
                    edge_dim=edge_dim,
                    dropout=dropout
                )
            )

    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x