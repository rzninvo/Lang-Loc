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
# # ============================================================
# # 2. Graph Attention Pooling with Edge Features
# # ============================================================  
# class EdgeAttentionPool(nn.Module):
#     def __init__(self, node_dim, edge_dim):
#         super().__init__()
#         self.node_proj = nn.Linear(node_dim, node_dim)
#         self.edge_proj = nn.Linear(edge_dim, node_dim)

#         self.att = nn.Sequential(
#             nn.Linear(node_dim, node_dim // 2),
#             nn.ReLU(),
#             nn.Linear(node_dim // 2, 1)
#         )

#     def forward(self, x, edge_index, edge_attr, batch):
#         """
#         x: (N, D)
#         edge_index: (2, E)
#         edge_attr: (E, D_e)
#         batch: (N,) graph id for each node
#         """
#         # Aggregate edge features to nodes (mean)
#         row, col = edge_index
#         edge_messages = self.edge_proj(edge_attr)  # (E, D)
#         agg_edges = torch.zeros_like(x).index_add_(0, col, edge_messages)
#         deg = torch.bincount(col, minlength=x.size(0)).unsqueeze(1).clamp(min=1)
#         agg_edges = agg_edges / deg

#         # Combine node features and aggregated edge features
#         h = self.node_proj(x) + agg_edges  # (N, D)

#         # Compute attention weights
#         w = self.att(h)                # (N,1)
#         w = w - w.max()                # stability
#         w = torch.exp(w)

#         # Sum attention weights per graph
#         denom = scatter_mean(w, batch, dim=0)[batch] * len(batch)

#         pooled = scatter_mean(w * x / (denom + 1e-8), batch, dim=0)
#         return pooled   # (num_graphs, D)