"""Graph-level GCN model for graph classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool


class GraphGCN(nn.Module):
    """GCN for graph-level classification with global pooling.

    :param in_channels: Number of input node features.
    :param hidden_channels: Number of hidden units.
    :param out_channels: Number of output classes.
    :param num_layers: Number of GCN layers.
    :param dropout: Dropout probability.
    :param pooling: Pooling method ('sum', 'mean').
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.pooling = pooling

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, hidden_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        :param x: Node features [num_nodes, in_channels].
        :param edge_index: Graph connectivity [2, num_edges].
        :param batch: Batch vector mapping nodes to graphs [num_nodes].
        :return: Graph-level predictions [num_graphs, out_channels].
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self.pooling == "sum":
            x = global_add_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)

        x = self.lin(x)
        return x
