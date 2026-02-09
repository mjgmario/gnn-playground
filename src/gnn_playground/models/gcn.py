"""Graph Convolutional Network (GCN) model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Graph Convolutional Network for node classification.

    :param in_channels: Number of input features.
    :param hidden_channels: Number of hidden units.
    :param out_channels: Number of output classes.
    :param num_layers: Number of GCN layers.
    :param dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Node features [num_nodes, in_channels].
        :param edge_index: Graph connectivity [2, num_edges].
        :return: Node logits [num_nodes, out_channels].
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
