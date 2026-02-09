"""Graph Attention Network (GAT) model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """Graph Attention Network for node classification.

    :param in_channels: Number of input features.
    :param hidden_channels: Number of hidden units per head.
    :param out_channels: Number of output classes.
    :param heads: Number of attention heads.
    :param dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        # First layer: multi-head attention
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Output layer: single head, average attention
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Node features [num_nodes, in_channels].
        :param edge_index: Graph connectivity [2, num_edges].
        :return: Node logits [num_nodes, out_channels].
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
