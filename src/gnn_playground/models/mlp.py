"""MLP baseline model (no message passing)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Multi-layer perceptron baseline (ignores graph structure).

    :param in_channels: Number of input features.
    :param hidden_channels: Number of hidden units.
    :param out_channels: Number of output classes.
    :param num_layers: Number of linear layers.
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

        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(in_channels, out_channels))
        else:
            self.layers.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass (edge_index is ignored).

        :param x: Node features [num_nodes, in_channels].
        :param edge_index: Ignored (for API compatibility with GNNs).
        :return: Node logits [num_nodes, out_channels].
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x
