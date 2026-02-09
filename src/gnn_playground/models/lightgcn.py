"""LightGCN model for collaborative filtering."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class LightGCNConv(MessagePassing):
    """LightGCN convolution layer (no transformation, no activation)."""

    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


class LightGCN(nn.Module):
    """LightGCN for collaborative filtering.

    Simplified GCN: no transformation, no activation, final embedding = mean across layers.

    :param num_users: Number of users.
    :param num_items: Number of items.
    :param embedding_dim: Embedding dimension.
    :param num_layers: Number of LightGCN layers.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        # Learnable embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # LightGCN layers (shared, no parameters)
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        :param edge_index: Bipartite edge index [2, num_edges].
                          First row: user indices, second row: item indices + num_users.
        :return: Tuple of (user_embeddings, item_embeddings).
        """
        # Concatenate user and item embeddings
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        # Collect embeddings at each layer
        all_embeddings = [x]

        for conv in self.convs:
            x = conv(x, edge_index)
            all_embeddings.append(x)

        # Mean pooling across layers
        stacked = torch.stack(all_embeddings, dim=1)
        final_embeddings = stacked.mean(dim=1)

        user_emb = final_embeddings[: self.num_users]
        item_emb = final_embeddings[self.num_users :]

        return user_emb, item_emb

    def predict(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Compute prediction scores (dot product).

        :param user_emb: User embeddings [batch_size, dim] or [num_users, dim].
        :param item_emb: Item embeddings [batch_size, dim] or [num_items, dim].
        :return: Scores.
        """
        return (user_emb * item_emb).sum(dim=-1)

    def get_embedding(
        self, edge_index: torch.Tensor, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings for specific users and items.

        :param edge_index: Bipartite edge index.
        :param user_ids: User indices.
        :param item_ids: Item indices.
        :return: Tuple of (user_emb, item_emb) for the given ids.
        """
        user_emb, item_emb = self.forward(edge_index)
        return user_emb[user_ids], item_emb[item_ids]


class MatrixFactorization(nn.Module):
    """Simple Matrix Factorization baseline.

    :param num_users: Number of users.
    :param num_items: Number of items.
    :param embedding_dim: Embedding dimension.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, edge_index: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get all user and item embeddings (edge_index ignored for MF).

        :param edge_index: Ignored (for API compatibility).
        :return: Tuple of (user_embeddings, item_embeddings).
        """
        return self.user_embedding.weight, self.item_embedding.weight

    def predict(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Compute prediction scores (dot product)."""
        return (user_emb * item_emb).sum(dim=-1)
