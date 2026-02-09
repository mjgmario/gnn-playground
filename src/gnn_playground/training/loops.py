"""Training and evaluation loops for different task types."""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.optim import Optimizer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


def train_node_epoch(
    model: nn.Module,
    data: Data,
    optimizer: Optimizer,
    criterion: nn.Module,
    mask: torch.Tensor,
) -> float:
    """Train one epoch for node-level classification.

    :param model: GNN model.
    :param data: PyG Data object with x, edge_index, y.
    :param optimizer: Optimizer instance.
    :param criterion: Loss function (e.g. CrossEntropyLoss).
    :param mask: Boolean mask for training nodes.
    :return: Training loss (float).
    """
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = criterion(out[mask], data.y[mask])

    loss.backward()
    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def eval_node_epoch(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate model on node-level classification.

    :param model: GNN model.
    :param data: PyG Data object with x, edge_index, y.
    :param mask: Boolean mask for evaluation nodes.
    :return: Tuple of (predictions, labels) tensors.
    """
    model.eval()

    out = model(data.x, data.edge_index)
    preds = out[mask].argmax(dim=1)
    labels = data.y[mask]

    return preds, labels


# ============================================================================
# Graph-level classification
# ============================================================================


def train_graph_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train one epoch for graph-level classification.

    :param model: GNN model with graph-level output.
    :param loader: DataLoader with batched graphs.
    :param optimizer: Optimizer instance.
    :param criterion: Loss function (e.g. CrossEntropyLoss).
    :param device: Device to run on.
    :return: Average training loss (float).
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def eval_graph_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate model on graph-level classification.

    :param model: GNN model with graph-level output.
    :param loader: DataLoader with batched graphs.
    :param device: Device to run on.
    :return: Tuple of (predictions, labels) tensors.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        preds = out.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(batch.y.cpu())

    return torch.cat(all_preds), torch.cat(all_labels)


# ============================================================================
# Link prediction
# ============================================================================


def train_link_epoch(
    model: nn.Module,
    decoder: nn.Module,
    data: Data,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """Train one epoch for link prediction.

    :param model: GNN encoder model.
    :param decoder: Link decoder (e.g. DotProductDecoder).
    :param data: Training data with edge_label_index and edge_label.
    :param optimizer: Optimizer instance.
    :param device: Device to run on.
    :return: Training loss (float).
    """
    model.train()
    decoder.train()
    optimizer.zero_grad()

    data = data.to(device)

    # Encode nodes
    z = model(data.x, data.edge_index)

    # Get edge embeddings
    edge_label_index = data.edge_label_index
    z_src = z[edge_label_index[0]]
    z_dst = z[edge_label_index[1]]

    # Decode and compute loss
    scores = decoder(z_src, z_dst)
    loss = nn.functional.binary_cross_entropy_with_logits(scores, data.edge_label.float())

    loss.backward()
    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def eval_link_epoch(
    model: nn.Module,
    decoder: nn.Module,
    data: Data,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate model on link prediction.

    :param model: GNN encoder model.
    :param decoder: Link decoder.
    :param data: Evaluation data with edge_label_index and edge_label.
    :param device: Device to run on.
    :return: Tuple of (scores, labels) tensors.
    """
    model.eval()
    decoder.eval()

    data = data.to(device)

    # Encode nodes
    z = model(data.x, data.edge_index)

    # Get edge embeddings
    edge_label_index = data.edge_label_index
    z_src = z[edge_label_index[0]]
    z_dst = z[edge_label_index[1]]

    # Decode
    scores = decoder(z_src, z_dst)

    return scores.cpu(), data.edge_label.cpu()


# ============================================================================
# Recommendation (BPR)
# ============================================================================


def _build_items_per_user(train_edges: torch.Tensor) -> dict[int, set[int]]:
    """Build a dict mapping user_id -> set of interacted item_ids."""
    items_per_user: dict[int, set[int]] = {}
    for u, i in zip(train_edges[0].tolist(), train_edges[1].tolist(), strict=True):
        if u not in items_per_user:
            items_per_user[u] = set()
        items_per_user[u].add(i)
    return items_per_user


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss.

    :param pos_scores: Scores for positive (user, item) pairs.
    :param neg_scores: Scores for negative (user, item) pairs.
    :return: BPR loss value.
    """
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()


def train_recsys_epoch(
    model: nn.Module,
    train_edges: torch.Tensor,
    optimizer: Optimizer,
    num_items: int,
    num_users: int,
    batch_size: int = 1024,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Train one epoch for recommendation with BPR loss.

    :param model: LightGCN or MF model.
    :param train_edges: Training edges [2, num_edges] (user_ids, item_ids).
    :param optimizer: Optimizer instance.
    :param num_items: Total number of items (for negative sampling).
    :param num_users: Total number of users.
    :param batch_size: Batch size for training.
    :param device: Device to run on.
    :return: Average training loss.
    """
    model.train()

    # Build bipartite edge_index for message passing
    # Users: 0 to num_users-1, Items: num_users to num_users+num_items-1
    user_ids = train_edges[0]
    item_ids = train_edges[1] + num_users  # Shift item IDs

    # Create bidirectional edges for undirected bipartite graph
    edge_index = torch.stack(
        [
            torch.cat([user_ids, item_ids]),
            torch.cat([item_ids, user_ids]),
        ]
    ).to(device)

    train_items_per_user = _build_items_per_user(train_edges)

    total_loss = 0.0
    num_batches = 0

    # Shuffle edges
    perm = torch.randperm(train_edges.shape[1])
    train_edges_shuffled = train_edges[:, perm]

    for start in range(0, train_edges.shape[1], batch_size):
        end = min(start + batch_size, train_edges.shape[1])
        batch_users = train_edges_shuffled[0, start:end].to(device)
        batch_pos_items = train_edges_shuffled[1, start:end].to(device)

        # Negative sampling (with max retries to avoid infinite loop)
        neg_items_list: list[int] = []
        for u in batch_users.tolist():
            neg_item = torch.randint(0, num_items, (1,)).item()
            user_items = train_items_per_user.get(u, set())
            retries = 0
            while neg_item in user_items and retries < 100:
                neg_item = torch.randint(0, num_items, (1,)).item()
                retries += 1
            if retries >= 100:
                logger.warning("Negative sampling exhausted retries for user %d", u)
            neg_items_list.append(int(neg_item))
        batch_neg_items = torch.tensor(neg_items_list, device=device)

        optimizer.zero_grad()

        # Get embeddings
        user_emb, item_emb = model(edge_index)

        # Compute scores
        user_e = user_emb[batch_users]
        pos_item_e = item_emb[batch_pos_items]
        neg_item_e = item_emb[batch_neg_items]

        pos_scores = model.predict(user_e, pos_item_e)
        neg_scores = model.predict(user_e, neg_item_e)

        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def eval_recsys(
    model: nn.Module,
    train_edges: torch.Tensor,
    ground_truth: dict[int, set[int]],
    num_users: int,
    num_items: int,
    k_list: list[int] | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """Evaluate recommendation model.

    :param model: LightGCN or MF model.
    :param train_edges: Training edges (to exclude from recommendations).
    :param ground_truth: Dict mapping user_id -> set of ground truth item_ids.
    :param num_users: Total number of users.
    :param num_items: Total number of items.
    :param k_list: List of K values for Recall@K and NDCG@K.
    :param device: Device to run on.
    :return: Dict with metrics.
    """
    model.eval()

    # Build edge_index
    user_ids = train_edges[0]
    item_ids = train_edges[1] + num_users

    edge_index = torch.stack(
        [
            torch.cat([user_ids, item_ids]),
            torch.cat([item_ids, user_ids]),
        ]
    ).to(device)

    # Get embeddings
    user_emb, item_emb = model(edge_index)

    if k_list is None:
        from gnn_playground.training.metrics import DEFAULT_K_LIST

        k_list = DEFAULT_K_LIST

    train_items_per_user = _build_items_per_user(train_edges)

    from gnn_playground.training.metrics import compute_ranking_metrics

    # Build rankings for users with ground truth
    rankings: dict[int, list[int]] = {}
    for user_id in ground_truth.keys():
        if user_id >= num_users:
            continue

        # Compute scores for all items
        user_e = user_emb[user_id].unsqueeze(0)
        scores = model.predict(user_e.expand(num_items, -1), item_emb)

        # Mask out training items
        for train_item in train_items_per_user.get(user_id, set()):
            if train_item < num_items:
                scores[train_item] = float("-inf")

        # Get top-K items
        max_k = max(k_list)
        _, top_items = torch.topk(scores, min(max_k, num_items))
        rankings[user_id] = top_items.cpu().tolist()

    return compute_ranking_metrics(rankings, ground_truth, k_list)
