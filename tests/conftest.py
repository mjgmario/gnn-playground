"""Shared test fixtures for all tests."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data


@pytest.fixture
def device() -> torch.device:
    """CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for test artifacts."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def synthetic_node_graph() -> Data:
    """Small synthetic graph for node-level tasks.

    20 nodes, ~50 edges, 5 features, 3 classes, with train/val/test masks.
    """
    num_nodes = 20
    num_features = 5
    num_classes = 3

    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 50))
    y = torch.randint(0, num_classes, (num_nodes,))

    # Masks: 10 train, 5 val, 5 test
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:10] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[10:15] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[15:] = True

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=num_classes,
    )


@pytest.fixture
def synthetic_graph_batch() -> Batch:
    """Batch of 8 small graphs for graph-level tasks.

    Each graph has 5-10 nodes, 3 features, binary labels.
    """
    graphs = []
    for i in range(8):
        num_nodes = 5 + i % 6  # 5 to 10
        x = torch.randn(num_nodes, 3)
        num_edges = num_nodes * 2
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        y = torch.tensor([i % 2])  # binary label
        graphs.append(Data(x=x, edge_index=edge_index, y=y))

    return Batch.from_data_list(graphs)


@pytest.fixture
def synthetic_link_data() -> Data:
    """Synthetic graph with edge labels for link prediction.

    30 nodes, with positive and negative edge_label_index.
    """
    num_nodes = 30
    num_features = 8

    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 80))

    # Simulate positive and negative labeled edges
    num_pos = 20
    num_neg = 20
    pos_edges = torch.randint(0, num_nodes, (2, num_pos))
    neg_edges = torch.randint(0, num_nodes, (2, num_neg))

    edge_label_index = torch.cat([pos_edges, neg_edges], dim=1)
    edge_label = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)])

    return Data(
        x=x,
        edge_index=edge_index,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
    )


@pytest.fixture
def synthetic_bipartite() -> dict:
    """Small bipartite graph for recommendation tasks.

    10 users, 15 items, 40 edges.
    """
    num_users = 10
    num_items = 15
    num_edges = 40

    user_ids = torch.randint(0, num_users, (num_edges,))
    item_ids = torch.randint(0, num_items, (num_edges,))

    edge_index = torch.stack([user_ids, item_ids + num_users])

    return {
        "num_users": num_users,
        "num_items": num_items,
        "edge_index": edge_index,
        "num_nodes": num_users + num_items,
    }
