"""Email-Eu-core dataset loader from SNAP for link prediction."""

from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

EDGE_URL = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"
LABELS_URL = "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz"


def download_file(url: str, path: Path) -> None:
    """Download a file from URL to path."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)


def load_email_eu_core(
    root: str = "data",
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    add_negative_train_samples: bool = True,
    seed: int = 42,
) -> tuple[Data, Data, Data]:
    """Load Email-Eu-core dataset and split for link prediction.

    :param root: Root directory for data storage.
    :param val_ratio: Ratio of edges for validation.
    :param test_ratio: Ratio of edges for test.
    :param add_negative_train_samples: Whether to add negative samples in training.
    :param seed: Random seed for reproducibility.
    :return: Tuple of (train_data, val_data, test_data).
    """
    root_path = Path(root) / "email_eu_core"
    root_path.mkdir(parents=True, exist_ok=True)

    edge_file = root_path / "email-Eu-core.txt.gz"
    labels_file = root_path / "email-Eu-core-department-labels.txt.gz"

    # Download files
    download_file(EDGE_URL, edge_file)
    download_file(LABELS_URL, labels_file)

    # Parse edge list
    edges = []
    with gzip.open(edge_file, "rt") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    src, dst = int(parts[0]), int(parts[1])
                    edges.append((src, dst))

    # Parse department labels
    node_labels = {}
    with gzip.open(labels_file, "rt") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    node_id, dept = int(parts[0]), int(parts[1])
                    node_labels[node_id] = dept

    # Build edge_index
    edges_arr = np.array(edges, dtype=np.int64)
    edge_index = torch.from_numpy(edges_arr.T).contiguous()

    # Get number of nodes
    num_nodes = int(edge_index.max().item()) + 1

    # Create node features based on degree
    # In-degree and out-degree normalized
    in_degree = torch.zeros(num_nodes)
    out_degree = torch.zeros(num_nodes)

    for src, dst in edges_arr:
        out_degree[src] += 1
        in_degree[dst] += 1

    # Normalize degrees
    in_degree_norm = in_degree / (in_degree.max() + 1e-8)
    out_degree_norm = out_degree / (out_degree.max() + 1e-8)
    total_degree_norm = (in_degree + out_degree) / ((in_degree + out_degree).max() + 1e-8)

    # Stack as features
    x = torch.stack([in_degree_norm, out_degree_norm, total_degree_norm], dim=1).float()

    # Create labels tensor
    y = torch.full((num_nodes,), fill_value=-1, dtype=torch.long)
    for node_id, dept in node_labels.items():
        if node_id < num_nodes:
            y[node_id] = dept

    # Create base Data object
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

    # Apply RandomLinkSplit
    torch.manual_seed(seed)
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=False,
        add_negative_train_samples=add_negative_train_samples,
        neg_sampling_ratio=1.0,
    )

    train_data, val_data, test_data = transform(data)

    return train_data, val_data, test_data


def load_email_eu_core_single(root: str = "data") -> Data:
    """Load Email-Eu-core as a single Data object (for community detection).

    :param root: Root directory for data storage.
    :return: Data object with x, edge_index, y (department labels).
    """
    train_data, _, _ = load_email_eu_core(root=root, val_ratio=0.0, test_ratio=0.0)
    return train_data
