"""Planetoid dataset loader (Cora, CiteSeer, PubMed)."""

from __future__ import annotations

from pathlib import Path

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


def load_planetoid(name: str, root: str = "data") -> Data:
    """Load a Planetoid dataset.

    :param name: Dataset name ('cora', 'citeseer', 'pubmed').
    :param root: Root directory for data storage.
    :return: PyG Data object with x, edge_index, y, and train/val/test masks.
    """
    name_lower = name.lower()
    valid_names = {"cora", "citeseer", "pubmed"}
    if name_lower not in valid_names:
        raise ValueError(f"Unknown Planetoid dataset: {name}. Valid options: {valid_names}")

    root_path = Path(root) / "planetoid"
    dataset = Planetoid(root=str(root_path), name=name_lower)
    data = dataset[0]

    # Store metadata
    data.num_classes = dataset.num_classes
    data.num_features = dataset.num_features

    return data
