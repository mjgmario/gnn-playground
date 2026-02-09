"""Elliptic Bitcoin dataset loader for fraud detection.

The Elliptic dataset requires manual download from Kaggle:
https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

Expected files in data/elliptic/:
- elliptic_txs_features.csv
- elliptic_txs_edgelist.csv
- elliptic_txs_classes.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


class EllipticData(NamedTuple):
    """Container for Elliptic dataset."""

    data: Data  # PyG Data with x, edge_index, y, masks
    num_licit: int
    num_illicit: int
    num_unknown: int
    timesteps: torch.Tensor  # Timestep for each node


def load_elliptic(
    root: str = "data",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> EllipticData:
    """Load Elliptic Bitcoin transaction dataset.

    Uses temporal split: first timesteps for training, middle for validation,
    last for testing. Unknown labels are excluded from all masks.

    :param root: Root directory containing elliptic/ subfolder with CSVs.
    :param train_ratio: Ratio of timesteps for training.
    :param val_ratio: Ratio of timesteps for validation.
    :param seed: Random seed for reproducibility.
    :return: EllipticData with PyG Data and metadata.
    """
    root_path = Path(root) / "elliptic"

    features_file = root_path / "elliptic_txs_features.csv"
    edges_file = root_path / "elliptic_txs_edgelist.csv"
    classes_file = root_path / "elliptic_txs_classes.csv"

    # Check files exist
    for f in [features_file, edges_file, classes_file]:
        if not f.exists():
            raise FileNotFoundError(
                f"Elliptic dataset file not found: {f}\n"
                "Please download from: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set\n"
                f"and extract to: {root_path}"
            )

    # Load features
    # Format: txId, timestep, 165 local features, 72 aggregated features
    features_df = pd.read_csv(features_file, header=None)
    tx_ids = features_df.iloc[:, 0].values
    timesteps = features_df.iloc[:, 1].values.astype(np.int64)
    features = features_df.iloc[:, 2:].values.astype(np.float32)

    # Create txId to index mapping
    tx_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}
    num_nodes = len(tx_ids)

    # Load edges
    edges_df = pd.read_csv(edges_file)
    src_col = edges_df.columns[0]
    dst_col = edges_df.columns[1]

    src_mapped = edges_df[src_col].map(tx_to_idx)
    dst_mapped = edges_df[dst_col].map(tx_to_idx)
    valid = src_mapped.notna() & dst_mapped.notna()
    if valid.any():
        edge_index = torch.tensor(
            np.column_stack([src_mapped[valid].astype(int).values, dst_mapped[valid].astype(int).values]),
            dtype=torch.long,
        ).T
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Load classes
    # Format: txId, class (1=licit, 2=illicit, unknown)
    classes_df = pd.read_csv(classes_file)
    tx_col = classes_df.columns[0]
    class_col = classes_df.columns[1]

    # Map labels: unknown -> -1, licit (1) -> 0, illicit (2) -> 1
    labels = np.full(num_nodes, -1, dtype=np.int64)
    class_indices = classes_df[tx_col].map(tx_to_idx)
    class_values = classes_df[class_col].astype(str).str.strip()
    valid_class = class_indices.notna()
    for idx, label_str in zip(class_indices[valid_class].astype(int), class_values[valid_class], strict=True):
        if label_str == "1":
            labels[idx] = 0  # licit
        elif label_str == "2":
            labels[idx] = 1  # illicit

    # Create tensors
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels)
    timesteps_tensor = torch.from_numpy(timesteps)

    # Temporal split based on timesteps
    unique_timesteps = sorted(set(timesteps))
    n_timesteps = len(unique_timesteps)

    train_end = int(n_timesteps * train_ratio)
    val_end = int(n_timesteps * (train_ratio + val_ratio))

    train_timesteps = set(unique_timesteps[:train_end])
    val_timesteps = set(unique_timesteps[train_end:val_end])
    test_timesteps = set(unique_timesteps[val_end:])

    # Create masks (exclude unknown labels)
    known_mask = y >= 0
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for i in range(num_nodes):
        if not known_mask[i]:
            continue
        ts = timesteps[i]
        if ts in train_timesteps:
            train_mask[i] = True
        elif ts in val_timesteps:
            val_mask[i] = True
        elif ts in test_timesteps:
            test_mask[i] = True

    # Count labels
    num_licit = int((y == 0).sum())
    num_illicit = int((y == 1).sum())
    num_unknown = int((y == -1).sum())

    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
        num_classes=2,
    )

    return EllipticData(
        data=data,
        num_licit=num_licit,
        num_illicit=num_illicit,
        num_unknown=num_unknown,
        timesteps=timesteps_tensor,
    )


def compute_class_weights(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute class weights for imbalanced classification.

    :param y: Label tensor.
    :param mask: Mask for nodes to consider.
    :return: Class weights tensor [num_classes].
    """
    y_masked = y[mask]
    num_classes = int(y_masked.max().item()) + 1

    weights = []
    total = len(y_masked)

    for c in range(num_classes):
        count = int((y_masked == c).sum())
        if count > 0:
            weight = total / (num_classes * count)
        else:
            weight = 1.0
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)
