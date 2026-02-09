"""TUDataset loader (MUTAG, PROTEINS, etc.) for graph classification.

The name "TUDataset" comes from PyTorch Geometric and refers to the Technische Universität (TU) Dortmund
benchmark graph datasets hosted at https://chrsmrrs.github.io/datasets/ (a.k.a. the “TU Dortmund” datasets).

"""

from __future__ import annotations

from typing import NamedTuple

from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


class GraphDataLoaders(NamedTuple):
    """Container for train/val/test DataLoaders."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    num_features: int
    num_classes: int


def load_tu_dataset(
    name: str,
    root: str = "data",
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> GraphDataLoaders:
    """Load a TUDataset and split into train/val/test DataLoaders.

    :param name: Dataset name ('mutag', 'proteins', etc.).
    :param root: Root directory for data storage.
    :param batch_size: Batch size for DataLoaders.
    :param train_ratio: Ratio of data for training.
    :param val_ratio: Ratio of data for validation.
    :param seed: Random seed for reproducibility.
    :return: GraphDataLoaders with train, val, test loaders and metadata.
    """
    name_upper = name.upper()

    dataset = TUDataset(root=root, name=name_upper)

    # Get labels for stratified splitting
    labels = [int(data.y) for data in dataset]

    # Stratified split: first train vs (val+test), then val vs test
    indices = list(range(len(dataset)))
    test_ratio = 1.0 - train_ratio - val_ratio

    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        indices, labels, test_size=(val_ratio + test_ratio), stratify=labels, random_state=seed
    )

    # Split temp into val and test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(temp_idx, test_size=(1 - val_size), stratify=temp_labels, random_state=seed)

    # Create subsets
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return GraphDataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
    )


def load_tu_dataset_cv(
    name: str,
    root: str = "data",
    batch_size: int = 32,
    n_folds: int = 10,
    seed: int = 42,
):
    """Load a TUDataset for cross-validation.

    :param name: Dataset name ('mutag', 'proteins', etc.).
    :param root: Root directory for data storage.
    :param batch_size: Batch size for DataLoaders.
    :param n_folds: Number of folds for cross-validation.
    :param seed: Random seed for reproducibility.
    :yields: Tuple of (fold_idx, train_loader, val_loader).
    """
    name_upper = name.upper()
    dataset = TUDataset(root=root, name=name_upper)

    labels = [int(data.y) for data in dataset]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        yield fold_idx, train_loader, val_loader, dataset.num_features, dataset.num_classes
