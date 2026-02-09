"""Dataset registry and loader."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from gnn_playground.datasets.elliptic import load_elliptic
from gnn_playground.datasets.movielens import load_movielens
from gnn_playground.datasets.planetoid import load_planetoid
from gnn_playground.datasets.snap_email import load_email_eu_core, load_email_eu_core_single
from gnn_playground.datasets.tudatasets import load_tu_dataset

DATASET_REGISTRY: dict[str, Callable[..., Any]] = {
    # Node classification (Planetoid)
    "cora": partial(load_planetoid, name="cora"),
    "citeseer": partial(load_planetoid, name="citeseer"),
    "pubmed": partial(load_planetoid, name="pubmed"),
    # Graph classification (TUDataset)
    "mutag": partial(load_tu_dataset, name="mutag"),
    "proteins": partial(load_tu_dataset, name="proteins"),
    # Link prediction (SNAP)
    "email_eu_core": load_email_eu_core,
    # Community detection (SNAP) - same dataset, no split
    "email_eu_core_full": load_email_eu_core_single,
    # Recommendation (MovieLens)
    "movielens_100k": load_movielens,
    # Fraud detection (Elliptic)
    "elliptic": load_elliptic,
}


def load_dataset(name: str, **kwargs: Any) -> Any:
    """Load a dataset by name from the registry.

    :param name: Dataset name (e.g. 'cora', 'mutag').
    :raises KeyError: If dataset is not registered.
    :return: Loaded dataset (typically PyG Data or DataLoader).
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name](**kwargs)
