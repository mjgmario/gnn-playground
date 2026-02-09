"""MovieLens 100K dataset loader for recommendation."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import NamedTuple

import torch

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


class RecsysData(NamedTuple):
    """Container for recommendation system data."""

    num_users: int
    num_items: int
    train_edges: torch.Tensor  # [2, num_train_edges]
    val_edges: torch.Tensor  # [2, num_val_edges]
    test_edges: torch.Tensor  # [2, num_test_edges]
    # For evaluation: ground truth items per user
    val_ground_truth: dict[int, set[int]]
    test_ground_truth: dict[int, set[int]]


def download_movielens(root: Path) -> Path:
    """Download MovieLens 100K dataset."""
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "ml-100k.zip"
    data_dir = root / "ml-100k"

    if not data_dir.exists():
        if not zip_path.exists():
            urllib.request.urlretrieve(ML100K_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(root)

    return data_dir


def load_movielens(
    root: str = "data",
    rating_threshold: int = 4,
    seed: int = 42,
) -> RecsysData:
    """Load MovieLens 100K dataset for implicit feedback recommendation.

    Uses leave-last-out split: for each user, last interaction -> test,
    second-to-last -> val, rest -> train.

    :param root: Root directory for data storage.
    :param rating_threshold: Minimum rating to consider as positive interaction.
    :param seed: Random seed for reproducibility.
    :return: RecsysData with train/val/test edges and ground truth.
    """
    root_path = Path(root) / "movielens"
    data_dir = download_movielens(root_path)

    # Parse u.data: user_id, item_id, rating, timestamp
    ratings = []
    with open(data_dir / "u.data") as f:
        for line in f:
            parts = line.strip().split("\t")
            user_id = int(parts[0]) - 1  # 0-indexed
            item_id = int(parts[1]) - 1  # 0-indexed
            rating = int(parts[2])
            timestamp = int(parts[3])
            if rating >= rating_threshold:
                ratings.append((user_id, item_id, timestamp))

    # Sort by user and timestamp
    ratings.sort(key=lambda x: (x[0], x[2]))

    # Group by user
    user_interactions: dict[int, list[int]] = {}
    for user_id, item_id, _ in ratings:
        if user_id not in user_interactions:
            user_interactions[user_id] = []
        user_interactions[user_id].append(item_id)

    # Leave-last-out split
    train_edges = []
    val_edges = []
    test_edges = []
    val_ground_truth: dict[int, set[int]] = {}
    test_ground_truth: dict[int, set[int]] = {}

    for user_id, items in user_interactions.items():
        if len(items) >= 3:
            # Last -> test, second-to-last -> val, rest -> train
            test_item = items[-1]
            val_item = items[-2]
            train_items = items[:-2]

            for item in train_items:
                train_edges.append((user_id, item))
            val_edges.append((user_id, val_item))
            test_edges.append((user_id, test_item))

            val_ground_truth[user_id] = {val_item}
            test_ground_truth[user_id] = {test_item}
        elif len(items) == 2:
            # Only train and test
            train_edges.append((user_id, items[0]))
            test_edges.append((user_id, items[1]))
            test_ground_truth[user_id] = {items[1]}
        elif len(items) == 1:
            # Only train
            train_edges.append((user_id, items[0]))

    # Get counts
    all_users = {u for u, _ in train_edges + val_edges + test_edges}
    all_items = {i for _, i in train_edges + val_edges + test_edges}
    num_users = max(all_users) + 1 if all_users else 0
    num_items = max(all_items) + 1 if all_items else 0

    # Convert to tensors
    train_tensor = (
        torch.tensor(train_edges, dtype=torch.long).T if train_edges else torch.zeros((2, 0), dtype=torch.long)
    )
    val_tensor = torch.tensor(val_edges, dtype=torch.long).T if val_edges else torch.zeros((2, 0), dtype=torch.long)
    test_tensor = torch.tensor(test_edges, dtype=torch.long).T if test_edges else torch.zeros((2, 0), dtype=torch.long)

    return RecsysData(
        num_users=num_users,
        num_items=num_items,
        train_edges=train_tensor,
        val_edges=val_tensor,
        test_edges=test_tensor,
        val_ground_truth=val_ground_truth,
        test_ground_truth=test_ground_truth,
    )
