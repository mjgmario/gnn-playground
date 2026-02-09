"""Tests for TUDataset loader."""

from __future__ import annotations

import pytest

from gnn_playground.datasets import DATASET_REGISTRY
from gnn_playground.datasets.tudatasets import GraphDataLoaders, load_tu_dataset


class TestTUDatasetLoader:
    def test_mutag_registered(self):
        assert "mutag" in DATASET_REGISTRY

    def test_proteins_registered(self):
        assert "proteins" in DATASET_REGISTRY

    def test_returns_graph_data_loaders(self, tmp_path, synthetic_graph_batch):
        # Mock test using synthetic data structure
        # Real test would download dataset
        loaders = GraphDataLoaders(
            train=None,  # type: ignore
            val=None,  # type: ignore
            test=None,  # type: ignore
            num_features=5,
            num_classes=2,
        )
        assert hasattr(loaders, "train")
        assert hasattr(loaders, "val")
        assert hasattr(loaders, "test")
        assert hasattr(loaders, "num_features")
        assert hasattr(loaders, "num_classes")


@pytest.mark.slow
class TestTUDatasetRealData:
    """Integration tests that download real data."""

    def test_mutag_loads(self, tmp_path):
        loaders = load_tu_dataset("mutag", root=str(tmp_path), batch_size=16)

        assert loaders.num_features > 0
        assert loaders.num_classes == 2  # MUTAG is binary

        # Check train loader
        assert len(loaders.train.dataset) > 0

        # Check that batched data has batch attribute
        batch = next(iter(loaders.train))
        assert hasattr(batch, "batch")
        assert hasattr(batch, "x")
        assert hasattr(batch, "edge_index")
        assert hasattr(batch, "y")

    def test_split_sizes(self, tmp_path):
        loaders = load_tu_dataset("mutag", root=str(tmp_path), train_ratio=0.8, val_ratio=0.1, batch_size=8)

        total = len(loaders.train.dataset) + len(loaders.val.dataset) + len(loaders.test.dataset)

        # MUTAG has 188 graphs
        assert total == 188

        # Check approximate split ratios
        assert len(loaders.train.dataset) / total > 0.7
        assert len(loaders.val.dataset) / total > 0.05
        assert len(loaders.test.dataset) / total > 0.05
