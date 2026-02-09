"""Tests for Planetoid dataset loader."""

from __future__ import annotations

import pytest

from gnn_playground.datasets import DATASET_REGISTRY, load_dataset
from gnn_playground.datasets.planetoid import load_planetoid


class TestLoadPlanetoid:
    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown Planetoid"):
            load_planetoid("invalid_name")

    def test_cora_registered(self):
        assert "cora" in DATASET_REGISTRY

    def test_citeseer_registered(self):
        assert "citeseer" in DATASET_REGISTRY

    def test_pubmed_registered(self):
        assert "pubmed" in DATASET_REGISTRY


@pytest.mark.slow
class TestPlanetoidRealData:
    """Integration tests that download real data."""

    def test_cora_loads(self, tmp_path):
        data = load_planetoid("cora", root=str(tmp_path))
        # Check required attributes
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "y")
        assert hasattr(data, "train_mask")
        assert hasattr(data, "val_mask")
        assert hasattr(data, "test_mask")
        # Check shapes
        assert data.x.shape[0] == data.y.shape[0]  # num_nodes
        assert data.edge_index.shape[0] == 2
        # Check metadata
        assert hasattr(data, "num_classes")
        assert hasattr(data, "num_features")
        assert data.num_classes == 7  # Cora has 7 classes
        assert data.num_features == 1433  # Cora has 1433 features

    def test_load_via_registry(self, tmp_path):
        data = load_dataset("cora", root=str(tmp_path))
        assert data.x is not None
        assert data.edge_index is not None
