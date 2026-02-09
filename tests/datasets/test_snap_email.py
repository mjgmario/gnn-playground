"""Tests for Email-Eu-core dataset loader."""

from __future__ import annotations

import pytest

from gnn_playground.datasets import DATASET_REGISTRY


class TestSnapEmailLoader:
    def test_email_eu_core_registered(self):
        assert "email_eu_core" in DATASET_REGISTRY


@pytest.mark.slow
class TestSnapEmailRealData:
    """Integration tests that download real data."""

    def test_loads_and_splits(self, tmp_path):
        from gnn_playground.datasets.snap_email import load_email_eu_core

        train_data, val_data, test_data = load_email_eu_core(root=str(tmp_path))

        # Check train data
        assert hasattr(train_data, "x")
        assert hasattr(train_data, "edge_index")
        assert hasattr(train_data, "edge_label_index")
        assert hasattr(train_data, "edge_label")

        # Check val data
        assert hasattr(val_data, "edge_label_index")
        assert hasattr(val_data, "edge_label")

        # Check test data
        assert hasattr(test_data, "edge_label_index")
        assert hasattr(test_data, "edge_label")

        # Check features are degree-based (3 features)
        assert train_data.x.shape[1] == 3

    def test_labels_loaded(self, tmp_path):
        from gnn_playground.datasets.snap_email import load_email_eu_core

        train_data, _, _ = load_email_eu_core(root=str(tmp_path))

        # Check y (department labels) exists
        assert hasattr(train_data, "y")
        assert train_data.y.shape[0] == train_data.num_nodes
