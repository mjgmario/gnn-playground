"""Tests for fraud detection task runner."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data

from gnn_playground.datasets.elliptic import EllipticData
from gnn_playground.tasks.fraud import FRAUD_MODEL_REGISTRY, _train_logreg, run


class TestFraudModelRegistry:
    """Tests for fraud model registry."""

    def test_graphsage_registered(self):
        """Test that graphsage is registered."""
        assert "graphsage" in FRAUD_MODEL_REGISTRY

    def test_gat_registered(self):
        """Test that gat is registered."""
        assert "gat" in FRAUD_MODEL_REGISTRY


class TestTrainLogreg:
    """Tests for logistic regression baseline."""

    @pytest.fixture
    def mock_data(self):
        """Create mock imbalanced data with both classes in train/test."""
        num_nodes = 100
        x = torch.randn(num_nodes, 10)
        edge_index = torch.randint(0, num_nodes, (2, 200))

        # Imbalanced labels: 80 licit, 20 illicit
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[80:] = 1

        # Ensure both classes in train and test
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[:60] = True  # 60 licit
        train_mask[80:90] = True  # 10 illicit

        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[60:80] = True  # 20 licit
        test_mask[90:] = True  # 10 illicit

        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=torch.zeros(num_nodes, dtype=torch.bool),
            test_mask=test_mask,
            num_nodes=num_nodes,
        )

    def test_returns_metrics_dict(self, mock_data):
        """Test that logreg returns metrics dict."""
        metrics = _train_logreg(mock_data, precision_target=0.9)

        assert isinstance(metrics, dict)
        assert "pr_auc" in metrics
        assert "f1" in metrics
        assert "recall@precision=0.9" in metrics

    def test_metrics_in_range(self, mock_data):
        """Test that metrics are in valid ranges."""
        metrics = _train_logreg(mock_data, precision_target=0.9)

        assert 0 <= metrics["pr_auc"] <= 1
        assert 0 <= metrics["f1"] <= 1


class TestFraudRun:
    """Tests for fraud run function."""

    @pytest.fixture
    def mock_elliptic_data(self):
        """Create mock Elliptic data with both classes in each split."""
        num_nodes = 60
        x = torch.randn(num_nodes, 20)
        edge_index = torch.randint(0, num_nodes, (2, 100))

        # Labels: 40 licit (0), 15 illicit (1), 5 unknown (-1)
        # Arrange so both classes are in train/val/test
        y = torch.zeros(num_nodes, dtype=torch.long)
        # Licit: 0-39, Illicit: 40-54, Unknown: 55-59
        y[40:55] = 1
        y[55:] = -1

        # Train: 0-24 (licit) + 40-47 (illicit) = 25 licit + 8 illicit
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[:25] = True
        train_mask[40:48] = True

        # Val: 25-32 (licit) + 48-51 (illicit) = 8 licit + 4 illicit
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[25:33] = True
        val_mask[48:52] = True

        # Test: 33-39 (licit) + 52-54 (illicit) = 7 licit + 3 illicit
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[33:40] = True
        test_mask[52:55] = True

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
            num_licit=40,
            num_illicit=15,
            num_unknown=5,
            timesteps=torch.randint(1, 5, (num_nodes,)),
        )

    @patch("gnn_playground.tasks.fraud.load_dataset")
    @patch("gnn_playground.tasks.fraud.plot_class_distribution")
    @patch("gnn_playground.tasks.fraud.plot_training_curves")
    @patch("gnn_playground.tasks.fraud.plot_pr_curve")
    def test_run_logreg_only(self, mock_pr, mock_curves, mock_dist, mock_load, mock_elliptic_data, tmp_path):
        """Test running with logreg only."""
        mock_load.return_value = mock_elliptic_data

        cfg = {
            "dataset": "elliptic",
            "models": ["logreg"],
            "output_dir": str(tmp_path),
        }

        results = run(cfg)

        assert "logreg" in results
        assert "pr_auc" in results["logreg"]
        assert "f1" in results["logreg"]

    @patch("gnn_playground.tasks.fraud.load_dataset")
    @patch("gnn_playground.tasks.fraud.plot_class_distribution")
    @patch("gnn_playground.tasks.fraud.plot_training_curves")
    @patch("gnn_playground.tasks.fraud.plot_pr_curve")
    def test_run_graphsage(self, mock_pr, mock_curves, mock_dist, mock_load, mock_elliptic_data, tmp_path):
        """Test running with GraphSAGE."""
        mock_load.return_value = mock_elliptic_data

        cfg = {
            "dataset": "elliptic",
            "models": ["graphsage"],
            "epochs": 2,
            "lr": 0.01,
            "hidden_dim": 16,
            "patience": 5,
            "output_dir": str(tmp_path),
        }

        results = run(cfg)

        assert "graphsage" in results
        assert "pr_auc" in results["graphsage"]
        mock_curves.assert_called()
        mock_pr.assert_called()

    @patch("gnn_playground.tasks.fraud.load_dataset")
    @patch("gnn_playground.tasks.fraud.plot_class_distribution")
    @patch("gnn_playground.tasks.fraud.plot_training_curves")
    @patch("gnn_playground.tasks.fraud.plot_pr_curve")
    def test_run_multiple_models(self, mock_pr, mock_curves, mock_dist, mock_load, mock_elliptic_data, tmp_path):
        """Test running with multiple models."""
        mock_load.return_value = mock_elliptic_data

        cfg = {
            "dataset": "elliptic",
            "models": ["logreg", "graphsage"],
            "epochs": 2,
            "hidden_dim": 16,
            "patience": 5,
            "output_dir": str(tmp_path),
        }

        results = run(cfg)

        assert "logreg" in results
        assert "graphsage" in results

    @patch("gnn_playground.tasks.fraud.load_dataset")
    @patch("gnn_playground.tasks.fraud.plot_class_distribution")
    @patch("gnn_playground.tasks.fraud.plot_training_curves")
    @patch("gnn_playground.tasks.fraud.plot_pr_curve")
    def test_run_skips_unknown_model(self, mock_pr, mock_curves, mock_dist, mock_load, mock_elliptic_data, tmp_path):
        """Test that unknown models are skipped."""
        mock_load.return_value = mock_elliptic_data

        cfg = {
            "dataset": "elliptic",
            "models": ["unknown_model", "logreg"],
            "output_dir": str(tmp_path),
        }

        results = run(cfg)

        assert "unknown_model" not in results
        assert "logreg" in results

    @patch("gnn_playground.tasks.fraud.load_dataset")
    @patch("gnn_playground.tasks.fraud.plot_class_distribution")
    @patch("gnn_playground.tasks.fraud.plot_training_curves")
    @patch("gnn_playground.tasks.fraud.plot_pr_curve")
    def test_run_with_class_weights(self, mock_pr, mock_curves, mock_dist, mock_load, mock_elliptic_data, tmp_path):
        """Test running with class weights enabled."""
        mock_load.return_value = mock_elliptic_data

        cfg = {
            "dataset": "elliptic",
            "models": ["graphsage"],
            "epochs": 2,
            "hidden_dim": 16,
            "patience": 5,
            "use_class_weights": True,
            "output_dir": str(tmp_path),
        }

        results = run(cfg)

        assert "graphsage" in results

    @patch("gnn_playground.tasks.fraud.load_dataset")
    @patch("gnn_playground.tasks.fraud.plot_class_distribution")
    @patch("gnn_playground.tasks.fraud.plot_training_curves")
    @patch("gnn_playground.tasks.fraud.plot_pr_curve")
    def test_run_creates_output_dir(self, mock_pr, mock_curves, mock_dist, mock_load, mock_elliptic_data, tmp_path):
        """Test that output directory is created."""
        mock_load.return_value = mock_elliptic_data
        output_dir = tmp_path / "nested" / "output"

        cfg = {
            "dataset": "elliptic",
            "models": ["logreg"],
            "output_dir": str(output_dir),
        }

        run(cfg)

        assert output_dir.exists()

    @patch("gnn_playground.tasks.fraud.load_dataset")
    @patch("gnn_playground.tasks.fraud.plot_class_distribution")
    @patch("gnn_playground.tasks.fraud.plot_training_curves")
    @patch("gnn_playground.tasks.fraud.plot_pr_curve")
    def test_run_custom_precision_target(
        self, mock_pr, mock_curves, mock_dist, mock_load, mock_elliptic_data, tmp_path
    ):
        """Test running with custom precision target."""
        mock_load.return_value = mock_elliptic_data

        cfg = {
            "dataset": "elliptic",
            "models": ["logreg"],
            "precision_target": 0.8,
            "output_dir": str(tmp_path),
        }

        results = run(cfg)

        assert "recall@precision=0.8" in results["logreg"]

    @patch("gnn_playground.tasks.fraud.load_dataset")
    @patch("gnn_playground.tasks.fraud.plot_class_distribution")
    @patch("gnn_playground.tasks.fraud.plot_training_curves")
    @patch("gnn_playground.tasks.fraud.plot_pr_curve")
    def test_metrics_in_range(self, mock_pr, mock_curves, mock_dist, mock_load, mock_elliptic_data, tmp_path):
        """Test that all metrics are in valid ranges."""
        mock_load.return_value = mock_elliptic_data

        cfg = {
            "dataset": "elliptic",
            "models": ["logreg", "graphsage"],
            "epochs": 2,
            "hidden_dim": 16,
            "patience": 5,
            "output_dir": str(tmp_path),
        }

        results = run(cfg)

        for model_name, metrics in results.items():
            assert 0 <= metrics["pr_auc"] <= 1, f"{model_name} pr_auc out of range"
            assert 0 <= metrics["f1"] <= 1, f"{model_name} f1 out of range"
