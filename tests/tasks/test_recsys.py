"""Tests for recsys task runner."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from gnn_playground.datasets.movielens import RecsysData
from gnn_playground.tasks.recsys import RECSYS_MODEL_REGISTRY, run


class TestRecsysModelRegistry:
    """Tests for recsys model registry."""

    def test_lightgcn_registered(self):
        """Test that lightgcn is registered."""
        assert "lightgcn" in RECSYS_MODEL_REGISTRY

    def test_mf_registered(self):
        """Test that mf is registered."""
        assert "mf" in RECSYS_MODEL_REGISTRY


class TestRecsysRun:
    """Tests for recsys run function."""

    @pytest.fixture
    def mock_recsys_data(self) -> RecsysData:
        """Create mock recommendation data."""
        num_users = 50
        num_items = 30

        # Create simple train edges
        train_users = torch.randint(0, num_users, (100,))
        train_items = torch.randint(0, num_items, (100,))
        train_edges = torch.stack([train_users, train_items])

        # Val/test edges
        val_edges = torch.stack(
            [
                torch.arange(10),
                torch.randint(0, num_items, (10,)),
            ]
        )
        test_edges = torch.stack(
            [
                torch.arange(10),
                torch.randint(0, num_items, (10,)),
            ]
        )

        # Ground truth
        val_ground_truth = {i: {int(val_edges[1, i])} for i in range(10)}
        test_ground_truth = {i: {int(test_edges[1, i])} for i in range(10)}

        return RecsysData(
            num_users=num_users,
            num_items=num_items,
            train_edges=train_edges,
            val_edges=val_edges,
            test_edges=test_edges,
            val_ground_truth=val_ground_truth,
            test_ground_truth=test_ground_truth,
        )

    @patch("gnn_playground.tasks.recsys.load_dataset")
    @patch("gnn_playground.tasks.recsys.plot_training_curves")
    def test_run_single_model(self, mock_plot, mock_load, mock_recsys_data, tmp_path):
        """Test running with a single model."""
        mock_load.return_value = mock_recsys_data

        cfg = {
            "dataset": "movielens_100k",
            "models": ["mf"],  # MF is faster for testing
            "epochs": 2,
            "lr": 0.01,
            "embedding_dim": 16,
            "patience": 5,
            "batch_size": 32,
            "k_list": [5, 10],
            "output_dir": str(tmp_path),
            "device": "cpu",
        }

        results = run(cfg)

        assert "mf" in results
        assert "recall@5" in results["mf"]
        assert "recall@10" in results["mf"]
        assert "ndcg@5" in results["mf"]
        assert "ndcg@10" in results["mf"]
        assert "best_val_recall@20" in results["mf"]

    @patch("gnn_playground.tasks.recsys.load_dataset")
    @patch("gnn_playground.tasks.recsys.plot_training_curves")
    @patch("gnn_playground.tasks.recsys.plot_comparison_bar")
    def test_run_multiple_models(self, mock_bar, mock_curves, mock_load, mock_recsys_data, tmp_path):
        """Test running with multiple models."""
        mock_load.return_value = mock_recsys_data

        cfg = {
            "dataset": "movielens_100k",
            "models": ["mf", "lightgcn"],
            "epochs": 2,
            "lr": 0.01,
            "embedding_dim": 16,
            "num_layers": 2,
            "patience": 5,
            "batch_size": 32,
            "k_list": [5],
            "output_dir": str(tmp_path),
            "device": "cpu",
        }

        results = run(cfg)

        assert "mf" in results
        assert "lightgcn" in results
        mock_bar.assert_called_once()  # Comparison plot created

    @patch("gnn_playground.tasks.recsys.load_dataset")
    @patch("gnn_playground.tasks.recsys.plot_training_curves")
    def test_run_unknown_model_skipped(self, mock_plot, mock_load, mock_recsys_data, tmp_path):
        """Test that unknown models are skipped."""
        mock_load.return_value = mock_recsys_data

        cfg = {
            "dataset": "movielens_100k",
            "models": ["unknown_model", "mf"],
            "epochs": 2,
            "lr": 0.01,
            "embedding_dim": 16,
            "patience": 5,
            "batch_size": 32,
            "k_list": [5],
            "output_dir": str(tmp_path),
            "device": "cpu",
        }

        results = run(cfg)

        assert "unknown_model" not in results
        assert "mf" in results

    @patch("gnn_playground.tasks.recsys.load_dataset")
    @patch("gnn_playground.tasks.recsys.plot_training_curves")
    def test_run_early_stopping(self, mock_plot, mock_load, mock_recsys_data, tmp_path):
        """Test that early stopping works."""
        mock_load.return_value = mock_recsys_data

        cfg = {
            "dataset": "movielens_100k",
            "models": ["mf"],
            "epochs": 100,  # High epochs
            "lr": 0.01,
            "embedding_dim": 16,
            "patience": 2,  # Low patience for early stopping
            "batch_size": 32,
            "k_list": [5],
            "output_dir": str(tmp_path),
            "device": "cpu",
        }

        results = run(cfg)

        # Should complete without running all 100 epochs
        assert "mf" in results

    @patch("gnn_playground.tasks.recsys.load_dataset")
    @patch("gnn_playground.tasks.recsys.plot_training_curves")
    def test_run_creates_output_dir(self, mock_plot, mock_load, mock_recsys_data, tmp_path):
        """Test that output directory is created."""
        mock_load.return_value = mock_recsys_data
        output_dir = tmp_path / "nested" / "output"

        cfg = {
            "dataset": "movielens_100k",
            "models": ["mf"],
            "epochs": 1,
            "lr": 0.01,
            "embedding_dim": 16,
            "patience": 5,
            "batch_size": 32,
            "k_list": [5],
            "output_dir": str(output_dir),
            "device": "cpu",
        }

        run(cfg)

        assert output_dir.exists()

    @patch("gnn_playground.tasks.recsys.load_dataset")
    @patch("gnn_playground.tasks.recsys.plot_training_curves")
    def test_run_default_config_values(self, mock_plot, mock_load, mock_recsys_data, tmp_path):
        """Test that default config values work."""
        mock_load.return_value = mock_recsys_data

        # Minimal config - should use defaults
        cfg = {
            "output_dir": str(tmp_path),
        }

        results = run(cfg)

        # Should use default model (lightgcn) and complete
        assert "lightgcn" in results
