"""Tests for node classification task runner."""

from __future__ import annotations

from gnn_playground.tasks import TASK_REGISTRY
from gnn_playground.tasks.node_classification import run


class TestNodeClassificationTask:
    def test_registered(self):
        assert "node_classification" in TASK_REGISTRY

    def test_run_with_synthetic_data(self, synthetic_node_graph, tmp_path):
        # Create a minimal config
        cfg = {
            "dataset": "cora",  # Will be overridden by providing data directly
            "models": ["gcn"],
            "epochs": 5,
            "lr": 0.01,
            "hidden_dim": 8,
            "weight_decay": 5e-4,
            "patience": 3,
            "output_dir": str(tmp_path),
            "device": "cpu",
        }

        # Patch load_dataset to return our synthetic data
        import gnn_playground.tasks.node_classification as nc_module

        original_load = nc_module.load_dataset

        def mock_load(*args, **kwargs):
            # Add metadata that the real dataset would have
            synthetic_node_graph.num_classes = 3
            synthetic_node_graph.num_features = synthetic_node_graph.x.shape[1]
            return synthetic_node_graph

        nc_module.load_dataset = mock_load
        try:
            results = run(cfg)
        finally:
            nc_module.load_dataset = original_load

        # Check results structure
        assert "gcn" in results
        assert "accuracy" in results["gcn"]
        assert "f1" in results["gcn"]
        assert 0.0 <= results["gcn"]["accuracy"] <= 1.0

        # Check output files
        assert (tmp_path / "gcn_training.png").exists()

    def test_run_multiple_models(self, synthetic_node_graph, tmp_path):
        cfg = {
            "models": ["gcn", "mlp"],
            "epochs": 3,
            "lr": 0.01,
            "hidden_dim": 8,
            "weight_decay": 5e-4,
            "patience": 2,
            "output_dir": str(tmp_path),
            "device": "cpu",
        }

        import gnn_playground.tasks.node_classification as nc_module

        original_load = nc_module.load_dataset

        def mock_load(*args, **kwargs):
            synthetic_node_graph.num_classes = 3
            synthetic_node_graph.num_features = synthetic_node_graph.x.shape[1]
            return synthetic_node_graph

        nc_module.load_dataset = mock_load
        try:
            results = run(cfg)
        finally:
            nc_module.load_dataset = original_load

        assert "gcn" in results
        assert "mlp" in results
        assert (tmp_path / "model_comparison.png").exists()

    def test_results_contain_expected_keys(self, synthetic_node_graph, tmp_path):
        cfg = {
            "models": ["gcn"],
            "epochs": 2,
            "lr": 0.01,
            "hidden_dim": 8,
            "output_dir": str(tmp_path),
            "device": "cpu",
        }

        import gnn_playground.tasks.node_classification as nc_module

        original_load = nc_module.load_dataset

        def mock_load(*args, **kwargs):
            synthetic_node_graph.num_classes = 3
            synthetic_node_graph.num_features = synthetic_node_graph.x.shape[1]
            return synthetic_node_graph

        nc_module.load_dataset = mock_load
        try:
            results = run(cfg)
        finally:
            nc_module.load_dataset = original_load

        expected_keys = {"accuracy", "f1", "best_val_accuracy"}
        assert expected_keys.issubset(results["gcn"].keys())
