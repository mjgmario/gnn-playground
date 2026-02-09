"""Tests for graph classification task runner."""

from __future__ import annotations

from torch_geometric.loader import DataLoader

from gnn_playground.datasets.tudatasets import GraphDataLoaders
from gnn_playground.tasks import TASK_REGISTRY
from gnn_playground.tasks.graph_classification import run


class TestGraphClassificationTask:
    def test_registered(self):
        assert "graph_classification" in TASK_REGISTRY

    def test_run_with_synthetic_data(self, synthetic_graph_batch, tmp_path):
        # Create train/val/test loaders from synthetic data
        graphs = [synthetic_graph_batch[i] for i in range(len(synthetic_graph_batch.y))]

        # Split into train/val/test
        n = len(graphs)
        train_graphs = graphs[: n // 2]
        val_graphs = graphs[n // 2 : 3 * n // 4]
        test_graphs = graphs[3 * n // 4 :]

        train_loader = DataLoader(train_graphs, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=2)
        test_loader = DataLoader(test_graphs, batch_size=2)

        loaders = GraphDataLoaders(
            train=train_loader,
            val=val_loader,
            test=test_loader,
            num_features=synthetic_graph_batch.x.shape[1],
            num_classes=2,
        )

        cfg = {
            "models": ["gin"],
            "epochs": 3,
            "lr": 0.01,
            "hidden_dim": 8,
            "num_layers": 2,
            "patience": 2,
            "output_dir": str(tmp_path),
            "device": "cpu",
        }

        # Patch load_dataset to return our synthetic loaders
        import gnn_playground.tasks.graph_classification as gc_module

        original_load = gc_module.load_dataset

        def mock_load(*args, **kwargs):
            return loaders

        gc_module.load_dataset = mock_load
        try:
            results = run(cfg)
        finally:
            gc_module.load_dataset = original_load

        # Check results structure
        assert "gin" in results
        assert "accuracy" in results["gin"]
        assert "f1" in results["gin"]
        assert 0.0 <= results["gin"]["accuracy"] <= 1.0

        # Check output files
        assert (tmp_path / "gin_training.png").exists()

    def test_results_contain_expected_keys(self, synthetic_graph_batch, tmp_path):
        graphs = [synthetic_graph_batch[i] for i in range(len(synthetic_graph_batch.y))]

        n = len(graphs)
        train_loader = DataLoader(graphs[: n // 2], batch_size=2)
        val_loader = DataLoader(graphs[n // 2 : 3 * n // 4], batch_size=2)
        test_loader = DataLoader(graphs[3 * n // 4 :], batch_size=2)

        loaders = GraphDataLoaders(
            train=train_loader,
            val=val_loader,
            test=test_loader,
            num_features=synthetic_graph_batch.x.shape[1],
            num_classes=2,
        )

        cfg = {
            "models": ["gin"],
            "epochs": 2,
            "lr": 0.01,
            "hidden_dim": 8,
            "output_dir": str(tmp_path),
            "device": "cpu",
        }

        import gnn_playground.tasks.graph_classification as gc_module

        original_load = gc_module.load_dataset

        def mock_load(*args, **kwargs):
            return loaders

        gc_module.load_dataset = mock_load
        try:
            results = run(cfg)
        finally:
            gc_module.load_dataset = original_load

        expected_keys = {"accuracy", "f1", "best_val_accuracy"}
        assert expected_keys.issubset(results["gin"].keys())
