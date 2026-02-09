"""Tests for link prediction task runner."""

from __future__ import annotations

from gnn_playground.tasks import TASK_REGISTRY
from gnn_playground.tasks.link_prediction import run


class TestLinkPredictionTask:
    def test_registered(self):
        assert "link_prediction" in TASK_REGISTRY

    def test_run_with_synthetic_data(self, synthetic_link_data, tmp_path):
        cfg = {
            "models": ["gcn"],
            "decoders": ["dot"],
            "epochs": 3,
            "lr": 0.01,
            "hidden_dim": 8,
            "num_layers": 2,
            "patience": 2,
            "output_dir": str(tmp_path),
            "device": "cpu",
        }

        # Patch load_dataset to return our synthetic data
        import gnn_playground.tasks.link_prediction as lp_module

        original_load = lp_module.load_dataset

        def mock_load(*args, **kwargs):
            # Return train, val, test (all same for testing)
            return synthetic_link_data, synthetic_link_data, synthetic_link_data

        lp_module.load_dataset = mock_load
        try:
            results = run(cfg)
        finally:
            lp_module.load_dataset = original_load

        # Check results structure
        assert "gcn_dot" in results
        assert "auc_roc" in results["gcn_dot"]
        assert "avg_precision" in results["gcn_dot"]
        assert 0.0 <= results["gcn_dot"]["auc_roc"] <= 1.0

        # Check output files
        assert (tmp_path / "gcn_dot_training.png").exists()
        assert (tmp_path / "gcn_dot_roc.png").exists()

    def test_results_contain_expected_keys(self, synthetic_link_data, tmp_path):
        cfg = {
            "models": ["gcn"],
            "decoders": ["dot"],
            "epochs": 2,
            "lr": 0.01,
            "hidden_dim": 8,
            "output_dir": str(tmp_path),
            "device": "cpu",
        }

        import gnn_playground.tasks.link_prediction as lp_module

        original_load = lp_module.load_dataset

        def mock_load(*args, **kwargs):
            return synthetic_link_data, synthetic_link_data, synthetic_link_data

        lp_module.load_dataset = mock_load
        try:
            results = run(cfg)
        finally:
            lp_module.load_dataset = original_load

        expected_keys = {"auc_roc", "avg_precision", "best_val_auc"}
        assert expected_keys.issubset(results["gcn_dot"].keys())

    def test_multiple_model_decoder_combos(self, synthetic_link_data, tmp_path):
        cfg = {
            "models": ["gcn", "mlp"],
            "decoders": ["dot"],
            "epochs": 2,
            "lr": 0.01,
            "hidden_dim": 8,
            "output_dir": str(tmp_path),
            "device": "cpu",
        }

        import gnn_playground.tasks.link_prediction as lp_module

        original_load = lp_module.load_dataset

        def mock_load(*args, **kwargs):
            return synthetic_link_data, synthetic_link_data, synthetic_link_data

        lp_module.load_dataset = mock_load
        try:
            results = run(cfg)
        finally:
            lp_module.load_dataset = original_load

        assert "gcn_dot" in results
        assert "mlp_dot" in results
        assert (tmp_path / "model_comparison.png").exists()
