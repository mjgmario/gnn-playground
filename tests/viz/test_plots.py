"""Tests for plotting functions."""

from __future__ import annotations

import numpy as np

from gnn_playground.viz.plots import (
    plot_class_distribution,
    plot_comparison_bar,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_curves,
)


class TestPlotTrainingCurves:
    def test_creates_file(self, tmp_output_dir):
        history = {"train_loss": [1.0, 0.5, 0.3], "val_accuracy": [0.5, 0.7, 0.8]}
        save_path = tmp_output_dir / "curves.png"
        plot_training_curves(history, save_path)
        assert save_path.exists()

    def test_loss_only(self, tmp_output_dir):
        history = {"train_loss": [1.0, 0.5]}
        save_path = tmp_output_dir / "loss_only.png"
        plot_training_curves(history, save_path)
        assert save_path.exists()

    def test_metric_only(self, tmp_output_dir):
        history = {"val_accuracy": [0.5, 0.7]}
        save_path = tmp_output_dir / "metric_only.png"
        plot_training_curves(history, save_path)
        assert save_path.exists()

    def test_empty_history(self, tmp_output_dir):
        save_path = tmp_output_dir / "empty.png"
        plot_training_curves({}, save_path)
        assert not save_path.exists()  # nothing to plot

    def test_creates_parent_dirs(self, tmp_output_dir):
        save_path = tmp_output_dir / "subdir" / "curves.png"
        history = {"train_loss": [1.0]}
        plot_training_curves(history, save_path)
        assert save_path.exists()


class TestPlotComparisonBar:
    def test_creates_file(self, tmp_output_dir):
        results = {"GCN": {"accuracy": 0.82}, "GAT": {"accuracy": 0.85}}
        save_path = tmp_output_dir / "comparison.png"
        plot_comparison_bar(results, "accuracy", save_path)
        assert save_path.exists()

    def test_missing_metric_uses_zero(self, tmp_output_dir):
        results = {"GCN": {"accuracy": 0.82}, "GAT": {}}
        save_path = tmp_output_dir / "missing.png"
        plot_comparison_bar(results, "accuracy", save_path)
        assert save_path.exists()

    def test_single_model(self, tmp_output_dir):
        results = {"GCN": {"accuracy": 0.82}}
        save_path = tmp_output_dir / "single.png"
        plot_comparison_bar(results, "accuracy", save_path)
        assert save_path.exists()


class TestPlotROCCurve:
    def test_creates_file(self, tmp_output_dir):
        fpr = np.array([0.0, 0.1, 0.5, 1.0])
        tpr = np.array([0.0, 0.5, 0.8, 1.0])
        save_path = tmp_output_dir / "roc.png"
        plot_roc_curve(fpr, tpr, 0.85, save_path)
        assert save_path.exists()

    def test_list_input(self, tmp_output_dir):
        save_path = tmp_output_dir / "roc_list.png"
        plot_roc_curve([0.0, 1.0], [0.0, 1.0], 1.0, save_path)
        assert save_path.exists()


class TestPlotPRCurve:
    def test_creates_file(self, tmp_output_dir):
        precision = np.array([1.0, 0.8, 0.6, 0.4])
        recall = np.array([0.0, 0.3, 0.6, 1.0])
        save_path = tmp_output_dir / "pr.png"
        plot_pr_curve(precision, recall, 0.75, save_path)
        assert save_path.exists()


class TestPlotClassDistribution:
    def test_creates_file(self, tmp_output_dir):
        labels = [0, 0, 0, 1, 1, 2]
        save_path = tmp_output_dir / "dist.png"
        plot_class_distribution(labels, save_path)
        assert save_path.exists()

    def test_numpy_input(self, tmp_output_dir):
        labels = np.array([0, 1, 1, 2, 2, 2])
        save_path = tmp_output_dir / "dist_np.png"
        plot_class_distribution(labels, save_path)
        assert save_path.exists()

    def test_with_class_names(self, tmp_output_dir):
        labels = [0, 0, 1, 1, 2]
        save_path = tmp_output_dir / "dist_names.png"
        plot_class_distribution(labels, save_path, class_names=["Licit", "Illicit", "Unknown"])
        assert save_path.exists()
