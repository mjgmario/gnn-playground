"""Smoke tests: verify package imports work."""


def test_import_package():
    import gnn_playground

    assert hasattr(gnn_playground, "__version__")


def test_import_cli():
    from gnn_playground.cli import app

    assert app is not None


def test_import_config():
    from gnn_playground.config import BaseConfig, build_config, load_config

    assert BaseConfig is not None
    assert load_config is not None
    assert build_config is not None


def test_import_registries():
    from gnn_playground.datasets import DATASET_REGISTRY
    from gnn_playground.models import MODEL_REGISTRY
    from gnn_playground.tasks import TASK_REGISTRY

    assert isinstance(DATASET_REGISTRY, dict)
    assert isinstance(MODEL_REGISTRY, dict)
    assert isinstance(TASK_REGISTRY, dict)


def test_import_training_utils():
    from gnn_playground.training.utils import EarlyStopping, get_device, set_seed

    assert callable(set_seed)
    assert callable(get_device)
    assert EarlyStopping is not None


def test_import_metrics():
    from gnn_playground.training.metrics import (
        compute_classification_metrics,
        compute_community_metrics,
        compute_fraud_metrics,
        compute_link_metrics,
        compute_ranking_metrics,
    )

    assert callable(compute_classification_metrics)
    assert callable(compute_link_metrics)
    assert callable(compute_ranking_metrics)
    assert callable(compute_fraud_metrics)
    assert callable(compute_community_metrics)


def test_import_viz():
    from gnn_playground.viz.plots import (
        plot_class_distribution,
        plot_comparison_bar,
        plot_pr_curve,
        plot_roc_curve,
        plot_training_curves,
    )

    assert callable(plot_training_curves)
    assert callable(plot_comparison_bar)
    assert callable(plot_roc_curve)
    assert callable(plot_pr_curve)
    assert callable(plot_class_distribution)
