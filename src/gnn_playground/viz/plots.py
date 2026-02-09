"""Plotting functions for training curves, ROC/PR, and comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Consistent style
sns.set_theme(style="whitegrid")
PLOT_DPI = 150


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | Path,
    title: str = "Training Curves",
) -> None:
    """Plot loss and metric curves over epochs.

    :param history: Dict with keys like 'train_loss', 'val_loss', 'val_accuracy', etc.
                    Each value is a list of per-epoch values.
    :param save_path: Path to save the figure.
    :param title: Figure title.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Separate loss and metric keys
    loss_keys = [k for k in history if "loss" in k]
    metric_keys = [k for k in history if "loss" not in k]

    n_plots = (1 if loss_keys else 0) + (1 if metric_keys else 0)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    idx = 0
    if loss_keys:
        ax = axes[idx]
        for key in loss_keys:
            ax.plot(history[key], label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend()
        idx += 1

    if metric_keys:
        ax = axes[idx]
        for key in metric_keys:
            ax.plot(history[key], label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        ax.set_title("Metrics")
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bar(
    results: dict[str, dict[str, float]],
    metric: str,
    save_path: str | Path,
    title: str | None = None,
) -> None:
    """Plot a bar chart comparing models on a given metric.

    :param results: Dict mapping model_name -> {metric_name: value, ...}.
    :param metric: Which metric to plot.
    :param save_path: Path to save the figure.
    :param title: Optional title (defaults to metric name).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(results.keys())
    values = [results[name].get(metric, 0.0) for name in names]

    fig, ax = plt.subplots(figsize=(max(4, len(names) * 1.5), 4))
    bars = ax.bar(names, values, color=sns.color_palette("viridis", len(names)))

    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or metric.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(
    fpr: np.ndarray | list[float],
    tpr: np.ndarray | list[float],
    auc_score: float,
    save_path: str | Path,
    title: str = "ROC Curve",
) -> None:
    """Plot ROC curve with AUC score.

    :param fpr: False positive rates.
    :param tpr: True positive rates.
    :param auc_score: Area under the ROC curve.
    :param save_path: Path to save the figure.
    :param title: Figure title.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(
    precision: np.ndarray | list[float],
    recall: np.ndarray | list[float],
    ap_score: float,
    save_path: str | Path,
    title: str = "Precision-Recall Curve",
) -> None:
    """Plot Precision-Recall curve with AP score.

    :param precision: Precision values.
    :param recall: Recall values.
    :param ap_score: Average precision score.
    :param save_path: Path to save the figure.
    :param title: Figure title.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"AP = {ap_score:.4f}", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1.05))
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(
    labels: np.ndarray | list[int],
    save_path: str | Path,
    class_names: list[str] | None = None,
    title: str = "Class Distribution",
) -> None:
    """Plot class distribution as a bar chart.

    :param labels: Array of class labels.
    :param save_path: Path to save the figure.
    :param class_names: Optional list of class names.
    :param title: Figure title.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(labels, list):
        labels = np.array(labels)

    unique, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(max(4, len(unique) * 1.2), 4))
    x_labels = class_names if class_names and len(class_names) == len(unique) else [str(u) for u in unique]
    bars = ax.bar(x_labels, counts, color=sns.color_palette("Set2", len(unique)))

    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha="center", va="bottom", fontsize=9
        )

    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
