"""Graph classification task runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table
from torch import nn
from torch.optim import Adam

from gnn_playground.datasets import load_dataset
from gnn_playground.models import get_model
from gnn_playground.training.loops import eval_graph_epoch, train_graph_epoch
from gnn_playground.training.metrics import compute_classification_metrics
from gnn_playground.training.utils import EarlyStopping
from gnn_playground.viz.plots import plot_comparison_bar, plot_training_curves

console = Console()


def run(cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Run graph classification experiment.

    :param cfg: Configuration dict with keys: dataset, models, epochs, lr, hidden_dim,
                weight_decay, patience, output_dir, device, seed, batch_size.
    :return: Results dict mapping model_name -> {metric: value}.
    """
    # Extract config
    dataset_name = cfg.get("dataset", "mutag")
    model_names = cfg.get("models", ["gin"])
    epochs = cfg.get("epochs", 100)
    lr = cfg.get("lr", 0.01)
    hidden_dim = cfg.get("hidden_dim", 64)
    weight_decay = cfg.get("weight_decay", 0.0)
    patience = cfg.get("patience", 20)
    batch_size = cfg.get("batch_size", 32)
    output_dir = Path(cfg.get("output_dir", "outputs/graph_classification"))
    device = torch.device(cfg.get("device", "cpu"))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    console.print(f"[cyan]Loading dataset: {dataset_name}[/cyan]")
    loaders = load_dataset(dataset_name, root=cfg.get("data_root", "data"), batch_size=batch_size)

    in_channels = loaders.num_features
    out_channels = loaders.num_classes

    console.print(
        f"  Graphs: train={len(loaders.train.dataset)}, val={len(loaders.val.dataset)}, test={len(loaders.test.dataset)}"
    )
    console.print(f"  Features: {in_channels}, Classes: {out_channels}")

    results: dict[str, dict[str, float]] = {}

    # Train each model
    for model_name in model_names:
        console.print(f"\n[bold green]Training {model_name}...[/bold green]")

        # Instantiate model
        model_kwargs = {
            "in_channels": in_channels,
            "hidden_channels": hidden_dim,
            "out_channels": out_channels,
            "num_layers": cfg.get("num_layers", 3),
            "dropout": cfg.get("dropout", 0.5),
        }

        model = get_model(model_name, **model_kwargs).to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=patience, mode="max")

        # Training history
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_accuracy": [],
        }

        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = train_graph_epoch(model, loaders.train, optimizer, criterion, device)
            history["train_loss"].append(train_loss)

            # Validate
            val_preds, val_labels = eval_graph_epoch(model, loaders.val, device)
            val_metrics = compute_classification_metrics(val_preds, val_labels)
            val_acc = val_metrics["accuracy"]
            history["val_accuracy"].append(val_acc)

            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Progress
            if epoch % 20 == 0 or epoch == 1:
                console.print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}, val_acc={val_acc:.4f}")

            # Early stopping
            if early_stopping(val_acc):
                console.print(f"  [yellow]Early stopping at epoch {epoch}[/yellow]")
                break

        # Load best model and evaluate on test set
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model = model.to(device)

        test_preds, test_labels = eval_graph_epoch(model, loaders.test, device)
        test_metrics = compute_classification_metrics(test_preds, test_labels)

        results[model_name] = {
            "accuracy": test_metrics["accuracy"],
            "f1": test_metrics["f1"],
            "best_val_accuracy": best_val_acc,
        }

        console.print(f"  [bold]Test accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}[/bold]")

        # Plot training curves
        plot_training_curves(
            history,
            save_path=output_dir / f"{model_name}_training.png",
            title=f"{model_name} Training on {dataset_name}",
        )

    # Plot comparison
    if len(results) > 1:
        plot_comparison_bar(
            results,
            metric="accuracy",
            save_path=output_dir / "model_comparison.png",
            title=f"Model Comparison on {dataset_name}",
        )

    # Print results table
    _print_results_table(results, dataset_name)

    return results


def _print_results_table(results: dict[str, dict[str, float]], dataset_name: str) -> None:
    """Print a rich table with results."""
    table = Table(title=f"Graph Classification Results - {dataset_name}")
    table.add_column("Model", style="cyan")
    table.add_column("Test Accuracy", justify="right")
    table.add_column("Test F1", justify="right")
    table.add_column("Best Val Acc", justify="right")

    for model_name, metrics in results.items():
        table.add_row(
            model_name,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['best_val_accuracy']:.4f}",
        )

    console.print()
    console.print(table)
