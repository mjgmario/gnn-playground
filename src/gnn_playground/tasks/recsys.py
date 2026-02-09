"""Recommendation system task runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table
from torch.optim import Adam

from gnn_playground.datasets import load_dataset
from gnn_playground.models.lightgcn import LightGCN, MatrixFactorization
from gnn_playground.training.loops import eval_recsys, train_recsys_epoch
from gnn_playground.training.utils import EarlyStopping
from gnn_playground.viz.plots import plot_comparison_bar, plot_training_curves

console = Console()

RECSYS_MODEL_REGISTRY = {
    "lightgcn": LightGCN,
    "mf": MatrixFactorization,
}


def run(cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Run recommendation experiment.

    :param cfg: Configuration dict with keys: dataset, models, epochs, lr, embedding_dim,
                weight_decay, patience, output_dir, device, seed, k_list.
    :return: Results dict mapping model_name -> {metric: value}.
    """
    # Extract config
    dataset_name = cfg.get("dataset", "movielens_100k")
    model_names = cfg.get("models", ["lightgcn"])
    epochs = cfg.get("epochs", 100)
    lr = cfg.get("lr", 0.001)
    embedding_dim = cfg.get("embedding_dim", 64)
    weight_decay = cfg.get("weight_decay", 1e-5)
    patience = cfg.get("patience", 20)
    batch_size = cfg.get("batch_size", 1024)
    k_list = cfg.get("k_list", [10, 20, 50])
    output_dir = Path(cfg.get("output_dir", "outputs/recsys"))
    device = torch.device(cfg.get("device", "cpu"))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    console.print(f"[cyan]Loading dataset: {dataset_name}[/cyan]")
    data = load_dataset(dataset_name, root=cfg.get("data_root", "data"))

    console.print(f"  Users: {data.num_users}, Items: {data.num_items}")
    console.print(f"  Train edges: {data.train_edges.shape[1]}")
    console.print(f"  Val users: {len(data.val_ground_truth)}")
    console.print(f"  Test users: {len(data.test_ground_truth)}")

    results: dict[str, dict[str, float]] = {}

    # Train each model
    for model_name in model_names:
        console.print(f"\n[bold green]Training {model_name}...[/bold green]")

        # Instantiate model
        model_cls = RECSYS_MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            console.print(f"[red]Unknown model: {model_name}. Skipping.[/red]")
            continue

        model_kwargs = {
            "num_users": data.num_users,
            "num_items": data.num_items,
            "embedding_dim": embedding_dim,
        }
        if model_name == "lightgcn":
            model_kwargs["num_layers"] = cfg.get("num_layers", 3)

        model = model_cls(**model_kwargs).to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        early_stopping = EarlyStopping(patience=patience, mode="max")

        # Training history
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_recall@20": [],
        }

        best_val_recall = 0.0
        best_model_state = None

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = train_recsys_epoch(
                model,
                data.train_edges,
                optimizer,
                data.num_items,
                data.num_users,
                batch_size,
                device,
            )
            history["train_loss"].append(train_loss)

            # Validate
            val_metrics = eval_recsys(
                model,
                data.train_edges,
                data.val_ground_truth,
                data.num_users,
                data.num_items,
                k_list,
                device,
            )
            val_recall = val_metrics.get("recall@20", 0.0)
            history["val_recall@20"].append(val_recall)

            # Track best model
            if val_recall > best_val_recall:
                best_val_recall = val_recall
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Progress
            if epoch % 20 == 0 or epoch == 1:
                console.print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}, val_recall@20={val_recall:.4f}")

            # Early stopping
            if early_stopping(val_recall):
                console.print(f"  [yellow]Early stopping at epoch {epoch}[/yellow]")
                break

        # Load best model and evaluate on test set
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model = model.to(device)

        test_metrics = eval_recsys(
            model,
            data.train_edges,
            data.test_ground_truth,
            data.num_users,
            data.num_items,
            k_list,
            device,
        )

        results[model_name] = {
            **test_metrics,
            "best_val_recall@20": best_val_recall,
        }

        console.print(
            f"  [bold]Test Recall@20: {test_metrics.get('recall@20', 0):.4f}, "
            f"NDCG@20: {test_metrics.get('ndcg@20', 0):.4f}[/bold]"
        )

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
            metric="recall@20",
            save_path=output_dir / "model_comparison.png",
            title=f"Model Comparison on {dataset_name}",
        )

    # Print results table
    _print_results_table(results, dataset_name, k_list)

    return results


def _print_results_table(results: dict[str, dict[str, float]], dataset_name: str, k_list: list[int]) -> None:
    """Print a rich table with results."""
    table = Table(title=f"Recommendation Results - {dataset_name}")
    table.add_column("Model", style="cyan")
    for k in k_list:
        table.add_column(f"Recall@{k}", justify="right")
        table.add_column(f"NDCG@{k}", justify="right")

    for model_name, metrics in results.items():
        row = [model_name]
        for k in k_list:
            row.append(f"{metrics.get(f'recall@{k}', 0):.4f}")
            row.append(f"{metrics.get(f'ndcg@{k}', 0):.4f}")
        table.add_row(*row)

    console.print()
    console.print(table)
