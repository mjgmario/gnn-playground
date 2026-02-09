"""Fraud detection task runner for Elliptic Bitcoin dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from torch import nn
from torch.optim import Adam

from gnn_playground.datasets import load_dataset
from gnn_playground.datasets.elliptic import compute_class_weights
from gnn_playground.models.gat import GAT
from gnn_playground.models.graphsage import GraphSAGE
from gnn_playground.training.loops import train_node_epoch
from gnn_playground.training.metrics import compute_fraud_metrics
from gnn_playground.training.utils import EarlyStopping
from gnn_playground.viz.plots import plot_class_distribution, plot_pr_curve, plot_training_curves

console = Console()

FRAUD_MODEL_REGISTRY = {
    "graphsage": GraphSAGE,
    "gat": GAT,
}


def run(cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Run fraud detection experiment.

    :param cfg: Configuration dict with keys: dataset, models, epochs, lr, hidden_dim,
                weight_decay, patience, output_dir, device, seed, use_class_weights.
    :return: Results dict mapping model_name -> {metric: value}.
    """
    # Extract config
    dataset_name = cfg.get("dataset", "elliptic")
    model_names = cfg.get("models", ["logreg", "graphsage"])
    epochs = cfg.get("epochs", 100)
    lr = cfg.get("lr", 0.01)
    hidden_dim = cfg.get("hidden_dim", 64)
    weight_decay = cfg.get("weight_decay", 5e-4)
    patience = cfg.get("patience", 20)
    use_class_weights = cfg.get("use_class_weights", True)
    precision_target = cfg.get("precision_target", 0.9)
    output_dir = Path(cfg.get("output_dir", "outputs/fraud"))
    device = torch.device(cfg.get("device", "cpu"))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    console.print(f"[cyan]Loading dataset: {dataset_name}[/cyan]")
    elliptic_data = load_dataset(dataset_name, root=cfg.get("data_root", "data"))

    data = elliptic_data.data
    data = data.to(device)

    console.print(f"  Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")
    console.print(
        f"  Licit: {elliptic_data.num_licit}, Illicit: {elliptic_data.num_illicit}, Unknown: {elliptic_data.num_unknown}"
    )
    console.print(f"  Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")

    # Plot class distribution
    known_labels = data.y[data.y >= 0].cpu().numpy()
    plot_class_distribution(
        known_labels,
        save_path=output_dir / "class_distribution.png",
        class_names=["Licit", "Illicit"],
        title="Elliptic Class Distribution (Known Labels)",
    )

    # Compute class weights if enabled
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(data.y, data.train_mask).to(device)
        console.print(f"  Class weights: {class_weights.tolist()}")

    results: dict[str, dict[str, float]] = {}

    # Train each model
    for model_name in model_names:
        console.print(f"\n[bold green]Training {model_name}...[/bold green]")

        if model_name == "logreg":
            # Logistic Regression baseline (no graph structure)
            metrics = _train_logreg(data, precision_target, seed=cfg.get("seed", 42))
            results["logreg"] = metrics
            console.print(f"  [bold]PR-AUC: {metrics['pr_auc']:.4f}, F1: {metrics['f1']:.4f}[/bold]")
            continue

        # GNN models
        model_cls = FRAUD_MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            console.print(f"[red]Unknown model: {model_name}. Skipping.[/red]")
            continue

        # Instantiate model
        model_kwargs = {
            "in_channels": data.x.shape[1],
            "hidden_channels": hidden_dim,
            "out_channels": 2,  # Binary classification
        }
        if model_name == "gat":
            model_kwargs["heads"] = cfg.get("gat_heads", 8)

        model = model_cls(**model_kwargs).to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Loss with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        early_stopping = EarlyStopping(patience=patience, mode="max")

        # Training history
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_pr_auc": [],
        }

        best_val_prauc = 0.0
        best_model_state = None

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = train_node_epoch(model, data, optimizer, criterion, data.train_mask)
            history["train_loss"].append(train_loss)

            # Validate
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                probs = torch.softmax(out, dim=1)
                val_scores = probs[data.val_mask, 1].cpu().numpy()
                val_labels = data.y[data.val_mask].cpu().numpy()

            val_metrics = compute_fraud_metrics(val_scores, val_labels, precision_target)
            val_prauc = val_metrics["pr_auc"]
            history["val_pr_auc"].append(val_prauc)

            # Track best model
            if val_prauc > best_val_prauc:
                best_val_prauc = val_prauc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Progress
            if epoch % 20 == 0 or epoch == 1:
                console.print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}, val_pr_auc={val_prauc:.4f}")

            # Early stopping
            if early_stopping(val_prauc):
                console.print(f"  [yellow]Early stopping at epoch {epoch}[/yellow]")
                break

        # Load best model and evaluate on test set
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model = model.to(device)

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            probs = torch.softmax(out, dim=1)
            test_scores = probs[data.test_mask, 1].cpu().numpy()
            test_labels = data.y[data.test_mask].cpu().numpy()

        test_metrics = compute_fraud_metrics(test_scores, test_labels, precision_target)
        test_metrics["best_val_pr_auc"] = best_val_prauc

        results[model_name] = test_metrics

        console.print(f"  [bold]Test PR-AUC: {test_metrics['pr_auc']:.4f}, F1: {test_metrics['f1']:.4f}[/bold]")

        # Plot training curves
        plot_training_curves(
            history,
            save_path=output_dir / f"{model_name}_training.png",
            title=f"{model_name} Training on {dataset_name}",
        )

        # Plot PR curve
        precision, recall, _ = precision_recall_curve(test_labels, test_scores)
        plot_pr_curve(
            precision,
            recall,
            test_metrics["pr_auc"],
            save_path=output_dir / f"{model_name}_pr_curve.png",
            title=f"{model_name} Precision-Recall Curve",
        )

    # Print results table
    _print_results_table(results, dataset_name, precision_target)

    return results


def _train_logreg(data, precision_target: float, seed: int = 42) -> dict[str, float]:
    """Train logistic regression baseline.

    :param data: PyG Data object.
    :param precision_target: Precision target for Recall@Precision metric.
    :return: Metrics dict.
    """
    # Extract features and labels for training
    x_train = data.x[data.train_mask].cpu().numpy()
    y_train = data.y[data.train_mask].cpu().numpy()

    x_test = data.x[data.test_mask].cpu().numpy()
    y_test = data.y[data.test_mask].cpu().numpy()

    # Train with class balancing
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
    )
    clf.fit(x_train, y_train)

    # Get probabilities for positive class
    test_probs = clf.predict_proba(x_test)[:, 1]

    return compute_fraud_metrics(test_probs, y_test, precision_target)


def _print_results_table(results: dict[str, dict[str, float]], dataset_name: str, precision_target: float) -> None:
    """Print a rich table with results."""
    table = Table(title=f"Fraud Detection Results - {dataset_name}")
    table.add_column("Model", style="cyan")
    table.add_column("PR-AUC", justify="right")
    table.add_column("F1", justify="right")
    table.add_column(f"Recall@P={precision_target}", justify="right")

    for model_name, metrics in results.items():
        table.add_row(
            model_name,
            f"{metrics.get('pr_auc', 0):.4f}",
            f"{metrics.get('f1', 0):.4f}",
            f"{metrics.get(f'recall@precision={precision_target}', 0):.4f}",
        )

    console.print()
    console.print(table)
