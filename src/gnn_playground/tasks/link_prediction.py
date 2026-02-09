"""Link prediction task runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table
from sklearn.metrics import roc_curve
from torch.optim import Adam

from gnn_playground.datasets import load_dataset
from gnn_playground.models import get_model
from gnn_playground.models.decoders import get_decoder
from gnn_playground.training.loops import eval_link_epoch, train_link_epoch
from gnn_playground.training.metrics import compute_link_metrics
from gnn_playground.training.utils import EarlyStopping
from gnn_playground.viz.plots import plot_comparison_bar, plot_roc_curve, plot_training_curves

console = Console()


def run(cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Run link prediction experiment.

    :param cfg: Configuration dict with keys: dataset, models, decoders, epochs, lr, hidden_dim,
                weight_decay, patience, output_dir, device, seed.
    :return: Results dict mapping 'model_decoder' -> {metric: value}.
    """
    # Extract config
    dataset_name = cfg.get("dataset", "email_eu_core")
    model_names = cfg.get("models", ["gcn"])
    decoder_names = cfg.get("decoders", ["dot"])
    epochs = cfg.get("epochs", 100)
    lr = cfg.get("lr", 0.01)
    hidden_dim = cfg.get("hidden_dim", 64)
    weight_decay = cfg.get("weight_decay", 0.0)
    patience = cfg.get("patience", 20)
    output_dir = Path(cfg.get("output_dir", "outputs/link_prediction"))
    device = torch.device(cfg.get("device", "cpu"))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    console.print(f"[cyan]Loading dataset: {dataset_name}[/cyan]")
    train_data, val_data, test_data = load_dataset(dataset_name, root=cfg.get("data_root", "data"))

    in_channels = train_data.x.shape[1]
    out_channels = hidden_dim  # Embedding dimension for link prediction

    console.print(f"  Nodes: {train_data.num_nodes}")
    console.print(f"  Train edges: {train_data.edge_label_index.shape[1]}")
    console.print(f"  Val edges: {val_data.edge_label_index.shape[1]}")
    console.print(f"  Test edges: {test_data.edge_label_index.shape[1]}")

    results: dict[str, dict[str, float]] = {}

    # Train each model x decoder combination
    for model_name in model_names:
        for decoder_name in decoder_names:
            combo_name = f"{model_name}_{decoder_name}"
            console.print(f"\n[bold green]Training {combo_name}...[/bold green]")

            # Instantiate model (output = hidden_dim for embeddings)
            model_kwargs = {
                "in_channels": in_channels,
                "hidden_channels": hidden_dim,
                "out_channels": out_channels,
                "num_layers": cfg.get("num_layers", 2),
                "dropout": cfg.get("dropout", 0.5),
            }

            model = get_model(model_name, **model_kwargs).to(device)

            # Instantiate decoder
            decoder_kwargs = {}
            if decoder_name in ["mlp", "bilinear"]:
                decoder_kwargs["in_channels"] = out_channels
            decoder = get_decoder(decoder_name, **decoder_kwargs).to(device)

            # Optimizer for both model and decoder
            optimizer = Adam(
                list(model.parameters()) + list(decoder.parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )
            early_stopping = EarlyStopping(patience=patience, mode="max")

            # Training history
            history: dict[str, list[float]] = {
                "train_loss": [],
                "val_auc": [],
            }

            best_val_auc = 0.0
            best_model_state = None
            best_decoder_state = None

            for epoch in range(1, epochs + 1):
                # Train
                train_loss = train_link_epoch(model, decoder, train_data, optimizer, device)
                history["train_loss"].append(train_loss)

                # Validate
                val_scores, val_labels = eval_link_epoch(model, decoder, val_data, device)
                val_metrics = compute_link_metrics(val_scores, val_labels)
                val_auc = val_metrics["auc_roc"]
                history["val_auc"].append(val_auc)

                # Track best model
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_decoder_state = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}

                # Progress
                if epoch % 20 == 0 or epoch == 1:
                    console.print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}, val_auc={val_auc:.4f}")

                # Early stopping
                if early_stopping(val_auc):
                    console.print(f"  [yellow]Early stopping at epoch {epoch}[/yellow]")
                    break

            # Load best model and evaluate on test set
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                model = model.to(device)
            if best_decoder_state is not None:
                decoder.load_state_dict(best_decoder_state)
                decoder = decoder.to(device)

            test_scores, test_labels = eval_link_epoch(model, decoder, test_data, device)
            test_metrics = compute_link_metrics(test_scores, test_labels)

            results[combo_name] = {
                "auc_roc": test_metrics["auc_roc"],
                "avg_precision": test_metrics["avg_precision"],
                "best_val_auc": best_val_auc,
            }

            console.print(
                f"  [bold]Test AUC: {test_metrics['auc_roc']:.4f}, AP: {test_metrics['avg_precision']:.4f}[/bold]"
            )

            # Plot training curves
            plot_training_curves(
                history,
                save_path=output_dir / f"{combo_name}_training.png",
                title=f"{combo_name} Training on {dataset_name}",
            )

            # Plot ROC curve
            fpr, tpr, _ = roc_curve(test_labels.numpy(), torch.sigmoid(test_scores).numpy())
            plot_roc_curve(
                fpr,
                tpr,
                test_metrics["auc_roc"],
                save_path=output_dir / f"{combo_name}_roc.png",
                title=f"{combo_name} ROC Curve",
            )

    # Plot comparison
    if len(results) > 1:
        plot_comparison_bar(
            results,
            metric="auc_roc",
            save_path=output_dir / "model_comparison.png",
            title=f"Model Comparison on {dataset_name}",
        )

    # Print results table
    _print_results_table(results, dataset_name)

    return results


def _print_results_table(results: dict[str, dict[str, float]], dataset_name: str) -> None:
    """Print a rich table with results."""
    table = Table(title=f"Link Prediction Results - {dataset_name}")
    table.add_column("Model_Decoder", style="cyan")
    table.add_column("Test AUC", justify="right")
    table.add_column("Test AP", justify="right")
    table.add_column("Best Val AUC", justify="right")

    for combo_name, metrics in results.items():
        table.add_row(
            combo_name,
            f"{metrics['auc_roc']:.4f}",
            f"{metrics['avg_precision']:.4f}",
            f"{metrics['best_val_auc']:.4f}",
        )

    console.print()
    console.print(table)
