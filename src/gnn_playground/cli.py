"""CLI entry point for GNN Playground."""

from __future__ import annotations

import typer
from rich.console import Console

from gnn_playground.config import build_config
from gnn_playground.tasks import TASK_REGISTRY, run_task
from gnn_playground.training.utils import get_device, set_seed

app = typer.Typer(
    name="gnn-playground",
    help="GNN Playground: modular GNN experiments with unified CLI.",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """GNN Playground: modular GNN experiments with unified CLI."""
    if ctx.invoked_subcommand is None:
        console.print("[bold blue]GNN Playground[/bold blue] - use 'run' command to start an experiment.")
        console.print("Run [bold]gnn-playground --help[/bold] for usage info.")
        raise typer.Exit(code=0)


@app.command()
def run(
    task: str | None = typer.Option(None, "--task", "-t", help="Task to run (e.g. node_classification)"),
    dataset: str | None = typer.Option(None, "--dataset", "-d", help="Dataset name (e.g. cora)"),
    model: str | None = typer.Option(None, "--model", "-m", help="Single model to use (overrides config list)"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
    epochs: int | None = typer.Option(None, "--epochs", help="Number of training epochs"),
    lr: float | None = typer.Option(None, "--lr", help="Learning rate"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed"),
    hidden_dim: int | None = typer.Option(None, "--hidden-dim", help="Hidden layer dimension"),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Output directory"),
) -> None:
    """Run a GNN experiment."""
    # Build config from defaults + YAML + CLI
    cli_overrides = {
        "task": task,
        "dataset": dataset,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "hidden_dim": hidden_dim,
        "output_dir": output_dir,
    }

    # Handle single model override
    if model is not None:
        cli_overrides["models"] = [model]  # type: ignore[assignment]

    cfg = build_config(config_path=config, **cli_overrides)

    # Validate required fields
    if not cfg.get("task"):
        console.print("[red]Error:[/red] --task is required (or set 'task' in config YAML)")
        raise typer.Exit(code=1)

    if cfg["task"] not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys())) if TASK_REGISTRY else "(none registered yet)"
        console.print(f"[red]Error:[/red] Unknown task '{cfg['task']}'. Available: {available}")
        raise typer.Exit(code=1)

    # Set seed and device
    set_seed(cfg["seed"])
    if cfg["device"] == "auto":
        cfg["device"] = str(get_device())

    # Print experiment info
    console.print("\n[bold blue]GNN Playground[/bold blue]")
    console.print(f"  Task:    {cfg['task']}")
    console.print(f"  Dataset: {cfg.get('dataset', 'N/A')}")
    console.print(f"  Models:  {cfg.get('models', 'N/A')}")
    console.print(f"  Device:  {cfg['device']}")
    console.print(f"  Seed:    {cfg['seed']}")
    console.print()

    # Dispatch to task
    run_task(cfg["task"], cfg)


if __name__ == "__main__":
    app()
