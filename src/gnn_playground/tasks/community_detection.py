"""Community detection task runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sklearn.cluster import KMeans, SpectralClustering

from gnn_playground.datasets import load_dataset
from gnn_playground.models.graphsage import GraphSAGE
from gnn_playground.training.metrics import compute_community_metrics
from gnn_playground.viz.graph_viz import draw_communities, draw_communities_grid

console = Console()


def _edge_index_to_networkx(edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
    """Convert PyG edge_index to NetworkX graph."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edge_index_np = edge_index.cpu().numpy()
    edges = [(int(edge_index_np[0, i]), int(edge_index_np[1, i])) for i in range(edge_index_np.shape[1])]
    G.add_edges_from(edges)
    return G


def louvain_partition(G: nx.Graph) -> dict[int, int]:
    """Run Louvain community detection.

    :param G: NetworkX graph.
    :return: Partition dict mapping node_id -> community_id.
    """
    import community as community_louvain

    return community_louvain.best_partition(G)


def spectral_partition(
    edge_index: torch.Tensor,
    num_nodes: int,
    n_clusters: int = 10,
    seed: int = 42,
) -> dict[int, int]:
    """Run spectral clustering on the graph.

    :param edge_index: Edge index tensor [2, num_edges].
    :param num_nodes: Total number of nodes.
    :param n_clusters: Number of clusters.
    :param seed: Random seed.
    :return: Partition dict mapping node_id -> community_id.
    """
    # Build adjacency matrix
    edge_index_np = edge_index.cpu().numpy()
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        adj[src, dst] = 1
        adj[dst, src] = 1  # Make undirected

    # Run spectral clustering
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=seed,
        assign_labels="kmeans",
    )
    labels = sc.fit_predict(adj)

    return {i: int(labels[i]) for i in range(num_nodes)}


def gnn_kmeans_partition(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    n_clusters: int = 10,
    hidden_dim: int = 64,
    num_layers: int = 2,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> dict[int, int]:
    """Use a GNN (GraphSAGE) to get node embeddings, then KMeans clustering.

    Uses untrained GNN with initial features (degree-based).

    :param x: Node features [num_nodes, num_features].
    :param edge_index: Edge index tensor [2, num_edges].
    :param num_nodes: Total number of nodes.
    :param n_clusters: Number of clusters for KMeans.
    :param hidden_dim: Hidden dimension of GNN.
    :param num_layers: Number of GNN layers.
    :param device: Device to use.
    :param seed: Random seed.
    :return: Partition dict mapping node_id -> community_id.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create GNN model (output dim = hidden_dim for embeddings)
    model = GraphSAGE(
        in_channels=x.shape[1],
        hidden_channels=hidden_dim,
        out_channels=hidden_dim,
        num_layers=num_layers,
        dropout=0.0,
    ).to(device)

    # Get embeddings (untrained model as feature extractor)
    model.eval()
    with torch.no_grad():
        embeddings = model(x.to(device), edge_index.to(device)).cpu().numpy()

    # Run KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    return {i: int(labels[i]) for i in range(num_nodes)}


def run(cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Run community detection experiment.

    :param cfg: Configuration dict with keys: dataset, methods, n_clusters, output_dir, device, seed.
    :return: Results dict mapping method_name -> {metric: value}.
    """
    # Extract config
    dataset_name = cfg.get("dataset", "email_eu_core_full")
    methods = cfg.get("methods", ["louvain", "spectral", "gnn_kmeans"])
    n_clusters = cfg.get("n_clusters", 10)
    output_dir = Path(cfg.get("output_dir", "outputs/community_detection"))
    device = torch.device(cfg.get("device", "cpu"))
    seed = cfg.get("seed", 42)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    console.print(f"[cyan]Loading dataset: {dataset_name}[/cyan]")
    data = load_dataset(dataset_name, root=cfg.get("data_root", "data"))

    num_nodes = data.num_nodes
    edge_index = data.edge_index
    x = data.x
    true_labels = data.y.cpu().numpy() if data.y is not None else None

    # Filter valid labels (some may be -1 for unknown)
    valid_mask = true_labels >= 0 if true_labels is not None else None
    if valid_mask is not None and true_labels is not None:
        true_labels_filtered = true_labels[valid_mask]
        console.print(f"  Nodes: {num_nodes}, Edges: {edge_index.shape[1]}")
        console.print(f"  Ground truth communities: {len(set(true_labels_filtered))}")
    else:
        true_labels_filtered = None
        console.print(f"  Nodes: {num_nodes}, Edges: {edge_index.shape[1]}")

    # Convert to NetworkX for Louvain
    G = _edge_index_to_networkx(edge_index, num_nodes)

    results: dict[str, dict[str, float]] = {}
    partitions: dict[str, dict[int, int]] = {}

    # Run each method
    for method in methods:
        console.print(f"\n[bold green]Running {method}...[/bold green]")

        partition: dict[int, int] = {}

        try:
            if method == "louvain":
                partition = louvain_partition(G)
            elif method == "spectral":
                partition = spectral_partition(edge_index, num_nodes, n_clusters=n_clusters, seed=seed)
            elif method == "gnn_kmeans":
                partition = gnn_kmeans_partition(
                    x,
                    edge_index,
                    num_nodes,
                    n_clusters=n_clusters,
                    hidden_dim=cfg.get("hidden_dim", 64),
                    num_layers=cfg.get("num_layers", 2),
                    device=device,
                    seed=seed,
                )
            else:
                console.print(f"[red]Unknown method: {method}. Skipping.[/red]")
                continue
        except (ImportError, RuntimeError) as e:
            console.print(f"[red]Error running {method}: {e}. Skipping.[/red]")
            continue

        partitions[method] = partition

        # Compute metrics
        metrics = compute_community_metrics(
            partition=partition,
            edge_index=edge_index,
            num_nodes=num_nodes,
            true_labels=true_labels_filtered.tolist() if true_labels_filtered is not None else None,
        )

        # Add number of communities
        metrics["num_communities"] = float(len(set(partition.values())))

        results[method] = metrics

        console.print(
            f"  Communities: {int(metrics['num_communities'])}, "
            f"Modularity: {metrics.get('modularity', 0):.4f}, "
            f"NMI: {metrics.get('nmi', 0):.4f}"
        )

        # Save individual visualization
        draw_communities(
            edge_index=edge_index,
            partition=partition,
            num_nodes=num_nodes,
            title=f"{method} on {dataset_name}",
            save_path=output_dir / f"{method}_communities.png",
            seed=seed,
        )

    # Save comparison grid
    if len(partitions) > 1:
        draw_communities_grid(
            edge_index=edge_index,
            partitions=partitions,
            num_nodes=num_nodes,
            title=f"Community Detection Comparison on {dataset_name}",
            save_path=output_dir / "comparison_grid.png",
            seed=seed,
        )

    # Print results table
    _print_results_table(results, dataset_name)

    return results


def _print_results_table(results: dict[str, dict[str, float]], dataset_name: str) -> None:
    """Print a rich table with results."""
    table = Table(title=f"Community Detection Results - {dataset_name}")
    table.add_column("Method", style="cyan")
    table.add_column("# Communities", justify="right")
    table.add_column("Modularity", justify="right")
    table.add_column("NMI", justify="right")

    for method, metrics in results.items():
        table.add_row(
            method,
            f"{int(metrics.get('num_communities', 0))}",
            f"{metrics.get('modularity', 0):.4f}",
            f"{metrics.get('nmi', 0):.4f}",
        )

    console.print()
    console.print(table)
