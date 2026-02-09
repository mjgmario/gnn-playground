"""Graph visualization for community detection."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def _build_graph(edge_index: np.ndarray, num_nodes: int) -> nx.Graph:
    """Build a NetworkX graph from edge index array."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    G.add_edges_from(edges)
    return G


def draw_communities(
    edge_index: torch.Tensor | np.ndarray,
    partition: dict[int, int],
    num_nodes: int,
    title: str = "Community Detection",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 10),
    node_size: int = 50,
    edge_alpha: float = 0.1,
    seed: int = 42,
) -> None:
    """Draw a graph colored by community assignment.

    :param edge_index: Edge index tensor [2, num_edges].
    :param partition: Dict mapping node_id -> community_id.
    :param num_nodes: Total number of nodes.
    :param title: Plot title.
    :param save_path: Path to save the figure (optional).
    :param figsize: Figure size as (width, height).
    :param node_size: Size of nodes in the plot.
    :param edge_alpha: Transparency of edges.
    :param seed: Random seed for layout reproducibility.
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    G = _build_graph(edge_index, num_nodes)

    # Get community colors
    communities = set(partition.values())
    num_communities = len(communities)
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(num_communities, 1))

    # Map community IDs to colors
    community_to_color = {c: i for i, c in enumerate(sorted(communities))}
    node_colors = [cmap(community_to_color.get(partition.get(node, 0), 0)) for node in range(num_nodes)]

    # Compute layout
    pos = nx.spring_layout(G, seed=seed, k=1.0 / np.sqrt(num_nodes) if num_nodes > 0 else 1.0)

    # Draw
    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_edges(G, pos, alpha=edge_alpha, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, ax=ax)

    ax.set_title(f"{title}\n({num_communities} communities)")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)


def draw_communities_grid(
    edge_index: torch.Tensor | np.ndarray,
    partitions: dict[str, dict[int, int]],
    num_nodes: int,
    title: str = "Community Detection Comparison",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (16, 16),
    node_size: int = 30,
    edge_alpha: float = 0.1,
    seed: int = 42,
) -> None:
    """Draw multiple partitions in a grid layout.

    :param edge_index: Edge index tensor [2, num_edges].
    :param partitions: Dict mapping method_name -> partition dict.
    :param num_nodes: Total number of nodes.
    :param title: Overall plot title.
    :param save_path: Path to save the figure (optional).
    :param figsize: Figure size as (width, height).
    :param node_size: Size of nodes in the plot.
    :param edge_alpha: Transparency of edges.
    :param seed: Random seed for layout reproducibility.
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    G = _build_graph(edge_index, num_nodes)

    # Compute shared layout
    pos = nx.spring_layout(G, seed=seed, k=1.0 / np.sqrt(num_nodes) if num_nodes > 0 else 1.0)

    # Determine grid size
    n_methods = len(partitions)
    n_cols = min(2, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_methods == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    cmap = plt.colormaps.get_cmap("tab20").resampled(20)

    for idx, (method_name, partition) in enumerate(partitions.items()):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        # Get community colors for this partition
        communities = set(partition.values())
        num_communities = len(communities)
        community_to_color = {c: i % 20 for i, c in enumerate(sorted(communities))}
        node_colors = [cmap(community_to_color.get(partition.get(node, 0), 0)) for node in range(num_nodes)]

        nx.draw_networkx_edges(G, pos, alpha=edge_alpha, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, ax=ax)

        ax.set_title(f"{method_name}\n({num_communities} communities)")
        ax.axis("off")

    # Hide empty subplots
    for idx in range(n_methods, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
