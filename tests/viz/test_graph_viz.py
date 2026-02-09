"""Tests for graph visualization functions."""

from __future__ import annotations

import pytest
import torch

from gnn_playground.viz.graph_viz import draw_communities, draw_communities_grid


class TestDrawCommunities:
    """Tests for draw_communities function."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph with 10 nodes."""
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 6, 7],
            ]
        )
        return edge_index, 10

    def test_creates_file(self, simple_graph, tmp_path):
        """Test that draw_communities creates a file."""
        edge_index, num_nodes = simple_graph
        partition = {i: i // 5 for i in range(num_nodes)}  # 2 communities

        save_path = tmp_path / "communities.png"
        draw_communities(edge_index, partition, num_nodes, save_path=save_path)

        assert save_path.exists()

    def test_handles_single_community(self, simple_graph, tmp_path):
        """Test with all nodes in one community."""
        edge_index, num_nodes = simple_graph
        partition = dict.fromkeys(range(num_nodes), 0)

        save_path = tmp_path / "single_community.png"
        draw_communities(edge_index, partition, num_nodes, save_path=save_path)

        assert save_path.exists()

    def test_handles_many_communities(self, simple_graph, tmp_path):
        """Test with each node in its own community."""
        edge_index, num_nodes = simple_graph
        partition = {i: i for i in range(num_nodes)}  # 10 communities

        save_path = tmp_path / "many_communities.png"
        draw_communities(edge_index, partition, num_nodes, save_path=save_path)

        assert save_path.exists()

    def test_creates_parent_dirs(self, simple_graph, tmp_path):
        """Test that parent directories are created."""
        edge_index, num_nodes = simple_graph
        partition = {i: i // 5 for i in range(num_nodes)}

        save_path = tmp_path / "nested" / "dir" / "communities.png"
        draw_communities(edge_index, partition, num_nodes, save_path=save_path)

        assert save_path.exists()

    def test_numpy_edge_index(self, simple_graph, tmp_path):
        """Test with numpy edge_index."""
        edge_index, num_nodes = simple_graph
        edge_index_np = edge_index.numpy()
        partition = {i: i // 5 for i in range(num_nodes)}

        save_path = tmp_path / "numpy_communities.png"
        draw_communities(edge_index_np, partition, num_nodes, save_path=save_path)

        assert save_path.exists()

    def test_custom_figsize(self, simple_graph, tmp_path):
        """Test with custom figure size."""
        edge_index, num_nodes = simple_graph
        partition = {i: i // 5 for i in range(num_nodes)}

        save_path = tmp_path / "custom_size.png"
        draw_communities(edge_index, partition, num_nodes, save_path=save_path, figsize=(8, 8))

        assert save_path.exists()


class TestDrawCommunitiesGrid:
    """Tests for draw_communities_grid function."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph with 10 nodes."""
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 6, 7],
            ]
        )
        return edge_index, 10

    def test_creates_file(self, simple_graph, tmp_path):
        """Test that draw_communities_grid creates a file."""
        edge_index, num_nodes = simple_graph
        partitions = {
            "Method A": {i: i // 5 for i in range(num_nodes)},
            "Method B": {i: i // 3 for i in range(num_nodes)},
        }

        save_path = tmp_path / "grid.png"
        draw_communities_grid(edge_index, partitions, num_nodes, save_path=save_path)

        assert save_path.exists()

    def test_single_partition(self, simple_graph, tmp_path):
        """Test with single partition."""
        edge_index, num_nodes = simple_graph
        partitions = {"Single": {i: i // 5 for i in range(num_nodes)}}

        save_path = tmp_path / "single_grid.png"
        draw_communities_grid(edge_index, partitions, num_nodes, save_path=save_path)

        assert save_path.exists()

    def test_four_partitions(self, simple_graph, tmp_path):
        """Test with four partitions (2x2 grid)."""
        edge_index, num_nodes = simple_graph
        partitions = {
            "Method A": {i: i // 5 for i in range(num_nodes)},
            "Method B": {i: i // 3 for i in range(num_nodes)},
            "Method C": {i: i // 2 for i in range(num_nodes)},
            "Method D": dict.fromkeys(range(num_nodes), 0),
        }

        save_path = tmp_path / "four_grid.png"
        draw_communities_grid(edge_index, partitions, num_nodes, save_path=save_path)

        assert save_path.exists()

    def test_three_partitions(self, simple_graph, tmp_path):
        """Test with three partitions (handles odd number)."""
        edge_index, num_nodes = simple_graph
        partitions = {
            "Method A": {i: i // 5 for i in range(num_nodes)},
            "Method B": {i: i // 3 for i in range(num_nodes)},
            "Method C": {i: i // 2 for i in range(num_nodes)},
        }

        save_path = tmp_path / "three_grid.png"
        draw_communities_grid(edge_index, partitions, num_nodes, save_path=save_path)

        assert save_path.exists()
