"""Tests for community detection task runner."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data

from gnn_playground.tasks.community_detection import (
    _edge_index_to_networkx,
    gnn_kmeans_partition,
    louvain_partition,
    run,
    spectral_partition,
)


class TestEdgeIndexToNetworkx:
    """Tests for edge_index to NetworkX conversion."""

    def test_creates_correct_nodes(self):
        """Test that all nodes are created."""
        edge_index = torch.tensor([[0, 1], [1, 2]])
        G = _edge_index_to_networkx(edge_index, num_nodes=5)

        assert len(G.nodes()) == 5
        assert set(G.nodes()) == {0, 1, 2, 3, 4}

    def test_creates_correct_edges(self):
        """Test that edges are created correctly."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        G = _edge_index_to_networkx(edge_index, num_nodes=3)

        assert len(G.edges()) == 3


class TestLouvainPartition:
    """Tests for Louvain community detection."""

    def test_returns_partition_dict(self):
        """Test that Louvain returns a partition dict."""
        edge_index = torch.tensor(
            [
                [0, 0, 1, 2, 2, 3, 4, 4, 5],
                [1, 2, 2, 3, 4, 4, 5, 6, 6],
            ]
        )
        G = _edge_index_to_networkx(edge_index, num_nodes=7)

        partition = louvain_partition(G)

        assert isinstance(partition, dict)
        assert len(partition) == 7
        assert all(isinstance(v, int) for v in partition.values())

    def test_all_nodes_assigned(self):
        """Test that all nodes are assigned to a community."""
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        G = _edge_index_to_networkx(edge_index, num_nodes=5)

        partition = louvain_partition(G)

        assert set(partition.keys()) == {0, 1, 2, 3, 4}


class TestSpectralPartition:
    """Tests for spectral clustering."""

    def test_returns_partition_dict(self):
        """Test that spectral returns a partition dict."""
        edge_index = torch.tensor(
            [
                [0, 0, 1, 2, 2, 3, 4, 4, 5],
                [1, 2, 2, 3, 4, 4, 5, 6, 6],
            ]
        )

        partition = spectral_partition(edge_index, num_nodes=7, n_clusters=2)

        assert isinstance(partition, dict)
        assert len(partition) == 7

    def test_correct_number_of_clusters(self):
        """Test that correct number of clusters is created."""
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1, 2, 3, 0, 5, 6, 7, 4],
            ]
        )

        partition = spectral_partition(edge_index, num_nodes=8, n_clusters=2)

        num_communities = len(set(partition.values()))
        assert num_communities == 2

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with same seed."""
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])

        partition1 = spectral_partition(edge_index, num_nodes=4, n_clusters=2, seed=42)
        partition2 = spectral_partition(edge_index, num_nodes=4, n_clusters=2, seed=42)

        assert partition1 == partition2


class TestGnnKmeansPartition:
    """Tests for GNN + KMeans partition."""

    def test_returns_partition_dict(self):
        """Test that GNN + KMeans returns a partition dict."""
        edge_index = torch.tensor(
            [
                [0, 0, 1, 2, 2, 3, 4, 4, 5],
                [1, 2, 2, 3, 4, 4, 5, 6, 6],
            ]
        )
        x = torch.randn(7, 3)

        partition = gnn_kmeans_partition(x, edge_index, num_nodes=7, n_clusters=2)

        assert isinstance(partition, dict)
        assert len(partition) == 7

    def test_correct_number_of_clusters(self):
        """Test that correct number of clusters is created."""
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        x = torch.randn(4, 5)

        partition = gnn_kmeans_partition(x, edge_index, num_nodes=4, n_clusters=2)

        num_communities = len(set(partition.values()))
        assert num_communities == 2


class TestCommunityDetectionRun:
    """Tests for community detection run function."""

    @pytest.fixture
    def mock_data(self):
        """Create mock graph data for testing."""
        num_nodes = 20
        edge_index = torch.tensor(
            [
                [0, 0, 1, 2, 2, 3, 4, 4, 5, 10, 10, 11, 12, 12, 13, 14, 14, 15],
                [1, 2, 2, 3, 4, 4, 5, 6, 6, 11, 12, 12, 13, 14, 14, 15, 16, 16],
            ]
        )
        x = torch.randn(num_nodes, 3)
        y = torch.tensor([0] * 10 + [1] * 10)  # 2 ground truth communities

        return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

    @patch("gnn_playground.tasks.community_detection.load_dataset")
    @patch("gnn_playground.tasks.community_detection.draw_communities")
    @patch("gnn_playground.tasks.community_detection.draw_communities_grid")
    def test_run_louvain_only(self, mock_grid, mock_draw, mock_load, mock_data, tmp_path):
        """Test running with Louvain only."""
        mock_load.return_value = mock_data

        cfg = {
            "dataset": "email_eu_core_full",
            "methods": ["louvain"],
            "output_dir": str(tmp_path),
        }

        results = run(cfg)

        assert "louvain" in results
        assert "modularity" in results["louvain"]
        assert "nmi" in results["louvain"]
        assert "num_communities" in results["louvain"]

    @patch("gnn_playground.tasks.community_detection.load_dataset")
    @patch("gnn_playground.tasks.community_detection.draw_communities")
    @patch("gnn_playground.tasks.community_detection.draw_communities_grid")
    def test_run_spectral_only(self, mock_grid, mock_draw, mock_load, mock_data, tmp_path):
        """Test running with spectral only."""
        mock_load.return_value = mock_data

        cfg = {
            "dataset": "email_eu_core_full",
            "methods": ["spectral"],
            "n_clusters": 2,
            "output_dir": str(tmp_path),
        }

        results = run(cfg)

        assert "spectral" in results
        assert results["spectral"]["num_communities"] == 2

    @patch("gnn_playground.tasks.community_detection.load_dataset")
    @patch("gnn_playground.tasks.community_detection.draw_communities")
    @patch("gnn_playground.tasks.community_detection.draw_communities_grid")
    def test_run_gnn_kmeans_only(self, mock_grid, mock_draw, mock_load, mock_data, tmp_path):
        """Test running with GNN + KMeans only."""
        mock_load.return_value = mock_data

        cfg = {
            "dataset": "email_eu_core_full",
            "methods": ["gnn_kmeans"],
            "n_clusters": 2,
            "hidden_dim": 16,
            "output_dir": str(tmp_path),
        }

        results = run(cfg)

        assert "gnn_kmeans" in results
        assert results["gnn_kmeans"]["num_communities"] == 2

    @patch("gnn_playground.tasks.community_detection.load_dataset")
    @patch("gnn_playground.tasks.community_detection.draw_communities")
    @patch("gnn_playground.tasks.community_detection.draw_communities_grid")
    def test_run_multiple_methods(self, mock_grid, mock_draw, mock_load, mock_data, tmp_path):
        """Test running with multiple methods."""
        mock_load.return_value = mock_data

        cfg = {
            "dataset": "email_eu_core_full",
            "methods": ["louvain", "spectral"],
            "n_clusters": 2,
            "output_dir": str(tmp_path),
        }

        results = run(cfg)

        assert "louvain" in results
        assert "spectral" in results
        mock_grid.assert_called_once()  # Comparison grid created

    @patch("gnn_playground.tasks.community_detection.load_dataset")
    @patch("gnn_playground.tasks.community_detection.draw_communities")
    def test_run_skips_unknown_method(self, mock_draw, mock_load, mock_data, tmp_path):
        """Test that unknown methods are skipped."""
        mock_load.return_value = mock_data

        cfg = {
            "dataset": "email_eu_core_full",
            "methods": ["unknown_method", "louvain"],
            "output_dir": str(tmp_path),
        }

        results = run(cfg)

        assert "unknown_method" not in results
        assert "louvain" in results

    @patch("gnn_playground.tasks.community_detection.load_dataset")
    @patch("gnn_playground.tasks.community_detection.draw_communities")
    def test_run_creates_output_dir(self, mock_draw, mock_load, mock_data, tmp_path):
        """Test that output directory is created."""
        mock_load.return_value = mock_data
        output_dir = tmp_path / "nested" / "output"

        cfg = {
            "dataset": "email_eu_core_full",
            "methods": ["louvain"],
            "output_dir": str(output_dir),
        }

        run(cfg)

        assert output_dir.exists()

    @patch("gnn_playground.tasks.community_detection.load_dataset")
    @patch("gnn_playground.tasks.community_detection.draw_communities")
    def test_modularity_in_range(self, mock_draw, mock_load, mock_data, tmp_path):
        """Test that modularity is in valid range."""
        mock_load.return_value = mock_data

        cfg = {
            "dataset": "email_eu_core_full",
            "methods": ["louvain"],
            "output_dir": str(tmp_path),
        }

        results = run(cfg)

        modularity = results["louvain"]["modularity"]
        assert -0.5 <= modularity <= 1.0

    @patch("gnn_playground.tasks.community_detection.load_dataset")
    @patch("gnn_playground.tasks.community_detection.draw_communities")
    def test_nmi_in_range(self, mock_draw, mock_load, mock_data, tmp_path):
        """Test that NMI is in valid range."""
        mock_load.return_value = mock_data

        cfg = {
            "dataset": "email_eu_core_full",
            "methods": ["louvain"],
            "output_dir": str(tmp_path),
        }

        results = run(cfg)

        nmi = results["louvain"]["nmi"]
        assert 0 <= nmi <= 1.0
