"""Tests for Elliptic dataset loader."""

from __future__ import annotations

import pytest
import torch

from gnn_playground.datasets.elliptic import EllipticData, compute_class_weights, load_elliptic


class TestEllipticData:
    """Tests for EllipticData container."""

    def test_namedtuple_fields(self):
        """Test that EllipticData has expected fields."""
        from torch_geometric.data import Data

        data = Data(
            x=torch.randn(10, 5), edge_index=torch.zeros(2, 5, dtype=torch.long), y=torch.zeros(10, dtype=torch.long)
        )
        elliptic = EllipticData(
            data=data,
            num_licit=7,
            num_illicit=2,
            num_unknown=1,
            timesteps=torch.zeros(10, dtype=torch.long),
        )
        assert elliptic.num_licit == 7
        assert elliptic.num_illicit == 2
        assert elliptic.num_unknown == 1


class TestComputeClassWeights:
    """Tests for compute_class_weights function."""

    def test_balanced_classes(self):
        """Test weights for balanced classes."""
        y = torch.tensor([0, 0, 0, 1, 1, 1])
        mask = torch.ones(6, dtype=torch.bool)

        weights = compute_class_weights(y, mask)

        # Balanced classes should have equal weights
        assert len(weights) == 2
        assert torch.allclose(weights[0], weights[1], atol=0.01)

    def test_imbalanced_classes(self):
        """Test weights for imbalanced classes."""
        y = torch.tensor([0, 0, 0, 0, 0, 1])  # 5:1 ratio
        mask = torch.ones(6, dtype=torch.bool)

        weights = compute_class_weights(y, mask)

        # Minority class should have higher weight
        assert weights[1] > weights[0]

    def test_respects_mask(self):
        """Test that mask is respected."""
        y = torch.tensor([0, 0, 0, 1, 1, 1])
        mask = torch.tensor([True, True, True, False, False, False])  # Only class 0

        weights = compute_class_weights(y, mask)

        # Only class 0 in mask
        assert weights[0] > 0


class TestLoadElliptic:
    """Tests for load_elliptic function."""

    @pytest.fixture
    def mock_elliptic_files(self, tmp_path):
        """Create mock Elliptic CSV files."""
        elliptic_dir = tmp_path / "elliptic"
        elliptic_dir.mkdir()

        # Create mock features CSV (txId, timestep, 3 features)
        features_content = """1,1,0.1,0.2,0.3
2,1,0.4,0.5,0.6
3,2,0.7,0.8,0.9
4,2,1.0,1.1,1.2
5,3,1.3,1.4,1.5
6,3,1.6,1.7,1.8
7,4,1.9,2.0,2.1
8,4,2.2,2.3,2.4
9,5,2.5,2.6,2.7
10,5,2.8,2.9,3.0"""
        (elliptic_dir / "elliptic_txs_features.csv").write_text(features_content)

        # Create mock edges CSV
        edges_content = """txId1,txId2
1,2
2,3
3,4
4,5
5,6
6,7
7,8
8,9
9,10"""
        (elliptic_dir / "elliptic_txs_edgelist.csv").write_text(edges_content)

        # Create mock classes CSV (1=licit, 2=illicit, unknown)
        classes_content = """txId,class
1,1
2,1
3,2
4,1
5,unknown
6,2
7,1
8,1
9,unknown
10,1"""
        (elliptic_dir / "elliptic_txs_classes.csv").write_text(classes_content)

        return tmp_path

    def test_load_creates_data(self, mock_elliptic_files):
        """Test that load_elliptic creates valid Data object."""
        result = load_elliptic(root=str(mock_elliptic_files))

        assert isinstance(result, EllipticData)
        assert result.data.x.shape[0] == 10  # 10 nodes
        assert result.data.x.shape[1] == 3  # 3 features
        assert result.data.edge_index.shape[0] == 2

    def test_label_mapping(self, mock_elliptic_files):
        """Test that labels are mapped correctly."""
        result = load_elliptic(root=str(mock_elliptic_files))

        # Check label counts
        assert result.num_licit == 6  # txIds 1,2,4,7,8,10
        assert result.num_illicit == 2  # txIds 3,6
        assert result.num_unknown == 2  # txIds 5,9

    def test_unknown_excluded_from_masks(self, mock_elliptic_files):
        """Test that unknown labels are excluded from masks."""
        result = load_elliptic(root=str(mock_elliptic_files))

        # Unknown labels (y=-1) should not be in any mask
        unknown_mask = result.data.y == -1
        assert not (result.data.train_mask & unknown_mask).any()
        assert not (result.data.val_mask & unknown_mask).any()
        assert not (result.data.test_mask & unknown_mask).any()

    def test_temporal_split(self, mock_elliptic_files):
        """Test temporal split assigns correct masks."""
        result = load_elliptic(root=str(mock_elliptic_files), train_ratio=0.6, val_ratio=0.2)

        # Should have nodes in each split (excluding unknown)
        assert result.data.train_mask.sum() > 0
        assert result.data.val_mask.sum() > 0
        assert result.data.test_mask.sum() > 0

        # Masks should not overlap
        overlap = (
            (result.data.train_mask & result.data.val_mask).any()
            or (result.data.train_mask & result.data.test_mask).any()
            or (result.data.val_mask & result.data.test_mask).any()
        )
        assert not overlap

    def test_timesteps_tensor(self, mock_elliptic_files):
        """Test that timesteps tensor is created."""
        result = load_elliptic(root=str(mock_elliptic_files))

        assert result.timesteps.shape[0] == 10
        assert result.timesteps.dtype == torch.int64

    def test_missing_file_raises_error(self, tmp_path):
        """Test that missing files raise FileNotFoundError."""
        elliptic_dir = tmp_path / "elliptic"
        elliptic_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            load_elliptic(root=str(tmp_path))

    def test_edges_filtered_to_valid_nodes(self, mock_elliptic_files):
        """Test that edges only include valid nodes."""
        result = load_elliptic(root=str(mock_elliptic_files))

        # All edge endpoints should be valid node indices
        max_idx = result.data.edge_index.max().item()
        assert max_idx < result.data.x.shape[0]
