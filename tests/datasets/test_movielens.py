"""Tests for MovieLens dataset loader."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from gnn_playground.datasets.movielens import RecsysData, load_movielens


class TestRecsysData:
    """Tests for RecsysData container."""

    def test_namedtuple_fields(self):
        """Test that RecsysData has expected fields."""
        data = RecsysData(
            num_users=10,
            num_items=20,
            train_edges=torch.zeros(2, 5, dtype=torch.long),
            val_edges=torch.zeros(2, 2, dtype=torch.long),
            test_edges=torch.zeros(2, 2, dtype=torch.long),
            val_ground_truth={0: {1}},
            test_ground_truth={0: {2}},
        )
        assert data.num_users == 10
        assert data.num_items == 20
        assert data.train_edges.shape == (2, 5)
        assert data.val_edges.shape == (2, 2)
        assert data.test_edges.shape == (2, 2)
        assert data.val_ground_truth == {0: {1}}
        assert data.test_ground_truth == {0: {2}}


class TestLoadMovielens:
    """Tests for load_movielens function."""

    @pytest.fixture
    def mock_ratings_data(self) -> str:
        """Sample ratings data in MovieLens format."""
        # Format: user_id \t item_id \t rating \t timestamp
        # User 1 has 5 interactions (items 1,2,3,4,5) - will have train/val/test
        # User 2 has 2 interactions (items 1,3) - will have train/test only
        # User 3 has 1 interaction (item 2) - will have train only
        lines = [
            "1\t1\t5\t100",  # User 1, Item 1, rating 5
            "1\t2\t4\t200",  # User 1, Item 2, rating 4
            "1\t3\t5\t300",  # User 1, Item 3, rating 5
            "1\t4\t4\t400",  # User 1, Item 4 (val)
            "1\t5\t5\t500",  # User 1, Item 5 (test)
            "2\t1\t4\t150",  # User 2, Item 1
            "2\t3\t5\t250",  # User 2, Item 3 (test)
            "3\t2\t4\t180",  # User 3, Item 2
        ]
        return "\n".join(lines)

    @patch("gnn_playground.datasets.movielens.download_movielens")
    def test_load_creates_correct_splits(self, mock_download, mock_ratings_data, tmp_path):
        """Test that load_movielens creates correct train/val/test splits."""
        mock_download.return_value = tmp_path

        # Create mock u.data file
        (tmp_path / "u.data").write_text(mock_ratings_data)

        data = load_movielens(root=str(tmp_path), rating_threshold=4)

        assert isinstance(data, RecsysData)
        assert data.num_users > 0
        assert data.num_items > 0
        assert data.train_edges.shape[0] == 2
        assert data.train_edges.shape[1] > 0

    @patch("gnn_playground.datasets.movielens.download_movielens")
    def test_leave_last_out_split(self, mock_download, mock_ratings_data, tmp_path):
        """Test leave-last-out split strategy."""
        mock_download.return_value = tmp_path
        (tmp_path / "u.data").write_text(mock_ratings_data)

        data = load_movielens(root=str(tmp_path), rating_threshold=4)

        # User 0 (1 in original) has 5 interactions -> 3 train, 1 val, 1 test
        # User 1 (2 in original) has 2 interactions -> 1 train, 1 test
        # User 2 (3 in original) has 1 interaction -> 1 train only

        # Total: 5 train, 1 val, 2 test (based on rules)
        assert data.train_edges.shape[1] == 5  # 3 + 1 + 1
        assert data.val_edges.shape[1] == 1  # Only user 0 has val
        assert data.test_edges.shape[1] == 2  # Users 0 and 1 have test

    @patch("gnn_playground.datasets.movielens.download_movielens")
    def test_ground_truth_dicts(self, mock_download, mock_ratings_data, tmp_path):
        """Test ground truth dictionaries."""
        mock_download.return_value = tmp_path
        (tmp_path / "u.data").write_text(mock_ratings_data)

        data = load_movielens(root=str(tmp_path), rating_threshold=4)

        # User 0 should have val and test ground truth
        assert 0 in data.val_ground_truth
        assert 0 in data.test_ground_truth
        assert isinstance(data.val_ground_truth[0], set)
        assert isinstance(data.test_ground_truth[0], set)

    @patch("gnn_playground.datasets.movielens.download_movielens")
    def test_rating_threshold_filtering(self, mock_download, tmp_path):
        """Test that rating threshold filters interactions."""
        mock_download.return_value = tmp_path

        # Ratings below and at threshold
        lines = [
            "1\t1\t3\t100",  # Below threshold (4)
            "1\t2\t4\t200",  # At threshold
            "1\t3\t5\t300",  # Above threshold
        ]
        (tmp_path / "u.data").write_text("\n".join(lines))

        data = load_movielens(root=str(tmp_path), rating_threshold=4)

        # Only 2 interactions should pass (ratings 4 and 5)
        total_edges = data.train_edges.shape[1] + data.val_edges.shape[1] + data.test_edges.shape[1]
        assert total_edges == 2

    @patch("gnn_playground.datasets.movielens.download_movielens")
    def test_empty_splits_handled(self, mock_download, tmp_path):
        """Test handling of users with few interactions."""
        mock_download.return_value = tmp_path

        # Single interaction only
        lines = ["1\t1\t5\t100"]
        (tmp_path / "u.data").write_text("\n".join(lines))

        data = load_movielens(root=str(tmp_path), rating_threshold=4)

        # Should handle gracefully with train only
        assert data.train_edges.shape[1] == 1
        assert data.val_edges.shape[1] == 0
        assert data.test_edges.shape[1] == 0

    @patch("gnn_playground.datasets.movielens.download_movielens")
    def test_tensor_dtypes(self, mock_download, mock_ratings_data, tmp_path):
        """Test that edge tensors have correct dtype."""
        mock_download.return_value = tmp_path
        (tmp_path / "u.data").write_text(mock_ratings_data)

        data = load_movielens(root=str(tmp_path), rating_threshold=4)

        assert data.train_edges.dtype == torch.long
        assert data.val_edges.dtype == torch.long
        assert data.test_edges.dtype == torch.long
