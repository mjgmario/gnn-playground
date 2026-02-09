"""Tests for metrics computation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gnn_playground.training.metrics import (
    compute_classification_metrics,
    compute_community_metrics,
    compute_fraud_metrics,
    compute_link_metrics,
    compute_modularity,
    compute_ranking_metrics,
)


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        preds = torch.tensor([0, 1, 2, 0, 1])
        labels = torch.tensor([0, 1, 2, 0, 1])
        result = compute_classification_metrics(preds, labels)
        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0

    def test_all_wrong_predictions(self):
        preds = torch.tensor([1, 2, 0])
        labels = torch.tensor([0, 1, 2])
        result = compute_classification_metrics(preds, labels)
        assert result["accuracy"] == 0.0

    def test_known_mixed_predictions(self):
        preds = torch.tensor([0, 1, 1, 0])
        labels = torch.tensor([0, 1, 0, 0])
        result = compute_classification_metrics(preds, labels)
        assert result["accuracy"] == pytest.approx(0.75)

    def test_numpy_input(self):
        preds = np.array([0, 1, 2])
        labels = np.array([0, 1, 2])
        result = compute_classification_metrics(preds, labels)
        assert result["accuracy"] == 1.0

    @pytest.mark.parametrize("average", ["macro", "micro", "weighted"])
    def test_f1_average_modes(self, average):
        preds = torch.tensor([0, 1, 1, 0, 2])
        labels = torch.tensor([0, 1, 0, 0, 2])
        result = compute_classification_metrics(preds, labels, average=average)
        assert 0.0 <= result["f1"] <= 1.0

    def test_single_class(self):
        preds = torch.tensor([0, 0, 0])
        labels = torch.tensor([0, 0, 0])
        result = compute_classification_metrics(preds, labels)
        assert result["accuracy"] == 1.0


class TestLinkMetrics:
    def test_perfect_scores(self):
        scores = torch.tensor([0.9, 0.8, 0.1, 0.05])
        labels = torch.tensor([1, 1, 0, 0])
        result = compute_link_metrics(scores, labels)
        assert result["auc_roc"] == 1.0
        assert result["avg_precision"] == 1.0

    def test_random_scores_auc_around_half(self):
        np.random.seed(42)
        n = 1000
        scores = torch.tensor(np.random.rand(n))
        labels = torch.tensor(np.random.randint(0, 2, n))
        result = compute_link_metrics(scores, labels)
        assert 0.3 < result["auc_roc"] < 0.7

    def test_numpy_input(self):
        scores = np.array([0.9, 0.1])
        labels = np.array([1, 0])
        result = compute_link_metrics(scores, labels)
        assert result["auc_roc"] == 1.0


class TestRankingMetrics:
    def test_perfect_ranking(self):
        rankings = {0: [1, 2, 3], 1: [4, 5, 6]}
        ground_truth = {0: {1, 2, 3}, 1: {4, 5, 6}}
        result = compute_ranking_metrics(rankings, ground_truth, k_list=[3])
        assert result["recall@3"] == 1.0
        assert result["ndcg@3"] == 1.0

    def test_empty_recommendations(self):
        rankings = {0: []}
        ground_truth = {0: {1, 2, 3}}
        result = compute_ranking_metrics(rankings, ground_truth, k_list=[10])
        assert result["recall@10"] == 0.0
        assert result["ndcg@10"] == 0.0

    @pytest.mark.parametrize("k", [10, 20, 50])
    def test_k_values_in_output(self, k):
        rankings = {0: list(range(100))}
        ground_truth = {0: {5, 15, 25}}
        result = compute_ranking_metrics(rankings, ground_truth, k_list=[k])
        assert f"recall@{k}" in result
        assert f"ndcg@{k}" in result

    def test_default_k_list(self):
        rankings = {0: list(range(100))}
        ground_truth = {0: {5}}
        result = compute_ranking_metrics(rankings, ground_truth)
        assert "recall@10" in result
        assert "recall@20" in result
        assert "recall@50" in result

    def test_no_relevant_users_skipped(self):
        rankings = {0: [1, 2, 3]}
        ground_truth = {}  # no ground truth for user 0
        result = compute_ranking_metrics(rankings, ground_truth, k_list=[3])
        assert result["recall@3"] == 0.0


class TestFraudMetrics:
    def test_perfect_classifier(self):
        scores = np.array([0.99, 0.98, 0.01, 0.02])
        labels = np.array([1, 1, 0, 0])
        result = compute_fraud_metrics(scores, labels)
        assert result["pr_auc"] == pytest.approx(1.0, abs=0.01)

    def test_f1_with_threshold(self):
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        result = compute_fraud_metrics(scores, labels)
        assert result["f1"] == 1.0

    def test_recall_at_precision_key(self):
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        result = compute_fraud_metrics(scores, labels, precision_target=0.9)
        assert "recall@precision=0.9" in result

    def test_torch_input(self):
        scores = torch.tensor([0.9, 0.1])
        labels = torch.tensor([1, 0])
        result = compute_fraud_metrics(scores, labels)
        assert "pr_auc" in result


class TestModularity:
    def test_modularity_nontrivial_partition(self):
        """Test modularity for a simple graph with clear communities."""
        # Two cliques connected by one edge
        # Clique 1: 0-1, 0-2, 1-2
        # Clique 2: 3-4, 3-5, 4-5
        # Bridge: 2-3
        edge_index = torch.tensor(
            [
                [0, 0, 1, 2, 3, 3, 4],
                [1, 2, 2, 3, 4, 5, 5],
            ]
        )
        partition = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
        num_nodes = 6

        modularity = compute_modularity(partition, edge_index, num_nodes)

        # Should be positive for good partition
        assert modularity > 0

    def test_modularity_single_community(self):
        """Test modularity when all nodes in one community."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        partition = {0: 0, 1: 0, 2: 0}
        num_nodes = 3

        modularity = compute_modularity(partition, edge_index, num_nodes)

        # Modularity should be in valid range [-0.5, 1]
        assert -0.5 <= modularity <= 1.0

    def test_modularity_empty_graph(self):
        """Test modularity for graph with no edges."""
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        partition = {0: 0, 1: 1}
        num_nodes = 2

        modularity = compute_modularity(partition, edge_index, num_nodes)

        assert modularity == 0.0

    def test_modularity_numpy_input(self):
        """Test modularity with numpy edge_index."""
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        partition = {0: 0, 1: 0, 2: 0}
        num_nodes = 3

        modularity = compute_modularity(partition, edge_index, num_nodes)

        assert isinstance(modularity, float)


class TestCommunityMetrics:
    def test_nmi_perfect_match(self):
        partition = {0: 0, 1: 0, 2: 1, 3: 1}
        true_labels = [0, 0, 1, 1]
        result = compute_community_metrics(partition, true_labels=true_labels)
        assert result["nmi"] == pytest.approx(1.0)

    def test_nmi_no_ground_truth(self):
        partition = {0: 0, 1: 1}
        result = compute_community_metrics(partition, true_labels=None)
        assert "nmi" not in result

    def test_nmi_numpy_labels(self):
        partition = {0: 0, 1: 0, 2: 1, 3: 1}
        true_labels = np.array([0, 0, 1, 1])
        result = compute_community_metrics(partition, true_labels=true_labels)
        assert result["nmi"] == pytest.approx(1.0)

    def test_modularity_computed_when_edge_index_provided(self):
        """Test that modularity is computed when edge_index is provided."""
        partition = {0: 0, 1: 0, 2: 1, 3: 1}
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        num_nodes = 4

        result = compute_community_metrics(partition, edge_index=edge_index, num_nodes=num_nodes)

        assert "modularity" in result
        assert isinstance(result["modularity"], float)

    def test_modularity_not_computed_without_edge_index(self):
        """Test that modularity is not computed without edge_index."""
        partition = {0: 0, 1: 1}
        result = compute_community_metrics(partition)
        assert "modularity" not in result

    def test_both_metrics_computed(self):
        """Test that both modularity and NMI are computed when both inputs provided."""
        partition = {0: 0, 1: 0, 2: 1, 3: 1}
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        num_nodes = 4
        true_labels = [0, 0, 1, 1]

        result = compute_community_metrics(
            partition, edge_index=edge_index, num_nodes=num_nodes, true_labels=true_labels
        )

        assert "modularity" in result
        assert "nmi" in result
