"""Tests for training loops."""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from gnn_playground.models.decoders import DotProductDecoder
from gnn_playground.models.gcn import GCN
from gnn_playground.models.gin import GIN
from gnn_playground.models.lightgcn import LightGCN, MatrixFactorization
from gnn_playground.training.loops import (
    bpr_loss,
    eval_graph_epoch,
    eval_link_epoch,
    eval_node_epoch,
    eval_recsys,
    train_graph_epoch,
    train_link_epoch,
    train_node_epoch,
    train_recsys_epoch,
)


class TestTrainNodeEpoch:
    def test_returns_float_loss(self, synthetic_node_graph):
        model = GCN(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=synthetic_node_graph.y.max().item() + 1,
        )
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        loss = train_node_epoch(model, synthetic_node_graph, optimizer, criterion, synthetic_node_graph.train_mask)

        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_decreases(self, synthetic_node_graph):
        model = GCN(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=synthetic_node_graph.y.max().item() + 1,
        )
        optimizer = Adam(model.parameters(), lr=0.1)  # Higher LR for faster convergence
        criterion = nn.CrossEntropyLoss()

        losses = []
        for _ in range(20):
            loss = train_node_epoch(model, synthetic_node_graph, optimizer, criterion, synthetic_node_graph.train_mask)
            losses.append(loss)

        # Loss should decrease (on average) over epochs
        first_half_avg = sum(losses[:10]) / 10
        second_half_avg = sum(losses[10:]) / 10
        assert second_half_avg < first_half_avg

    def test_model_parameters_update(self, synthetic_node_graph):
        model = GCN(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=synthetic_node_graph.y.max().item() + 1,
        )
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        train_node_epoch(model, synthetic_node_graph, optimizer, criterion, synthetic_node_graph.train_mask)

        # Check parameters changed
        params_changed = False
        for p_init, p_new in zip(initial_params, list(model.parameters()), strict=True):
            if not torch.allclose(p_init, p_new):
                params_changed = True
                break
        assert params_changed


class TestEvalNodeEpoch:
    def test_returns_preds_and_labels(self, synthetic_node_graph):
        model = GCN(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=synthetic_node_graph.y.max().item() + 1,
        )

        preds, labels = eval_node_epoch(model, synthetic_node_graph, synthetic_node_graph.val_mask)

        assert isinstance(preds, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert preds.shape == labels.shape

    def test_correct_number_of_predictions(self, synthetic_node_graph):
        model = GCN(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=synthetic_node_graph.y.max().item() + 1,
        )

        val_count = synthetic_node_graph.val_mask.sum().item()
        preds, labels = eval_node_epoch(model, synthetic_node_graph, synthetic_node_graph.val_mask)

        assert preds.shape[0] == val_count
        assert labels.shape[0] == val_count

    def test_no_gradients_computed(self, synthetic_node_graph):
        model = GCN(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=synthetic_node_graph.y.max().item() + 1,
        )

        preds, labels = eval_node_epoch(model, synthetic_node_graph, synthetic_node_graph.val_mask)

        # Results should not require gradients
        assert not preds.requires_grad
        assert not labels.requires_grad

    def test_different_masks(self, synthetic_node_graph):
        model = GCN(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=synthetic_node_graph.y.max().item() + 1,
        )

        train_preds, _ = eval_node_epoch(model, synthetic_node_graph, synthetic_node_graph.train_mask)
        val_preds, _ = eval_node_epoch(model, synthetic_node_graph, synthetic_node_graph.val_mask)
        test_preds, _ = eval_node_epoch(model, synthetic_node_graph, synthetic_node_graph.test_mask)

        # Different masks should give different sized outputs
        train_count = synthetic_node_graph.train_mask.sum().item()
        val_count = synthetic_node_graph.val_mask.sum().item()
        test_count = synthetic_node_graph.test_mask.sum().item()

        assert train_preds.shape[0] == train_count
        assert val_preds.shape[0] == val_count
        assert test_preds.shape[0] == test_count


# ============================================================================
# Graph-level training loop tests
# ============================================================================


class TestTrainGraphEpoch:
    def test_returns_float_loss(self, synthetic_graph_batch):
        # Create a DataLoader from the batch
        graphs = [synthetic_graph_batch[i] for i in range(len(synthetic_graph_batch.y))]
        loader = DataLoader(graphs, batch_size=4)

        model = GIN(
            in_channels=synthetic_graph_batch.x.shape[1],
            hidden_channels=16,
            out_channels=2,
        )
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        loss = train_graph_epoch(model, loader, optimizer, criterion, torch.device("cpu"))

        assert isinstance(loss, float)
        assert loss > 0

    def test_model_parameters_update(self, synthetic_graph_batch):
        graphs = [synthetic_graph_batch[i] for i in range(len(synthetic_graph_batch.y))]
        loader = DataLoader(graphs, batch_size=4)

        model = GIN(
            in_channels=synthetic_graph_batch.x.shape[1],
            hidden_channels=16,
            out_channels=2,
        )
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        train_graph_epoch(model, loader, optimizer, criterion, torch.device("cpu"))

        # Check parameters changed
        params_changed = False
        for p_init, p_new in zip(initial_params, list(model.parameters()), strict=True):
            if not torch.allclose(p_init, p_new):
                params_changed = True
                break
        assert params_changed


class TestEvalGraphEpoch:
    def test_returns_preds_and_labels(self, synthetic_graph_batch):
        graphs = [synthetic_graph_batch[i] for i in range(len(synthetic_graph_batch.y))]
        loader = DataLoader(graphs, batch_size=4)

        model = GIN(
            in_channels=synthetic_graph_batch.x.shape[1],
            hidden_channels=16,
            out_channels=2,
        )

        preds, labels = eval_graph_epoch(model, loader, torch.device("cpu"))

        assert isinstance(preds, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert preds.shape == labels.shape

    def test_correct_number_of_predictions(self, synthetic_graph_batch):
        graphs = [synthetic_graph_batch[i] for i in range(len(synthetic_graph_batch.y))]
        loader = DataLoader(graphs, batch_size=4)

        model = GIN(
            in_channels=synthetic_graph_batch.x.shape[1],
            hidden_channels=16,
            out_channels=2,
        )

        num_graphs = len(synthetic_graph_batch.y)
        preds, labels = eval_graph_epoch(model, loader, torch.device("cpu"))

        assert preds.shape[0] == num_graphs
        assert labels.shape[0] == num_graphs

    def test_no_gradients_computed(self, synthetic_graph_batch):
        graphs = [synthetic_graph_batch[i] for i in range(len(synthetic_graph_batch.y))]
        loader = DataLoader(graphs, batch_size=4)

        model = GIN(
            in_channels=synthetic_graph_batch.x.shape[1],
            hidden_channels=16,
            out_channels=2,
        )

        preds, labels = eval_graph_epoch(model, loader, torch.device("cpu"))

        # Results should not require gradients
        assert not preds.requires_grad
        assert not labels.requires_grad


# ============================================================================
# Link prediction training loop tests
# ============================================================================


class TestTrainLinkEpoch:
    def test_returns_float_loss(self, synthetic_link_data):
        model = GCN(
            in_channels=synthetic_link_data.x.shape[1],
            hidden_channels=16,
            out_channels=16,
        )
        decoder = DotProductDecoder()
        optimizer = Adam(list(model.parameters()) + list(decoder.parameters()), lr=0.01)

        loss = train_link_epoch(model, decoder, synthetic_link_data, optimizer, torch.device("cpu"))

        assert isinstance(loss, float)
        assert loss > 0

    def test_model_parameters_update(self, synthetic_link_data):
        model = GCN(
            in_channels=synthetic_link_data.x.shape[1],
            hidden_channels=16,
            out_channels=16,
        )
        decoder = DotProductDecoder()
        optimizer = Adam(list(model.parameters()) + list(decoder.parameters()), lr=0.01)

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        train_link_epoch(model, decoder, synthetic_link_data, optimizer, torch.device("cpu"))

        # Check parameters changed
        params_changed = False
        for p_init, p_new in zip(initial_params, list(model.parameters()), strict=True):
            if not torch.allclose(p_init, p_new):
                params_changed = True
                break
        assert params_changed


class TestEvalLinkEpoch:
    def test_returns_scores_and_labels(self, synthetic_link_data):
        model = GCN(
            in_channels=synthetic_link_data.x.shape[1],
            hidden_channels=16,
            out_channels=16,
        )
        decoder = DotProductDecoder()

        scores, labels = eval_link_epoch(model, decoder, synthetic_link_data, torch.device("cpu"))

        assert isinstance(scores, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert scores.shape == labels.shape

    def test_correct_number_of_edges(self, synthetic_link_data):
        model = GCN(
            in_channels=synthetic_link_data.x.shape[1],
            hidden_channels=16,
            out_channels=16,
        )
        decoder = DotProductDecoder()

        num_edges = synthetic_link_data.edge_label_index.shape[1]
        scores, labels = eval_link_epoch(model, decoder, synthetic_link_data, torch.device("cpu"))

        assert scores.shape[0] == num_edges
        assert labels.shape[0] == num_edges

    def test_no_gradients_computed(self, synthetic_link_data):
        model = GCN(
            in_channels=synthetic_link_data.x.shape[1],
            hidden_channels=16,
            out_channels=16,
        )
        decoder = DotProductDecoder()

        scores, labels = eval_link_epoch(model, decoder, synthetic_link_data, torch.device("cpu"))

        # Results should not require gradients
        assert not scores.requires_grad
        assert not labels.requires_grad


# ============================================================================
# Recommendation (BPR) training loop tests
# ============================================================================


class TestBPRLoss:
    def test_returns_tensor(self):
        pos_scores = torch.tensor([1.0, 2.0, 3.0])
        neg_scores = torch.tensor([0.5, 1.0, 1.5])

        loss = bpr_loss(pos_scores, neg_scores)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_positive_loss(self):
        pos_scores = torch.tensor([1.0, 2.0, 3.0])
        neg_scores = torch.tensor([0.5, 1.0, 1.5])

        loss = bpr_loss(pos_scores, neg_scores)

        assert loss.item() > 0

    def test_lower_loss_when_pos_greater_than_neg(self):
        # When pos >> neg, loss should be low
        pos_high = torch.tensor([5.0, 5.0, 5.0])
        neg_low = torch.tensor([0.0, 0.0, 0.0])
        loss_good = bpr_loss(pos_high, neg_low)

        # When pos << neg, loss should be high
        pos_low = torch.tensor([0.0, 0.0, 0.0])
        neg_high = torch.tensor([5.0, 5.0, 5.0])
        loss_bad = bpr_loss(pos_low, neg_high)

        assert loss_good < loss_bad

    def test_gradient_flow(self):
        pos_scores = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        neg_scores = torch.tensor([0.5, 1.0, 1.5], requires_grad=True)

        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()

        assert pos_scores.grad is not None
        assert neg_scores.grad is not None


class TestTrainRecsysEpoch:
    def test_returns_float_loss(self):
        num_users = 20
        num_items = 15
        model = MatrixFactorization(num_users, num_items, embedding_dim=16)
        optimizer = Adam(model.parameters(), lr=0.01)

        # Create simple train edges
        train_edges = torch.stack(
            [
                torch.randint(0, num_users, (50,)),
                torch.randint(0, num_items, (50,)),
            ]
        )

        loss = train_recsys_epoch(
            model, train_edges, optimizer, num_items, num_users, batch_size=16, device=torch.device("cpu")
        )

        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_decreases(self):
        num_users = 20
        num_items = 15
        model = MatrixFactorization(num_users, num_items, embedding_dim=16)
        optimizer = Adam(model.parameters(), lr=0.1)

        train_edges = torch.stack(
            [
                torch.randint(0, num_users, (50,)),
                torch.randint(0, num_items, (50,)),
            ]
        )

        losses = []
        for _ in range(10):
            loss = train_recsys_epoch(
                model, train_edges, optimizer, num_items, num_users, batch_size=16, device=torch.device("cpu")
            )
            losses.append(loss)

        # Loss should generally decrease
        first_half_avg = sum(losses[:5]) / 5
        second_half_avg = sum(losses[5:]) / 5
        assert second_half_avg < first_half_avg

    def test_model_parameters_update(self):
        num_users = 20
        num_items = 15
        model = MatrixFactorization(num_users, num_items, embedding_dim=16)
        optimizer = Adam(model.parameters(), lr=0.01)

        train_edges = torch.stack(
            [
                torch.randint(0, num_users, (50,)),
                torch.randint(0, num_items, (50,)),
            ]
        )

        initial_params = [p.clone() for p in model.parameters()]

        train_recsys_epoch(
            model, train_edges, optimizer, num_items, num_users, batch_size=16, device=torch.device("cpu")
        )

        params_changed = False
        for p_init, p_new in zip(initial_params, list(model.parameters()), strict=True):
            if not torch.allclose(p_init, p_new):
                params_changed = True
                break
        assert params_changed

    def test_works_with_lightgcn(self):
        num_users = 20
        num_items = 15
        model = LightGCN(num_users, num_items, embedding_dim=16, num_layers=2)
        optimizer = Adam(model.parameters(), lr=0.01)

        train_edges = torch.stack(
            [
                torch.randint(0, num_users, (50,)),
                torch.randint(0, num_items, (50,)),
            ]
        )

        loss = train_recsys_epoch(
            model, train_edges, optimizer, num_items, num_users, batch_size=16, device=torch.device("cpu")
        )

        assert isinstance(loss, float)
        assert loss > 0


class TestEvalRecsys:
    def test_returns_metrics_dict(self):
        num_users = 20
        num_items = 15
        model = MatrixFactorization(num_users, num_items, embedding_dim=16)

        train_edges = torch.stack(
            [
                torch.randint(0, num_users, (50,)),
                torch.randint(0, num_items, (50,)),
            ]
        )

        ground_truth = {i: {i % num_items} for i in range(10)}

        metrics = eval_recsys(
            model, train_edges, ground_truth, num_users, num_items, k_list=[5, 10], device=torch.device("cpu")
        )

        assert isinstance(metrics, dict)
        assert "recall@5" in metrics
        assert "recall@10" in metrics
        assert "ndcg@5" in metrics
        assert "ndcg@10" in metrics

    def test_metrics_are_floats(self):
        num_users = 20
        num_items = 15
        model = MatrixFactorization(num_users, num_items, embedding_dim=16)

        train_edges = torch.stack(
            [
                torch.randint(0, num_users, (50,)),
                torch.randint(0, num_items, (50,)),
            ]
        )

        ground_truth = {i: {i % num_items} for i in range(10)}

        metrics = eval_recsys(
            model, train_edges, ground_truth, num_users, num_items, k_list=[5], device=torch.device("cpu")
        )

        for k, v in metrics.items():
            assert isinstance(v, float), f"{k} should be float"

    def test_metrics_in_range(self):
        num_users = 20
        num_items = 15
        model = MatrixFactorization(num_users, num_items, embedding_dim=16)

        train_edges = torch.stack(
            [
                torch.randint(0, num_users, (50,)),
                torch.randint(0, num_items, (50,)),
            ]
        )

        ground_truth = {i: {i % num_items} for i in range(10)}

        metrics = eval_recsys(
            model, train_edges, ground_truth, num_users, num_items, k_list=[5, 10], device=torch.device("cpu")
        )

        for k, v in metrics.items():
            assert 0 <= v <= 1, f"{k}={v} should be in [0, 1]"

    def test_works_with_lightgcn(self):
        num_users = 20
        num_items = 15
        model = LightGCN(num_users, num_items, embedding_dim=16, num_layers=2)

        train_edges = torch.stack(
            [
                torch.randint(0, num_users, (50,)),
                torch.randint(0, num_items, (50,)),
            ]
        )

        ground_truth = {i: {i % num_items} for i in range(10)}

        metrics = eval_recsys(
            model, train_edges, ground_truth, num_users, num_items, k_list=[5], device=torch.device("cpu")
        )

        assert isinstance(metrics, dict)
        assert "recall@5" in metrics
