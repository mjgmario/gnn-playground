"""Tests for MLP baseline model."""

from __future__ import annotations

import pytest
import torch

from gnn_playground.models import MODEL_REGISTRY, get_model
from gnn_playground.models.mlp import MLP


class TestMLP:
    def test_registered(self):
        assert "mlp" in MODEL_REGISTRY

    def test_forward_shape(self, synthetic_node_graph):
        model = MLP(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=3,
        )
        out = model(synthetic_node_graph.x, synthetic_node_graph.edge_index)
        assert out.shape == (synthetic_node_graph.x.shape[0], 3)

    def test_ignores_edge_index(self, synthetic_node_graph):
        model = MLP(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=3,
        )
        model.eval()  # Disable dropout for deterministic comparison
        # Forward with edge_index
        out_with_edges = model(synthetic_node_graph.x, synthetic_node_graph.edge_index)
        # Forward without edge_index
        out_without_edges = model(synthetic_node_graph.x, None)
        # Should be same (edge_index is ignored)
        assert torch.allclose(out_with_edges, out_without_edges)

    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_num_layers(self, synthetic_node_graph, num_layers):
        model = MLP(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=3,
            num_layers=num_layers,
        )
        out = model(synthetic_node_graph.x, synthetic_node_graph.edge_index)
        assert out.shape == (synthetic_node_graph.x.shape[0], 3)

    @pytest.mark.parametrize("dropout", [0.0, 0.5])
    def test_dropout(self, synthetic_node_graph, dropout):
        model = MLP(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=3,
            dropout=dropout,
        )
        model.train()
        out = model(synthetic_node_graph.x, synthetic_node_graph.edge_index)
        assert out.shape[0] == synthetic_node_graph.x.shape[0]

    def test_eval_mode_deterministic(self, synthetic_node_graph):
        model = MLP(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=3,
            dropout=0.5,
        )
        model.eval()
        out1 = model(synthetic_node_graph.x, synthetic_node_graph.edge_index)
        out2 = model(synthetic_node_graph.x, synthetic_node_graph.edge_index)
        assert torch.allclose(out1, out2)

    def test_get_model_factory(self, synthetic_node_graph):
        model = get_model(
            "mlp",
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=3,
        )
        assert isinstance(model, MLP)
