"""Tests for GAT model."""

from __future__ import annotations

import pytest
import torch

from gnn_playground.models import MODEL_REGISTRY, get_model
from gnn_playground.models.gat import GAT


class TestGAT:
    def test_registered(self):
        assert "gat" in MODEL_REGISTRY

    def test_forward_shape(self, synthetic_node_graph):
        model = GAT(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=8,
            out_channels=3,
        )
        out = model(synthetic_node_graph.x, synthetic_node_graph.edge_index)
        assert out.shape == (synthetic_node_graph.x.shape[0], 3)

    @pytest.mark.parametrize("heads", [1, 4, 8])
    def test_heads(self, synthetic_node_graph, heads):
        model = GAT(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=8,
            out_channels=3,
            heads=heads,
        )
        out = model(synthetic_node_graph.x, synthetic_node_graph.edge_index)
        assert out.shape == (synthetic_node_graph.x.shape[0], 3)

    @pytest.mark.parametrize("dropout", [0.0, 0.6])
    def test_dropout(self, synthetic_node_graph, dropout):
        model = GAT(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=8,
            out_channels=3,
            dropout=dropout,
        )
        model.train()
        out = model(synthetic_node_graph.x, synthetic_node_graph.edge_index)
        assert out.shape[0] == synthetic_node_graph.x.shape[0]

    def test_eval_mode_deterministic(self, synthetic_node_graph):
        model = GAT(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=8,
            out_channels=3,
            dropout=0.6,
        )
        model.eval()
        out1 = model(synthetic_node_graph.x, synthetic_node_graph.edge_index)
        out2 = model(synthetic_node_graph.x, synthetic_node_graph.edge_index)
        assert torch.allclose(out1, out2)

    def test_get_model_factory(self, synthetic_node_graph):
        model = get_model(
            "gat",
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=8,
            out_channels=3,
        )
        assert isinstance(model, GAT)
