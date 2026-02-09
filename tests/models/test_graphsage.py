"""Tests for GraphSAGE model."""

from __future__ import annotations

import pytest
import torch

from gnn_playground.models import MODEL_REGISTRY, get_model
from gnn_playground.models.graphsage import GraphSAGE


class TestGraphSAGE:
    def test_registered(self):
        assert "graphsage" in MODEL_REGISTRY

    def test_forward_shape(self, synthetic_node_graph):
        model = GraphSAGE(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=3,
        )
        out = model(synthetic_node_graph.x, synthetic_node_graph.edge_index)
        assert out.shape == (synthetic_node_graph.x.shape[0], 3)

    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_num_layers(self, synthetic_node_graph, num_layers):
        model = GraphSAGE(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=3,
            num_layers=num_layers,
        )
        out = model(synthetic_node_graph.x, synthetic_node_graph.edge_index)
        assert out.shape == (synthetic_node_graph.x.shape[0], 3)

    @pytest.mark.parametrize("dropout", [0.0, 0.5])
    def test_dropout(self, synthetic_node_graph, dropout):
        model = GraphSAGE(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=3,
            dropout=dropout,
        )
        model.train()
        out_train = model(synthetic_node_graph.x, synthetic_node_graph.edge_index)
        assert out_train.shape[0] == synthetic_node_graph.x.shape[0]

    def test_eval_mode_deterministic(self, synthetic_node_graph):
        model = GraphSAGE(
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
            "graphsage",
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=3,
        )
        assert isinstance(model, GraphSAGE)
