"""Tests for graph-level GCN model."""

from __future__ import annotations

import pytest
import torch

from gnn_playground.models import MODEL_REGISTRY, get_model
from gnn_playground.models.gcn_graph import GraphGCN


class TestGraphGCN:
    def test_registered(self):
        assert "graph_gcn" in MODEL_REGISTRY

    def test_forward_shape(self, synthetic_graph_batch):
        model = GraphGCN(
            in_channels=synthetic_graph_batch.x.shape[1],
            hidden_channels=16,
            out_channels=2,
        )
        out = model(synthetic_graph_batch.x, synthetic_graph_batch.edge_index, synthetic_graph_batch.batch)

        num_graphs = synthetic_graph_batch.batch.max().item() + 1
        assert out.shape == (num_graphs, 2)

    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_num_layers(self, synthetic_graph_batch, num_layers):
        model = GraphGCN(
            in_channels=synthetic_graph_batch.x.shape[1],
            hidden_channels=16,
            out_channels=2,
            num_layers=num_layers,
        )
        out = model(synthetic_graph_batch.x, synthetic_graph_batch.edge_index, synthetic_graph_batch.batch)

        num_graphs = synthetic_graph_batch.batch.max().item() + 1
        assert out.shape == (num_graphs, 2)

    @pytest.mark.parametrize("pooling", ["sum", "mean"])
    def test_pooling(self, synthetic_graph_batch, pooling):
        model = GraphGCN(
            in_channels=synthetic_graph_batch.x.shape[1],
            hidden_channels=16,
            out_channels=2,
            pooling=pooling,
        )
        out = model(synthetic_graph_batch.x, synthetic_graph_batch.edge_index, synthetic_graph_batch.batch)

        num_graphs = synthetic_graph_batch.batch.max().item() + 1
        assert out.shape == (num_graphs, 2)

    def test_single_graph(self, synthetic_node_graph):
        # Test with a single graph (no batch tensor)
        model = GraphGCN(
            in_channels=synthetic_node_graph.x.shape[1],
            hidden_channels=16,
            out_channels=2,
        )
        out = model(synthetic_node_graph.x, synthetic_node_graph.edge_index)
        assert out.shape == (1, 2)

    def test_eval_mode_deterministic(self, synthetic_graph_batch):
        model = GraphGCN(
            in_channels=synthetic_graph_batch.x.shape[1],
            hidden_channels=16,
            out_channels=2,
            dropout=0.5,
        )
        model.eval()
        out1 = model(synthetic_graph_batch.x, synthetic_graph_batch.edge_index, synthetic_graph_batch.batch)
        out2 = model(synthetic_graph_batch.x, synthetic_graph_batch.edge_index, synthetic_graph_batch.batch)
        assert torch.allclose(out1, out2)

    def test_get_model_factory(self, synthetic_graph_batch):
        model = get_model(
            "graph_gcn",
            in_channels=synthetic_graph_batch.x.shape[1],
            hidden_channels=16,
            out_channels=2,
        )
        assert isinstance(model, GraphGCN)
