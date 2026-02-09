"""Tests for LightGCN and MatrixFactorization models."""

from __future__ import annotations

import pytest
import torch

from gnn_playground.models.lightgcn import LightGCN, LightGCNConv, MatrixFactorization


class TestLightGCNConv:
    """Tests for LightGCNConv layer."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        conv = LightGCNConv()
        x = torch.randn(10, 64)  # 10 nodes, 64 features
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])  # 4 edges

        out = conv(x, edge_index)

        assert out.shape == (10, 64)

    def test_symmetric_normalization(self):
        """Test that convolution applies symmetric normalization."""
        conv = LightGCNConv()

        # Simple graph: 0 <-> 1
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        edge_index = torch.tensor([[0, 1], [1, 0]])

        out = conv(x, edge_index)

        # With symmetric normalization on degree-1 nodes: norm = 1/sqrt(1) * 1/sqrt(1) = 1
        assert out.shape == (2, 2)
        # Output at node 0 should be x[1] normalized, and vice versa
        assert torch.allclose(out[0], x[1], atol=1e-5)
        assert torch.allclose(out[1], x[0], atol=1e-5)


class TestLightGCN:
    """Tests for LightGCN model."""

    @pytest.fixture
    def model(self) -> LightGCN:
        """Create a small LightGCN model."""
        return LightGCN(
            num_users=100,
            num_items=50,
            embedding_dim=32,
            num_layers=2,
        )

    @pytest.fixture
    def bipartite_edge_index(self) -> torch.Tensor:
        """Create a simple bipartite edge index."""
        # Users: 0-99, Items: 100-149 (shifted)
        user_ids = torch.tensor([0, 1, 2, 0, 1])
        item_ids = torch.tensor([100, 101, 102, 101, 100])
        # Bidirectional for undirected graph
        edge_index = torch.stack(
            [
                torch.cat([user_ids, item_ids]),
                torch.cat([item_ids, user_ids]),
            ]
        )
        return edge_index

    def test_init(self, model):
        """Test model initialization."""
        assert model.num_users == 100
        assert model.num_items == 50
        assert model.num_layers == 2
        assert model.user_embedding.num_embeddings == 100
        assert model.item_embedding.num_embeddings == 50

    def test_forward_shape(self, model, bipartite_edge_index):
        """Test forward pass produces correct shapes."""
        user_emb, item_emb = model(bipartite_edge_index)

        assert user_emb.shape == (100, 32)
        assert item_emb.shape == (50, 32)

    def test_predict(self, model, bipartite_edge_index):
        """Test prediction method."""
        user_emb, item_emb = model(bipartite_edge_index)

        # Single prediction
        user_e = user_emb[0].unsqueeze(0)
        item_e = item_emb[0].unsqueeze(0)
        score = model.predict(user_e, item_e)
        assert score.shape == (1,)

        # Batch prediction
        batch_user_e = user_emb[:10]
        batch_item_e = item_emb[:10]
        scores = model.predict(batch_user_e, batch_item_e)
        assert scores.shape == (10,)

    def test_get_embedding(self, model, bipartite_edge_index):
        """Test get_embedding method."""
        user_ids = torch.tensor([0, 1, 2])
        item_ids = torch.tensor([0, 1, 2])

        user_e, item_e = model.get_embedding(bipartite_edge_index, user_ids, item_ids)

        assert user_e.shape == (3, 32)
        assert item_e.shape == (3, 32)

    def test_layer_aggregation(self, model, bipartite_edge_index):
        """Test that embeddings are averaged across layers."""
        # This is an indirect test - verify model produces reasonable outputs
        user_emb, item_emb = model(bipartite_edge_index)

        # Embeddings should be finite
        assert torch.isfinite(user_emb).all()
        assert torch.isfinite(item_emb).all()

        # Embeddings should not all be zero
        assert user_emb.abs().sum() > 0
        assert item_emb.abs().sum() > 0

    def test_gradient_flow(self, model, bipartite_edge_index):
        """Test that gradients flow through the model."""
        user_emb, item_emb = model(bipartite_edge_index)

        # Compute a simple loss
        loss = user_emb.sum() + item_emb.sum()
        loss.backward()

        # Check gradients exist
        assert model.user_embedding.weight.grad is not None
        assert model.item_embedding.weight.grad is not None


class TestMatrixFactorization:
    """Tests for MatrixFactorization baseline."""

    @pytest.fixture
    def model(self) -> MatrixFactorization:
        """Create a small MF model."""
        return MatrixFactorization(
            num_users=100,
            num_items=50,
            embedding_dim=32,
        )

    def test_init(self, model):
        """Test model initialization."""
        assert model.user_embedding.num_embeddings == 100
        assert model.item_embedding.num_embeddings == 50

    def test_forward_ignores_edge_index(self, model):
        """Test that forward ignores edge_index."""
        # With edge_index
        user_emb1, item_emb1 = model(torch.tensor([[0, 1], [1, 0]]))
        # Without edge_index
        user_emb2, item_emb2 = model(None)

        assert torch.equal(user_emb1, user_emb2)
        assert torch.equal(item_emb1, item_emb2)

    def test_forward_shape(self, model):
        """Test forward pass produces correct shapes."""
        user_emb, item_emb = model()

        assert user_emb.shape == (100, 32)
        assert item_emb.shape == (50, 32)

    def test_predict(self, model):
        """Test prediction method."""
        user_emb, item_emb = model()

        # Single prediction
        score = model.predict(user_emb[0:1], item_emb[0:1])
        assert score.shape == (1,)

        # Batch prediction
        scores = model.predict(user_emb[:10], item_emb[:10])
        assert scores.shape == (10,)

    def test_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        user_emb, item_emb = model()
        loss = user_emb.sum() + item_emb.sum()
        loss.backward()

        assert model.user_embedding.weight.grad is not None
        assert model.item_embedding.weight.grad is not None
