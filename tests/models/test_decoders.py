"""Tests for link prediction decoders."""

from __future__ import annotations

import pytest
import torch

from gnn_playground.models.decoders import (
    BilinearDecoder,
    DotProductDecoder,
    MLPDecoder,
    get_decoder,
)


class TestDotProductDecoder:
    def test_output_shape(self):
        decoder = DotProductDecoder()
        z_src = torch.randn(100, 64)
        z_dst = torch.randn(100, 64)
        scores = decoder(z_src, z_dst)
        assert scores.shape == (100,)

    def test_symmetric(self):
        decoder = DotProductDecoder()
        z_src = torch.randn(50, 32)
        z_dst = torch.randn(50, 32)
        scores_ab = decoder(z_src, z_dst)
        scores_ba = decoder(z_dst, z_src)
        assert torch.allclose(scores_ab, scores_ba)

    def test_gradients_flow(self):
        decoder = DotProductDecoder()
        z_src = torch.randn(10, 16, requires_grad=True)
        z_dst = torch.randn(10, 16, requires_grad=True)
        scores = decoder(z_src, z_dst)
        loss = scores.sum()
        loss.backward()
        assert z_src.grad is not None
        assert z_dst.grad is not None


class TestMLPDecoder:
    def test_output_shape(self):
        decoder = MLPDecoder(in_channels=64)
        z_src = torch.randn(100, 64)
        z_dst = torch.randn(100, 64)
        scores = decoder(z_src, z_dst)
        assert scores.shape == (100,)

    def test_gradients_flow(self):
        decoder = MLPDecoder(in_channels=32)
        z_src = torch.randn(10, 32, requires_grad=True)
        z_dst = torch.randn(10, 32, requires_grad=True)
        scores = decoder(z_src, z_dst)
        loss = scores.sum()
        loss.backward()
        assert z_src.grad is not None

    def test_eval_mode_deterministic(self):
        decoder = MLPDecoder(in_channels=32, dropout=0.5)
        decoder.eval()
        z_src = torch.randn(10, 32)
        z_dst = torch.randn(10, 32)
        scores1 = decoder(z_src, z_dst)
        scores2 = decoder(z_src, z_dst)
        assert torch.allclose(scores1, scores2)


class TestBilinearDecoder:
    def test_output_shape(self):
        decoder = BilinearDecoder(in_channels=64)
        z_src = torch.randn(100, 64)
        z_dst = torch.randn(100, 64)
        scores = decoder(z_src, z_dst)
        assert scores.shape == (100,)

    def test_gradients_flow(self):
        decoder = BilinearDecoder(in_channels=32)
        z_src = torch.randn(10, 32, requires_grad=True)
        z_dst = torch.randn(10, 32, requires_grad=True)
        scores = decoder(z_src, z_dst)
        loss = scores.sum()
        loss.backward()
        assert z_src.grad is not None


class TestGetDecoder:
    @pytest.mark.parametrize("name", ["dot", "mlp", "bilinear"])
    def test_get_decoder(self, name):
        kwargs = {"in_channels": 32} if name in ["mlp", "bilinear"] else {}
        decoder = get_decoder(name, **kwargs)
        assert decoder is not None

    def test_unknown_decoder_raises(self):
        with pytest.raises(KeyError, match="Unknown decoder"):
            get_decoder("unknown")
