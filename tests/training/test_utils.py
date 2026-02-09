"""Tests for training utilities."""

from __future__ import annotations

import pytest
import torch

from gnn_playground.training.utils import EarlyStopping, get_device, set_seed


class TestSetSeed:
    def test_deterministic_torch(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(99)
        b = torch.randn(5)
        assert not torch.equal(a, b)

    def test_deterministic_numpy(self):
        import numpy as np

        set_seed(42)
        a = np.random.randn(5)
        set_seed(42)
        b = np.random.randn(5)
        assert (a == b).all()

    def test_deterministic_random(self):
        import random

        set_seed(42)
        a = [random.random() for _ in range(5)]
        set_seed(42)
        b = [random.random() for _ in range(5)]
        assert a == b


class TestGetDevice:
    def test_returns_valid_device(self):
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "cuda", "mps")

    def test_returns_cpu_when_no_gpu(self):
        # On CI/machines without GPU, should at least return cpu
        device = get_device()
        assert device is not None


class TestEarlyStopping:
    def test_does_not_trigger_before_patience(self):
        es = EarlyStopping(patience=5, mode="min")
        for _ in range(4):
            assert es(10.0) is False
        assert not es.should_stop

    def test_triggers_after_patience_no_improvement(self):
        es = EarlyStopping(patience=3, mode="min")
        es(1.0)  # best
        es(2.0)  # worse
        es(2.0)  # worse
        assert es(2.0) is True
        assert es.should_stop

    def test_resets_counter_on_improvement(self):
        es = EarlyStopping(patience=3, mode="min")
        es(1.0)  # best
        es(2.0)  # worse, counter=1
        es(2.0)  # worse, counter=2
        es(0.5)  # better! counter=0
        assert es.counter == 0
        assert not es.should_stop

    def test_min_delta_respected(self):
        es = EarlyStopping(patience=2, min_delta=0.1, mode="min")
        es(1.0)  # best
        es(0.95)  # improvement < min_delta, counts as no improvement
        assert es.counter == 1
        es(0.85)  # improvement > min_delta from best (1.0), resets
        assert es.counter == 0

    def test_max_mode(self):
        es = EarlyStopping(patience=3, mode="max")
        es(0.5)  # best
        es(0.4)  # worse
        es(0.3)  # worse
        assert es(0.2) is True

    def test_max_mode_improvement(self):
        es = EarlyStopping(patience=3, mode="max")
        es(0.5)
        es(0.6)  # improvement
        assert es.counter == 0
        assert es.best_score == 0.6

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            EarlyStopping(mode="invalid")

    def test_first_call_never_stops(self):
        es = EarlyStopping(patience=1, mode="min")
        assert es(100.0) is False

    @pytest.mark.parametrize("patience", [1, 5, 10])
    def test_various_patience_values(self, patience):
        es = EarlyStopping(patience=patience, mode="min")
        es(1.0)
        for _ in range(patience - 1):
            assert es(2.0) is False
        assert es(2.0) is True
