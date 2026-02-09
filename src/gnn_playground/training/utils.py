"""Training utilities: seed, device, early stopping."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.

    :param seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Auto-detect the best available device.

    :return: torch.device for cuda, mps, or cpu.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EarlyStopping:
    """Early stopping to terminate training when validation metric stops improving.

    :param patience: Number of epochs to wait after last improvement.
    :param min_delta: Minimum change to qualify as an improvement.
    :param mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best - self.min_delta
        return current > best + self.min_delta

    def __call__(self, metric: float) -> bool:
        """Check if training should stop.

        :param metric: Current epoch's validation metric.
        :return: True if training should stop.
        """
        if self.best_score is None:
            self.best_score = metric
            return False

        if self._is_improvement(metric, self.best_score):
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False
