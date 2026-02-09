"""Task registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gnn_playground.tasks import (
    community_detection,
    fraud,
    graph_classification,
    link_prediction,
    node_classification,
    recsys,
)

TASK_REGISTRY: dict[str, Callable[..., Any]] = {
    "node_classification": node_classification.run,
    "graph_classification": graph_classification.run,
    "link_prediction": link_prediction.run,
    "recsys": recsys.run,
    "community_detection": community_detection.run,
    "fraud": fraud.run,
}


def run_task(name: str, cfg: dict[str, Any]) -> Any:
    """Run a task by name.

    :param name: Task name (e.g. 'node_classification').
    :param cfg: Configuration dictionary.
    :raises KeyError: If task is not registered.
    :return: Task results.
    """
    if name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise KeyError(f"Unknown task '{name}'. Available: {available}")
    return TASK_REGISTRY[name](cfg)
