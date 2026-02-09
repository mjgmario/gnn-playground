"""Model registry."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from gnn_playground.models.gat import GAT
from gnn_playground.models.gcn import GCN
from gnn_playground.models.gcn_graph import GraphGCN
from gnn_playground.models.gin import GIN
from gnn_playground.models.graphsage import GraphSAGE
from gnn_playground.models.mlp import MLP

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    # Node-level models
    "gcn": GCN,
    "graphsage": GraphSAGE,
    "gat": GAT,
    "mlp": MLP,
    # Graph-level models
    "gin": GIN,
    "graph_gcn": GraphGCN,
}


def get_model(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a model by name from the registry.

    :param name: Model name (e.g. 'gcn', 'gat').
    :raises KeyError: If model is not registered.
    :return: Instantiated model.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name](**kwargs)
