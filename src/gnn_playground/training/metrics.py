"""Metrics computation for all task types."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_curve,
    roc_auc_score,
)

DEFAULT_K_LIST = [10, 20, 50]


def compute_classification_metrics(
    preds: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    average: str = "macro",
) -> dict[str, float]:
    """Compute classification metrics (accuracy, F1).

    :param preds: Predicted class indices.
    :param labels: Ground truth class indices.
    :param average: Averaging strategy for F1 ('macro', 'micro', 'weighted').
    :return: Dict with 'accuracy' and 'f1'.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average=average, zero_division=0)),
    }


def compute_link_metrics(
    scores: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    """Compute link prediction metrics (AUC-ROC, Average Precision).

    :param scores: Predicted scores (higher = more likely positive).
    :param labels: Binary ground truth (0 or 1).
    :return: Dict with 'auc_roc' and 'avg_precision'.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    return {
        "auc_roc": float(roc_auc_score(labels, scores)),
        "avg_precision": float(average_precision_score(labels, scores)),
    }


def compute_ranking_metrics(
    rankings: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k_list: list[int] | None = None,
) -> dict[str, float]:
    """Compute ranking metrics (Recall@K, NDCG@K).

    :param rankings: Dict mapping user_id -> ordered list of recommended item_ids.
    :param ground_truth: Dict mapping user_id -> set of relevant item_ids.
    :param k_list: List of K values to evaluate.
    :return: Dict with 'recall@K' and 'ndcg@K' for each K.
    """
    if k_list is None:
        k_list = DEFAULT_K_LIST

    results: dict[str, float] = {}

    for k in k_list:
        recalls = []
        ndcgs = []

        for user_id, ranked_items in rankings.items():
            relevant = ground_truth.get(user_id, set())
            if not relevant:
                continue

            top_k = ranked_items[:k]

            # Recall@K
            hits = len(set(top_k) & relevant)
            recalls.append(hits / len(relevant))

            # NDCG@K
            dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(top_k) if item in relevant)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        results[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        results[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0

    return results


def compute_fraud_metrics(
    scores: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    precision_target: float = 0.9,
) -> dict[str, float]:
    """Compute fraud detection metrics (PR-AUC, F1, Recall@Precision).

    :param scores: Predicted scores for the positive (fraud) class.
    :param labels: Binary ground truth (0=licit, 1=fraud).
    :param precision_target: Precision target for Recall@Precision metric.
    :return: Dict with 'pr_auc', 'f1', and 'recall_at_precision'.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    pr_auc = float(average_precision_score(labels, scores))

    # Find recall at target precision
    precision, recall, _ = precision_recall_curve(labels, scores)
    recall_at_target = 0.0
    for p, r in zip(precision, recall, strict=False):
        if p >= precision_target:
            recall_at_target = max(recall_at_target, r)

    # F1 with default threshold 0.5
    preds_binary = (scores >= 0.5).astype(int)
    f1 = float(f1_score(labels, preds_binary, zero_division=0))

    return {
        "pr_auc": pr_auc,
        "f1": f1,
        f"recall@precision={precision_target}": recall_at_target,
    }


def compute_modularity(
    partition: dict[int, int],
    edge_index: np.ndarray | torch.Tensor,
    num_nodes: int,
) -> float:
    """Compute modularity of a partition.

    :param partition: Dict mapping node_id -> community_id.
    :param edge_index: Edge index tensor [2, num_edges].
    :param num_nodes: Total number of nodes.
    :return: Modularity score in [-0.5, 1].
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    # Count edges and degrees
    m = edge_index.shape[1]  # Number of edges
    if m == 0:
        return 0.0

    # Compute degree for each node
    degree = np.zeros(num_nodes)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        degree[src] += 1
        degree[dst] += 1

    # Compute modularity
    # Q = (1/2m) * sum_ij [(A_ij - k_i*k_j/2m) * delta(c_i, c_j)]
    q = 0.0
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if partition.get(src, -1) == partition.get(dst, -2):
            q += 1 - (degree[src] * degree[dst]) / (2 * m)

    return float(q / (2 * m))


def compute_community_metrics(
    partition: dict[int, int],
    edge_index: np.ndarray | torch.Tensor | None = None,
    num_nodes: int | None = None,
    true_labels: np.ndarray | list[int] | None = None,
) -> dict[str, float]:
    """Compute community detection metrics (modularity, NMI).

    :param partition: Dict mapping node_id -> community_id.
    :param edge_index: Edge index tensor [2, num_edges] (for modularity).
    :param num_nodes: Total number of nodes (for modularity).
    :param true_labels: Ground truth community labels (for NMI, optional).
    :return: Dict with available metrics.
    """
    results: dict[str, float] = {}

    # Compute modularity if edge_index provided
    if edge_index is not None and num_nodes is not None:
        results["modularity"] = compute_modularity(partition, edge_index, num_nodes)

    # Compute NMI if ground truth provided
    if true_labels is not None:
        if isinstance(true_labels, np.ndarray):
            true_labels = true_labels.tolist()
        pred_labels = [partition[i] for i in sorted(partition.keys())]
        results["nmi"] = float(normalized_mutual_info_score(true_labels, pred_labels))

    return results
