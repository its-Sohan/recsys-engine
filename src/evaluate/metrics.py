"""Ranking and rating metrics, implemented from scratch (no implicit-lib crutch).

For ranking metrics we evaluate **per-user top-K recommendation lists**
against the set of items the user actually interacted with in the test set.

Conventions:
  - `recommended` : list[int] of item ids, ordered by predicted score desc, len >= K
  - `relevant`    : set[int] of item ids the user liked in the test split
  - `k`           : cutoff

These functions operate on plain python iterables so they're trivially testable.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


def precision_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of the top-k recommended items that are relevant."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    topk = list(recommended)[:k]
    hits = sum(1 for item in topk if item in relevant_set)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of relevant items recovered in the top-k."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    topk = list(recommended)[:k]
    hits = sum(1 for item in topk if item in relevant_set)
    return hits / len(relevant_set)


def hit_ratio_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> int:
    """1.0 if ANY relevant item appears in the top-k, else 0.0. (Binary hit.)"""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0
    topk = list(recommended)[:k]
    return 1 if any(item in relevant_set for item in topk) else 0


def average_precision_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """AP@k — sum of precision@i for each hit, normalized by min(k, |relevant|)."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    topk = list(recommended)[:k]
    hits = 0
    score = 0.0
    for i, item in enumerate(topk, start=1):
        if item in relevant_set:
            hits += 1
            score += hits / i
    return score / min(k, len(relevant_set))


def ndcg_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ k.

    DCG = sum_{i=1..k} rel_i / log2(i + 1),  rel_i in {0,1} for ranking.
    IDCG = DCG of the ideal (perfect) ordering, capped at |relevant|.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    topk = list(recommended)[:k]

    dcg = 0.0
    for i, item in enumerate(topk, start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(i + 1)

    n_rel = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def rmse(predicted: Iterable[float], actual: Iterable[float]) -> float:
    """Root mean squared error for rating prediction (used by SVD)."""
    predicted = np.fromiter(predicted, dtype=np.float64)
    actual = np.fromiter(actual, dtype=np.float64)
    if predicted.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


# ---- Aggregate evaluation over a population of users ---------------------


def evaluate_ranking(
    recommendations: dict[int, Sequence[int]],
    relevant_by_user: dict[int, set[int]],
    k: int = 10,
) -> dict[str, float]:
    """Compute mean ranking metrics across all users that have test interactions.

    Args:
        recommendations: user_id -> ordered list of recommended item ids
        relevant_by_user: user_id -> set of relevant item ids (from test split)
        k: cutoff

    Returns:
        dict with precision@k, recall@k, ndcg@k, hit_ratio@k, map@k
    """
    users = [u for u, rel in relevant_by_user.items() if len(rel) > 0]
    if not users:
        return {m: 0.0 for m in ("precision", "recall", "ndcg", "hit_ratio", "map")}

    precisions, recalls, ndcgs, hits, aps = [], [], [], [], []
    for user in users:
        recs = recommendations.get(user, [])
        rel = relevant_by_user[user]
        precisions.append(precision_at_k(recs, rel, k))
        recalls.append(recall_at_k(recs, rel, k))
        ndcgs.append(ndcg_at_k(recs, rel, k))
        hits.append(hit_ratio_at_k(recs, rel, k))
        aps.append(average_precision_at_k(recs, rel, k))

    return {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"hit_ratio@{k}": float(np.mean(hits)),
        f"map@{k}": float(np.mean(aps)),
    }
