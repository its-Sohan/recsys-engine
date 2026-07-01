"""Unit tests for the from-scratch ranking metrics.

These verify mathematical correctness against hand-computed expected values.
"""
from __future__ import annotations

import math

from src.evaluate.metrics import (
    average_precision_at_k,
    evaluate_ranking,
    hit_ratio_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)


def test_precision_at_k_basic():
    # 2 of top-5 are relevant -> 0.4
    recs = [101, 102, 103, 104, 105]
    relevant = {102, 104, 999}
    assert precision_at_k(recs, relevant, 5) == 2 / 5


def test_precision_at_k_truncates_to_k():
    recs = [1, 2, 3, 4, 5, 6, 7]
    relevant = {1, 7}
    # only top-3 considered: {1,2,3} -> 1 hit -> 1/3
    assert precision_at_k(recs, relevant, 3) == 1 / 3


def test_recall_at_k():
    recs = [1, 2, 3]
    relevant = {1, 3, 5}
    # 2 of 3 relevant recovered -> 2/3
    assert recall_at_k(recs, relevant, 3) == 2 / 3


def test_hit_ratio_hit():
    recs = [1, 2, 3]
    relevant = {2}
    assert hit_ratio_at_k(recs, relevant, 3) == 1


def test_hit_ratio_miss():
    recs = [1, 2, 3]
    relevant = {9}
    assert hit_ratio_at_k(recs, relevant, 3) == 0


def test_ndcg_perfect_ordering():
    # All relevant items ranked first -> NDCG = 1.0
    recs = [1, 2, 3]
    relevant = {1, 2, 3}
    assert abs(ndcg_at_k(recs, relevant, 3) - 1.0) < 1e-9


def test_ndcg_known_value():
    # recs: positions 1..4, relevant at positions 2 and 4 (0-indexed 1,3)
    recs = [10, 11, 12, 13]
    relevant = {11, 13}
    # DCG = 1/log2(3) + 1/log2(5)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    # IDCG (ideal: both at front) = 1/log2(2) + 1/log2(3)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    expected = dcg / idcg
    assert abs(ndcg_at_k(recs, relevant, 4) - expected) < 1e-9


def test_ndcg_empty_relevant_is_zero():
    assert ndcg_at_k([1, 2, 3], set(), 3) == 0.0


def test_average_precision():
    # AP@k: precision at each hit position, averaged.
    recs = [1, 2, 3, 4, 5]
    relevant = {2, 4}
    # hit at pos2 -> P=1/2 ; hit at pos4 -> P=2/4 ; sum=1.0 ; /min(5,2)=2 -> 0.5
    assert abs(average_precision_at_k(recs, relevant, 5) - 0.5) < 1e-9


def test_rmse():
    preds = [3.0, 4.0, 5.0]
    actual = [4.0, 4.0, 4.0]
    # errors: 1, 0, 1 -> mean sq = 2/3 -> rmse = sqrt(2/3)
    assert abs(rmse(preds, actual) - math.sqrt(2 / 3)) < 1e-9


def test_evaluate_ranking_aggregate():
    recommendations = {
        1: [10, 20, 30],
        2: [10, 40, 50],
    }
    relevant = {
        1: {20, 30},   # 2 hits in top-3
        2: {99},       # 0 hits
    }
    res = evaluate_ranking(recommendations, relevant, k=3)
    # precision@3: user1=2/3, user2=0 -> mean 1/3
    assert abs(res["precision@3"] - (1 / 3)) < 1e-9
    # hit_ratio@3: user1=1, user2=0 -> mean 0.5
    assert abs(res["hit_ratio@3"] - 0.5) < 1e-9
