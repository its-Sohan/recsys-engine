"""SVD collaborative filtering via scikit-surprise.

A thin wrapper around `surprise.SVD` implementing the project's Recommender
contract. SVD is the standard matrix-factorization baseline; it produces
both rating predictions (for RMSE) and user/item factors we can use to
score candidate items for top-K ranking.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import Recommender


class SVDRecommender(Recommender):
    name = "svd"

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        random_state: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self._algo = None
        self._trainset = None
        self._all_items: list[int] = []
        self._seen: dict[int, set[int]] = {}
        self._inner_to_raw_item: dict[int, int] = {}

    def fit(self, train: pd.DataFrame, **kwargs: Any) -> None:
        from surprise import SVD
        from surprise import Dataset
        from surprise import Reader

        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            train[["userId", "movieId", "rating"]], reader
        )
        trainset = data.build_full_trainset()

        algo = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state,
        )
        algo.fit(trainset)

        self._algo = algo
        self._trainset = trainset
        self._all_items = [
            trainset.to_raw_iid(i) for i in trainset.all_items()
        ]
        self._inner_to_raw_item = {
            i: trainset.to_raw_iid(i) for i in trainset.all_items()
        }
        self._seen = (
            train.groupby("userId")["movieId"].apply(set).to_dict()
        )

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        if self._algo is None or self._trainset is None:
            return []

        try:
            inner_uid = self._trainset.to_inner_uid(user_id)
        except ValueError:
            # Cold user: fall back to global popularity-ish ranking via item bias.
            return self._popularity_fallback(k, exclude_seen, user_id)

        seen = self._seen.get(user_id, set())
        candidates = [i for i in self._all_items if not (exclude_seen and i in seen)]

        # Score every candidate; this is O(|items|) per user which is fine for
        # batch precompute. For real-time serving we precompute top-N per user
        # into Parquet instead (see src/serving/precompute.py).
        scores = np.array(
            [self._algo.estimate(inner_uid, self._trainset.to_inner_iid(i)) for i in candidates]
        )
        if candidates:
            top_idx = np.argsort(-scores)[:k]
            return [candidates[i] for i in top_idx]
        return []

    def predict(self, user_id: int, item_id: int) -> float:
        """Raw rating prediction (used for RMSE on the test split)."""
        if self._algo is None:
            return 0.0
        return float(self._algo.predict(user_id, item_id).est)

    def _popularity_fallback(self, k: int, exclude_seen: bool, user_id: int) -> list[int]:
        seen = self._seen.get(user_id, set())
        # Cheap fallback: rank items by their bias term (proxy for popularity).
        biases = self._algo.bu if hasattr(self._algo, "bu") else None
        if biases is None:
            return self._all_items[:k]
        order = np.argsort(-biases)
        out = []
        for inner in order:
            raw = self._inner_to_raw_item.get(int(inner))
            if raw is None:
                continue
            if exclude_seen and raw in seen:
                continue
            out.append(int(raw))
            if len(out) >= k:
                break
        return out
