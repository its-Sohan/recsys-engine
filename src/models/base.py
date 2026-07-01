"""Common recommender interface.

Every model implements `fit(train_df)` and `recommend(user_id, k) -> list[int]`.
This uniform contract lets the evaluation harness treat popularity, SVD, NCF,
content-based, and hybrid models identically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Recommender(ABC):
    """Abstract base class for all recommenders in recsys-engine."""

    name: str = "base"

    @abstractmethod
    def fit(self, train: pd.DataFrame, **kwargs: Any) -> None:
        """Train on a ratings DataFrame with columns: userId, movieId, rating, timestamp."""

    @abstractmethod
    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        """Return top-k movie ids for `user_id`, ordered by predicted relevance desc."""

    def recommend_batch(
        self, user_ids: list[int], k: int = 10, exclude_seen: bool = True
    ) -> dict[int, list[int]]:
        """Convenience batch wrapper. Models may override for speed."""
        return {u: self.recommend(u, k=k, exclude_seen=exclude_seen) for u in user_ids}
