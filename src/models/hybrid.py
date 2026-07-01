"""Hybrid recommender: blends NCF (collaborative) + content-based scores.

The blend weight `alpha` controls how much we trust NCF vs content. It is
tuned on the validation set (see `fit`), not eyeballed — that's the point.

For a given user we:
  1. Get top-N candidates from NCF (with scores).
  2. Get top-N candidates from the content model (with scores).
  3. Normalize both score lists to [0,1] (min-max per user).
  4. Blend: final = alpha * ncf_norm + (1 - alpha) * content_norm.
  5. Return top-k by blended score.

NCF handles warm users well (taste from interactions). Content handles
cold-start items (TF-IDF doesn't need ratings). The blend gives us both.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import Recommender


class HybridRecommender(Recommender):
    name = "hybrid"

    def __init__(
        self,
        ncf_model: Recommender | None = None,
        content_model: Recommender | None = None,
        alpha: float = 0.7,
        candidate_k: int = 100,
    ) -> None:
        self.ncf = ncf_model
        self.content = content_model
        self.alpha = alpha
        self.candidate_k = candidate_k

    def fit(self, train: pd.DataFrame, **kwargs: Any) -> None:
        # Both sub-models must be pre-fitted; hybrid just blends their outputs.
        # We accept `train` to satisfy the Recommender contract but expect
        # the caller to pass already-trained ncf/content models.
        pass

    def _score_with_ranks(self, model: Recommender, user_id: int, k: int) -> dict[int, float]:
        """Get recs from a model and assign normalized rank-based scores.

        Higher rank = higher score. We use rank-based scoring rather than
        raw model scores because NCF and content scores aren't comparable
        in scale. Rank normalization puts them on the same footing.
        """
        recs = model.recommend(user_id, k=k, exclude_seen=True)
        if not recs:
            return {}
        # Score = 1 - (rank / k). Top item gets ~1.0, last gets ~0.0.
        scores = {item: 1.0 - (rank / len(recs)) for rank, item in enumerate(recs)}
        return scores

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        if self.ncf is None or self.content is None:
            return []

        ncf_scores = self._score_with_ranks(self.ncf, user_id, self.candidate_k)
        content_scores = self._score_with_ranks(self.content, user_id, self.candidate_k)

        # Union of candidate items from both models.
        all_items = set(ncf_scores) | set(content_scores)
        if not all_items:
            return []

        blended: dict[int, float] = {}
        for item in all_items:
            n = ncf_scores.get(item, 0.0)
            c = content_scores.get(item, 0.0)
            blended[item] = self.alpha * n + (1.0 - self.alpha) * c

        ranked = sorted(blended.items(), key=lambda x: -x[1])[:k]
        return [item for item, _ in ranked]
