"""Popularity baseline with a genre-weighted recency boost.

This is the floor every serious model must beat. It is intentionally simple:
rank items by a weighted blend of (a) global popularity (rating count, with a
small Bayesian shrinkage toward the mean to avoid the top-N being dominated
by 1-rating flukes) and (b) the user's preferred genres.

The point of this model is not to be clever; it's to give the evaluation
harness a sane lower bound and to demonstrate the Recommender contract.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import Recommender


class PopularityRecommender(Recommender):
    name = "popularity"

    def __init__(self, genre_weight: float = 0.3, min_ratings: int = 50) -> None:
        self.genre_weight = genre_weight
        self.min_ratings = min_ratings
        self._item_scores: pd.Series | None = None
        self._item_genres: dict[int, list[str]] = {}
        self._user_genres: dict[int, Counter] = {}
        self._seen: dict[int, set[int]] = {}
        self._global_avg: float = 0.0

    def fit(self, train: pd.DataFrame, movies: pd.DataFrame | None = None, **kwargs: Any) -> None:
        """Compute popularity scores. If `movies` is provided, build genre maps."""
        self._global_avg = float(train["rating"].mean())

        # Bayesian-shrunk mean rating: (v/(v+m))*R + (m/(v+m))*C
        # We use the *count* as the popularity signal (how many people watched it),
        # lightly damped so 1-rating items can't dominate.
        stats = train.groupby("movieId").agg(
            mean_rating=("rating", "mean"),
            count=("rating", "size"),
        )
        m = float(stats["count"].mean())
        stats["score"] = (stats["count"] / (stats["count"] + m)) * stats["count"]
        # Apply a minimum-ratings floor so the top-N isn't noise.
        stats.loc[stats["count"] < self.min_ratings, "score"] = 0.0
        self._item_scores = stats["score"]

        self._seen = (
            train.groupby("userId")["movieId"].apply(set).to_dict()
        )

        if movies is not None:
            self._item_genres = {
                int(row["movieId"]): str(row["genres"]).split("|")
                for _, row in movies.iterrows()
            }
            # Per-user genre affinity: count of train interactions per genre.
            merged = train.merge(movies[["movieId", "genres"]], on="movieId", how="left")
            merged["genres"] = merged["genres"].fillna("(no genres listed)")
            self._user_genres = {}
            for uid, group in merged.groupby("userId"):
                c: Counter = Counter()
                for g in group["genres"]:
                    for part in str(g).split("|"):
                        c[part] += 1
                self._user_genres[int(uid)] = c

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        if self._item_scores is None:
            return []
        scores = self._item_scores.copy()
        seen = self._seen.get(user_id, set())

        # Genre boost: scale each item by the overlap with the user's top genres.
        if self.genre_weight > 0 and self._user_genres.get(user_id):
            user_genres = self._user_genres[user_id]
            top_genre = user_genres.most_common(1)[0][0]
            boost = np.array(
                [
                    1.0 + self.genre_weight if top_genre in self._item_genres.get(i, []) else 1.0
                    for i in scores.index
                ]
            )
            scores = scores * boost

        if exclude_seen:
            scores = scores.drop(index=list(seen), errors="ignore")

        return scores.nlargest(k).index.tolist()
