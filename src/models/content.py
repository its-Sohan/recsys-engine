"""Content-based recommender using TF-IDF on tags + genres.

Purpose: handle COLD-START. Collaborative models (SVD, NCF) cannot score
items that have no ratings in the train set, or users not in the train set.
This model builds a TF-IDF representation of each movie from its genres
and user-applied tags, then recommends via cosine similarity to items the
user already liked.

This is deliberately simple — it's the cold-start complement to NCF, and
its scores get blended by the hybrid model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base import Recommender
from src.utils.config import IMPLICIT_RATING_THRESHOLD


class ContentRecommender(Recommender):
    name = "content"

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 1)) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._sim_matrix: np.ndarray | None = None
        self._movie_ids: np.ndarray = None
        self._movie_id_to_idx: dict[int, int] = {}
        self._user_profile: dict[int, np.ndarray] = {}
        self._seen: dict[int, set[int]] = {}

    def _build_movie_text(self, movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
        """Combine genres + aggregated tags into one text field per movie."""
        movies = movies.copy()
        movies["genres_text"] = movies["genres"].fillna("").str.replace("|", " ", regex=False)

        if tags is not None and len(tags) > 0:
            tag_text = (
                tags.groupby("movieId")["tag"]
                .apply(lambda x: " ".join(x.dropna().astype(str)))
                .reset_index()
                .rename(columns={"tag": "tags_text"})
            )
            movies = movies.merge(tag_text, on="movieId", how="left")
        else:
            movies["tags_text"] = ""

        movies["content_text"] = movies["genres_text"] + " " + movies["tags_text"].fillna("")
        return movies

    def fit(
        self,
        train: pd.DataFrame,
        movies: pd.DataFrame | None = None,
        tags: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        if movies is None:
            raise ValueError("ContentRecommender requires movies metadata.")

        movie_df = self._build_movie_text(movies, tags)

        # Fit TF-IDF over the combined genre+tag text.
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(movie_df["content_text"])
        self._movie_ids = movie_df["movieId"].to_numpy()
        self._movie_id_to_idx = {int(mid): i for i, mid in enumerate(self._movie_ids)}

        # Precompute item-item cosine similarity. With 62k movies this is
        # ~62k x 62k — too big to hold dense. We keep it sparse and query
        # on the fly per user's liked items instead.
        # (Full dense sim would be ~28GB. Sparse top-k query is fine.)

        # Build per-user taste profiles: mean TF-IDF vector of items the user
        # rated >= threshold, then we'll rank by cosine sim to that profile.
        positives = train[train["rating"] >= IMPLICIT_RATING_THRESHOLD]
        self._seen = train.groupby("userId")["movieId"].apply(set).to_dict()

        for uid, group in positives.groupby("userId"):
            idxs = [self._movie_id_to_idx[m] for m in group["movieId"] if m in self._movie_id_to_idx]
            if idxs:
                self._user_profile[int(uid)] = np.asarray(
                    self._tfidf_matrix[idxs].mean(axis=0)
                ).ravel()

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        if self._tfidf_matrix is None:
            return []

        profile = self._user_profile.get(user_id)
        if profile is None:
            # Cold user: fall back to globally popular-ish items (highest TF-IDF norm).
            norms = np.asarray(self._tfidf_matrix.norm(axis=0)).ravel()
            scores = norms
        else:
            # Cosine similarity between user profile and every movie.
            scores = np.asarray(
                self._tfidf_matrix.dot(profile.reshape(-1, 1)).ravel()
            )

        seen = self._seen.get(user_id, set())
        if exclude_seen and seen:
            seen_idx = [self._movie_id_to_idx[m] for m in seen if m in self._movie_id_to_idx]
            scores = scores.copy()
            scores[seen_idx] = -np.inf

        top_idx = np.argsort(-scores)[:k]
        return [int(self._movie_ids[i]) for i in top_idx]
