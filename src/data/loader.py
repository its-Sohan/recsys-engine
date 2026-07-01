"""Data loading utilities for MovieLens 25M.

Loads ratings, movies, and tags into typed DataFrames and produces
a **time-based** train/test split. Random K-fold splits leak future
interactions and inflate ranking metrics; we instead cut by timestamp
so the test set is strictly "the future" relative to training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.config import (
    IMPLICIT_RATING_THRESHOLD,
    MOVIES_PATH,
    PROCESSED_DIR,
    RATINGS_PATH,
    TAGS_PATH,
    TEST_FRACTION,
)

DTYPE_RATINGS = {
    "userId": np.int32,
    "movieId": np.int32,
    "rating": np.float32,
    "timestamp": np.int64,
}


@dataclass
class DataBundle:
    """Container holding train/test ratings plus movie metadata."""

    train: pd.DataFrame
    test: pd.DataFrame
    movies: pd.DataFrame
    tags: pd.DataFrame
    n_users: int
    n_items: int


def load_ratings() -> pd.DataFrame:
    """Load the full ratings table with memory-efficient dtypes."""
    df = pd.read_csv(RATINGS_PATH, dtype=DTYPE_RATINGS)
    return df


def load_movies() -> pd.DataFrame:
    """Load movies with genres (pipe-separated) and a parsed year from the title."""
    df = pd.read_csv(MOVIES_PATH, dtype={"movieId": np.int32, "title": "string"})
    df["genres"] = df["genres"].fillna("(no genres listed)").astype("string")
    df["year"] = df["title"].str.extract(r"\((\d{4})\)\s*$").astype("Int16")
    return df


def load_tags() -> pd.DataFrame:
    """Load user-applied tags."""
    df = pd.read_csv(
        TAGS_PATH,
        dtype={
            "userId": np.int32,
            "movieId": np.int32,
            "tag": "string",
            "timestamp": np.int64,
        },
    )
    return df


def time_based_split(
    ratings: pd.DataFrame, test_fraction: float = TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings by timestamp: the latest `test_fraction` of interactions
    become the test set. Every user in the test set is guaranteed to also
    appear in train (we drop cold-start users from the test set because
    evaluating them is meaningless for personalized recommenders).
    """
    ratings = ratings.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    n_test = int(len(ratings) * test_fraction)
    train = ratings.iloc[:-n_test].copy()
    test = ratings.iloc[-n_test:].copy()

    train_users = set(train["userId"].unique())
    before = len(test)
    test = test[test["userId"].isin(train_users)].copy()
    dropped = before - len(test)
    if dropped:
        print(
            f"Dropped {dropped} test interactions ({dropped / before:.2%} of test) "
            f"from users not present in train (cold-start)."
        )
    return train, test


def build_implicit(ratings: pd.DataFrame) -> pd.DataFrame:
    """Binarize ratings: >= IMPLICIT_RATING_THRESHOLD -> 1, else 0."""
    out = ratings[["userId", "movieId", "timestamp"]].copy()
    out["label"] = (ratings["rating"] >= IMPLICIT_RATING_THRESHOLD).astype(np.int8)
    return out


def build_id_maps(train: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """Build contiguous user/item id maps for embedding layers (used by NCF).

    Returns:
        user2idx, idx2user, item2idx, idx2item
    """
    users = np.sort(train["userId"].unique())
    items = np.sort(train["movieId"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: it for it, i in item2idx.items()}
    return user2idx, idx2user, item2idx, idx2item


def save_processed(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Persist processed splits to Parquet for fast reloading."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    print(f"Saved train ({len(train):,}) and test ({len(test):,}) to {PROCESSED_DIR}")


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    return train, test


def load_data() -> DataBundle:
    """Load ratings, split by time, load metadata. Convenience entrypoint."""
    ratings = load_ratings()
    train, test = time_based_split(ratings)
    movies = load_movies()
    tags = load_tags()
    save_processed(train, test)
    n_users = train["userId"].nunique()
    n_items = train["movieId"].nunique()
    return DataBundle(train=train, test=test, movies=movies, tags=tags, n_users=n_users, n_items=n_items)
