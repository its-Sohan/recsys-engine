"""Smoke tests for data splitting and the popularity baseline.

These run on a tiny synthetic ratings frame so CI can validate the logic
without downloading the 250MB MovieLens dataset.
"""
from __future__ import annotations

import pandas as pd

from src.data.loader import build_id_maps, build_implicit, time_based_split
from src.models.baseline import PopularityRecommender


def _toy_ratings() -> pd.DataFrame:
    rows = [
        # user, movie, rating, timestamp
        (1, 101, 5.0, 1000),
        (1, 102, 4.0, 1001),
        (1, 103, 2.0, 1002),
        (2, 101, 4.0, 1003),
        (2, 104, 5.0, 1004),
        (2, 105, 3.0, 1005),
        (3, 102, 5.0, 1006),
        (3, 104, 4.0, 1007),
        (3, 106, 5.0, 1008),
        (1, 107, 4.0, 1009),  # latest -> test
        (2, 108, 5.0, 1010),  # latest -> test
        (3, 109, 4.0, 1011),  # latest -> test
    ]
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


def test_time_based_split_preserves_ordering():
    ratings = _toy_ratings()
    train, test = time_based_split(ratings, test_fraction=0.25)
    # Test timestamps must all be >= max train timestamp (strictly future).
    assert test["timestamp"].min() >= train["timestamp"].max()
    # 12 rows * 0.25 = 3 in test
    assert len(test) == 3


def test_time_based_split_drops_cold_start_users():
    ratings = _toy_ratings()
    # Make user 4 appear ONLY in the test window (cold start -> should be dropped).
    cold = pd.DataFrame(
        [(4, 200, 5.0, 9999)], columns=["userId", "movieId", "rating", "timestamp"]
    )
    ratings = pd.concat([ratings, cold], ignore_index=True)
    train, test = time_based_split(ratings, test_fraction=0.3)
    assert 4 not in test["userId"].unique()


def test_build_implicit_threshold():
    ratings = _toy_ratings()
    impl = build_implicit(ratings)
    # rating 5.0 and 4.0 -> label 1 ; 2.0 and 3.0 -> label 0
    assert impl.loc[(impl["movieId"] == 103) & (impl["userId"] == 1), "label"].iloc[0] == 0
    assert impl.loc[(impl["movieId"] == 101) & (impl["userId"] == 1), "label"].iloc[0] == 1


def test_build_id_maps_contiguous():
    train = _toy_ratings().iloc[:9]  # drop the test rows
    u2i, i2u, item2i, i2item = build_id_maps(train)
    assert len(u2i) == 3
    assert len(item2i) == 6
    # indices must be contiguous 0..n-1
    assert sorted(u2i.values()) == list(range(3))
    assert sorted(item2i.values()) == list(range(6))
    # round trip
    assert i2u[u2i[1]] == 1
    assert i2item[item2i[101]] == 101


def test_popularity_recommender_excludes_seen():
    ratings = _toy_ratings().iloc[:9]  # train portion only
    movies = pd.DataFrame(
        [
            {"movieId": 101, "title": "A (1990)", "genres": "Action|Comedy"},
            {"movieId": 102, "title": "B (1991)", "genres": "Action"},
            {"movieId": 103, "title": "C (1992)", "genres": "Drama"},
            {"movieId": 104, "title": "D (1993)", "genres": "Action"},
            {"movieId": 105, "title": "E (1994)", "genres": "Drama"},
            {"movieId": 106, "title": "F (1995)", "genres": "Comedy"},
        ]
    )
    model = PopularityRecommender(min_ratings=1)
    model.fit(ratings, movies=movies)
    recs = model.recommend(user_id=1, k=3, exclude_seen=True)
    # User 1 has seen 101,102,103 in train -> none should appear.
    assert not (set(recs) & {101, 102, 103})
    assert len(recs) == 3


def test_popularity_recommender_returns_k():
    ratings = _toy_ratings().iloc[:9]
    movies = pd.DataFrame(
        [{"movieId": m, "title": f"M{m} (2000)", "genres": "Action"} for m in [101, 102, 103, 104, 105, 106]]
    )
    model = PopularityRecommender(min_ratings=1)
    model.fit(ratings, movies=movies)
    recs = model.recommend(user_id=1, k=4)
    # Only 6 items total, user 1 has seen 3 -> only 3 unseen candidates exist.
    assert len(recs) == 3
    assert not (set(recs) & {101, 102, 103})
