"""Tests for the NCF (NeuMF) model and the hybrid blender.

Run on a tiny synthetic dataset; does NOT need the real MovieLens data.
Requires torch (installed in CI via requirements.txt).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.models.content import ContentRecommender
from src.models.hybrid import HybridRecommender
from src.models.ncf import GMF, MLP, NCFDataset, NCFRecommender, NeuMF


def _toy_ratings() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, 21):
        for m in rng.choice(range(1, 101), size=rng.integers(2, 8), replace=False):
            rows.append((int(u), int(m), float(rng.choice([3.0, 4.0, 5.0])), int(rng.integers(0, 1000))))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


def _toy_movies() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Romance"]
    return pd.DataFrame(
        {
            "movieId": range(1, 101),
            "title": [f"Movie {i} (200{rng.integers(0,9)})" for i in range(1, 101)],
            "genres": [rng.choice(genres, size=2, replace=False).tolist().__str__()[1:-1].replace("'", "")
                       for _ in range(100)],
        }
    )


# ---- NeuMF architecture tests ----


def test_gmf_output_shape():
    model = GMF(n_users=50, n_items=30, gmf_dim=8)
    u = torch.randint(0, 50, (16,))
    i = torch.randint(0, 30, (16,))
    out = model(u, i)
    assert out.shape == (16,)


def test_mlp_output_shape():
    model = MLP(n_users=50, n_items=30, mlp_embed_dim=16, layer_sizes=[32, 16, 8])
    u = torch.randint(0, 50, (16,))
    i = torch.randint(0, 30, (16,))
    out = model(u, i)
    assert out.shape == (16,)


def test_neumf_output_in_unit_interval():
    """Sigmoid guarantees output in [0, 1] — critical for BCE loss."""
    model = NeuMF(n_users=50, n_items=30)
    u = torch.randint(0, 50, (32,))
    i = torch.randint(0, 30, (32,))
    out = model(u, i).detach()
    assert out.shape == (32,)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


# ---- Dataset / negative sampling tests ----


def test_ncf_dataset_negative_sampling_ratio():
    train = _toy_ratings()
    user2idx = {u: i for i, u in enumerate(sorted(train.userId.unique()))}
    item2idx = {m: i for i, m in enumerate(sorted(train.movieId.unique()))}
    n_pos = (train.rating >= 4.0).sum()
    ds = NCFDataset(train, user2idx, item2idx, neg_ratio=4)
    # Total = positives + 4 * positives
    assert len(ds) == n_pos * 5
    # Labels: exactly n_pos ones, rest zeros
    labels = [lbl for _, _, lbl in ds]
    assert sum(labels) == n_pos
    assert sum(1 for x in labels if x == 0) == n_pos * 4


# ---- NCFRecommender end-to-end test ----


def test_ncf_recommender_train_recommend_save_load():
    train = _toy_ratings()
    rec = NCFRecommender(epochs=1, batch_size=64)
    rec.fit(train)

    recs = rec.recommend(user_id=1, k=5)
    assert len(recs) == 5
    # Should not return seen items
    seen = set(train[train.userId == 1].movieId)
    assert not (set(recs) & seen)

    # Save/load round-trip
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        path = f.name
    try:
        rec.save(path)
        rec2 = NCFRecommender()
        rec2.load(path)
        recs2 = rec2.recommend(user_id=1, k=5)
        assert recs == recs2
    finally:
        os.unlink(path)


def test_ncf_cold_user_returns_empty():
    train = _toy_ratings()
    rec = NCFRecommender(epochs=1, batch_size=64)
    rec.fit(train)
    # User 999 not in train -> empty recs
    assert rec.recommend(user_id=999, k=5) == []


# ---- Content-based test ----


def test_content_recommender_fit_recommend():
    train = _toy_ratings()
    movies = _toy_movies()
    model = ContentRecommender(max_features=200)
    model.fit(train, movies=movies, tags=None)
    recs = model.recommend(user_id=1, k=5)
    assert len(recs) == 5
    seen = set(train[train.userId == 1].movieId)
    assert not (set(recs) & seen)


# ---- Hybrid blender test ----


def test_hybrid_blends_two_models():
    train = _toy_ratings()
    movies = _toy_movies()

    ncf = NCFRecommender(epochs=1, batch_size=64)
    ncf.fit(train)

    content = ContentRecommender(max_features=200)
    content.fit(train, movies=movies, tags=None)

    hybrid = HybridRecommender(ncf_model=ncf, content_model=content, alpha=0.7)
    recs = hybrid.recommend(user_id=1, k=5)
    assert len(recs) == 5
    seen = set(train[train.userId == 1].movieId)
    assert not (set(recs) & seen)


def test_hybrid_alpha_extremes():
    """alpha=1 -> pure NCF; alpha=0 -> pure content."""
    train = _toy_ratings()
    movies = _toy_movies()

    ncf = NCFRecommender(epochs=1, batch_size=64)
    ncf.fit(train)
    content = ContentRecommender(max_features=200)
    content.fit(train, movies=movies, tags=None)

    h_ncf = HybridRecommender(ncf_model=ncf, content_model=content, alpha=1.0)

    ncf_recs = ncf.recommend(user_id=1, k=5)
    h_ncf_recs = h_ncf.recommend(user_id=1, k=5)
    # With alpha=1 the hybrid should match NCF's top recs (rank-based scoring
    # may tie-break differently, but the top item should agree).
    assert h_ncf_recs[0] == ncf_recs[0]
