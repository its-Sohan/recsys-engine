"""Streamlit frontend for recsys-engine.

Two pages:
  1. User Recommendations — pick a user + model, see top movies
  2. Movie Explorer — search movies, find similar ones

Requires precomputed artifacts (run `python -m src.serving.precompute` first).

TMDB_API_KEY env var for movie posters — optional, fails silently.
"""
from __future__ import annotations

import os
import urllib.request
import urllib.error
import json

import numpy as np
import pandas as pd
import streamlit as st

from src.utils.config import ARTIFACTS_DIR

st.set_page_config(page_title="Movie Recommender", layout="wide")

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185"


# ── Load data once ──────────────────────────────────────────────────────

@st.cache_resource
def load_movies():
    return pd.read_parquet(ARTIFACTS_DIR / "movies.parquet")


@st.cache_resource
def load_precomputed():
    return pd.read_parquet(ARTIFACTS_DIR / "precomputed.parquet")


@st.cache_resource
def load_similar():
    return pd.read_parquet(ARTIFACTS_DIR / "similar.parquet")


def tmdb_poster(tmdb_id: int | None) -> str | None:
    if not tmdb_id or not TMDB_API_KEY:
        return None
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "recsys-engine/1.0"})
        with urllib.request.urlopen(req, timeout=3) as r:
            data = json.loads(r.read().decode())
            path = data.get("poster_path")
            if path:
                return TMDB_IMAGE_BASE + path
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError):
        pass
    return None


@st.cache_resource
def tmdb_poster_batch(tmdb_ids: list[int]) -> dict[int, str | None]:
    return {tid: tmdb_poster(tid) for tid in tmdb_ids if tid is not None}


def movie_card_html(title: str, genres: str, score: float | None, poster_url: str | None, rank: int) -> str:
    poster = (
        f'<img src="{poster_url}" width="92" style="border-radius:6px;float:left;margin-right:12px" />'
        if poster_url else ""
    )
    score_str = f"<small>score: {score:.3f}</small>" if score is not None else ""
    return f"""
    <div style="display:flex;align-items:center;margin-bottom:10px;border-bottom:1px solid #eee;padding:6px 0">
      {poster}
      <div>
        <strong>#{rank}</strong> {title}<br>
        <small style="color:#666">{genres}</small><br>
        {score_str}
      </div>
    </div>
    """


# ── App ─────────────────────────────────────────────────────────────────

movies = load_movies()
precomputed = load_precomputed()
similar = load_similar()
all_users = sorted(precomputed["user_id"].unique())
all_models = ["popularity", "svd", "ncf", "hybrid"]

page = st.sidebar.radio("Page", ["User Recommendations", "Movie Explorer"])

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(all_users):,} users · {len(movies):,} movies")

# ── Page 1: User Recommendations ──────────────────────────────────────

if page == "User Recommendations":
    st.title("🎬 Movie Recommender System")

    col1, col2 = st.columns([1, 3])
    with col1:
        user_id = st.selectbox("Select user", all_users, index=0)
        model_name = st.radio("Model", all_models, horizontal=True)
        top_k = st.slider("Top K", 5, 50, 10)

    recs = precomputed[
        (precomputed["user_id"] == user_id)
        & (precomputed["model"] == model_name)
        & (precomputed["rank"] < top_k)
    ].sort_values("rank")

    with col2:
        if recs.empty:
            st.info("No recommendations for this user with the selected model.")
        else:
            merged = recs.merge(movies, left_on="movie_id", right_on="movieId", how="left")
            tmdb_ids = merged["tmdbId"].dropna().astype(int).tolist()
            posters = tmdb_poster_batch(tmdb_ids)

            for _, row in merged.iterrows():
                tid = int(row["tmdbId"]) if pd.notna(row["tmdbId"]) else None
                poster = posters.get(tid)
                st.markdown(
                    movie_card_html(
                        row["title"], row["genres"], None, poster, row["rank"] + 1
                    ),
                    unsafe_allow_html=True,
                )

    # Model comparison table
    st.subheader("Model comparison for this user")
    all_recs = precomputed[
        (precomputed["user_id"] == user_id) & (precomputed["rank"] < 10)
    ]
    cmp = (
        all_recs.merge(movies[["movieId", "title"]], left_on="movie_id", right_on="movieId")
        .pivot_table(index="title", columns="model", values="rank", aggfunc="first")
        .fillna("-")
    )
    st.dataframe(cmp, use_container_width=True)

# ── Page 2: Movie Explorer ────────────────────────────────────────────

elif page == "Movie Explorer":
    st.title("🔍 Movie Explorer")

    search = st.text_input("Search movies", placeholder="e.g. Batman, Toy Story")

    if search:
        mask = movies["title"].str.contains(search, case=False, na=False)
        results = movies[mask]
        if results.empty:
            st.warning("No movies found.")
        else:
            st.caption(f"{len(results)} results")

            for _, row in results.iterrows():
                tid = int(row["tmdbId"]) if pd.notna(row["tmdbId"]) else None
                poster = tmdb_poster(tid) if tid else None
                with st.expander(f"{row['title']}  ({row['genres']})"):
                    st.markdown(
                        movie_card_html(row["title"], row["genres"], None, poster, 0),
                        unsafe_allow_html=True,
                    )

                    # Show similar movies
                    sim = similar[similar["movie_id"] == int(row["movieId"])].head(10)
                    if not sim.empty:
                        st.markdown("**Similar movies (by genre):**")
                        sim_movies = sim.merge(
                            movies[["movieId", "title", "genres"]],
                            left_on="similar_movie_id",
                            right_on="movieId",
                        )
                        for _, sr in sim_movies.iterrows():
                            st.markdown(
                                f"- **{sr['title']}** — {sr['genres']}",
                                unsafe_allow_html=True,
                            )
