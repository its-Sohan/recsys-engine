"""Precompute recommendations and metadata for the Streamlit frontend."""
from __future__ import annotations

from collections import defaultdict
from time import time

import numpy as np
import pandas as pd

from src.data.loader import load_data
from src.utils.config import ARTIFACTS_DIR
from src.utils.io import load_artifact

MODEL_NAMES = ["popularity", "svd", "ncf", "hybrid"]


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time()
    bundle = load_data()
    print(f"data loaded in {time()-t0:.0f}s", flush=True)

    movies = bundle.movies.copy()
    links = pd.read_csv("data/raw/ml-latest-small/links.csv", dtype={"movieId": np.int32})
    movies = movies.merge(links[["movieId", "tmdbId", "imdbId"]], on="movieId", how="left")
    movies.to_parquet(ARTIFACTS_DIR / "movies.parquet", index=False)
    print(f"saved movies ({len(movies):,} rows)", flush=True)

    t1 = time()
    movies["genres_list"] = movies["genres"].str.split("|")
    genre_to_movies = defaultdict(set)
    for _, row in movies.iterrows():
        for g in row["genres_list"]:
            genre_to_movies[g].add(int(row["movieId"]))

    genres_by_id = {int(r.movieId): r.genres_list for r in movies.itertuples()}
    movie_ids = list(genres_by_id.keys())
    sim_rows = []
    for mid in movie_ids:
        genre_set = set(genres_by_id[mid])
        candidates = set()
        for g in genre_set:
            candidates.update(genre_to_movies[g])
        candidates.discard(mid)
        scored = []
        for oid in candidates:
            o_genres = set(genres_by_id[oid])
            overlap = len(genre_set & o_genres)
            if overlap > 0:
                scored.append((oid, overlap))
        scored.sort(key=lambda x: -x[1], reverse=True)
        for rank, (oid, sc) in enumerate(scored[:20]):
            sim_rows.append((mid, rank, oid, sc))
    sim_df = pd.DataFrame(sim_rows, columns=["movie_id", "rank", "similar_movie_id", "score"], dtype=np.int32)
    sim_df.to_parquet(ARTIFACTS_DIR / "similar.parquet", index=False)
    print(f"saved similar ({len(sim_df):,} rows) in {time()-t1:.0f}s", flush=True)

    # Precompute top-100 recs per model per user.
    users = sorted(int(u) for u in bundle.train["userId"].unique())
    all_rows = []
    for name in MODEL_NAMES:
        path = ARTIFACTS_DIR / f"{name}.joblib"
        if not path.exists():
            print(f"  [skip] {name}", flush=True)
            continue
        model = load_artifact(path)
        print(f"  precomputing {name} for {len(users):,} users...", flush=True)
        t2 = time()
        for uid in users:
            recs = model.recommend(uid, k=100, exclude_seen=True)
            for rank, mid in enumerate(recs):
                all_rows.append((uid, name, rank, mid))
        print(f"    done in {time()-t2:.1f}s", flush=True)

    df = pd.DataFrame(all_rows, columns=["user_id", "model", "rank", "movie_id"])
    df.to_parquet(ARTIFACTS_DIR / "precomputed.parquet", index=False)
    print(f"saved precomputed ({len(df):,} rows)", flush=True)
    print(f"total: {time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
