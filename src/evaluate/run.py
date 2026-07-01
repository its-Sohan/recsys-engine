"""Evaluate trained recommenders on the time-based test split.

For each model we:
  1. Build `relevant_by_user` = {user: {items rated >= threshold in test}}.
  2. Generate top-K recommendations per user (excluding train-seen items).
  3. Compute ranking metrics (precision/recall/ndcg/hit_ratio/map @K).
  4. For rating-prediction models (SVD), also compute RMSE on held-out ratings.

Results are written to:
  - artifacts/results.csv
  - artifacts/results.md   (markdown table for the README)

Usage:
    make evaluate
    python -m src.evaluate.run --k 10 --sample 5000   # sample users for speed
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import load_data
from src.evaluate.metrics import evaluate_ranking, rmse
from src.models.base import Recommender
from src.utils.config import ARTIFACTS_DIR, IMPLICIT_RATING_THRESHOLD
from src.utils.io import load_artifact


def build_relevant_by_user(test: pd.DataFrame, threshold: float) -> dict[int, set[int]]:
    pos = test[test["rating"] >= threshold]
    return pos.groupby("userId")["movieId"].apply(set).to_dict()


def evaluate_model(
    model: Recommender,
    relevant_by_user: dict[int, set[int]],
    test: pd.DataFrame,
    k: int,
    sample: int | None,
) -> dict[str, Any]:
    users = [u for u, rel in relevant_by_user.items() if len(rel) > 0]
    if sample and len(users) > sample:
        rng = np.random.default_rng(42)
        users = rng.choice(users, size=sample, replace=False).tolist()
    print(f"  Evaluating on {len(users):,} users (k={k})")

    t0 = time.time()
    recommendations = model.recommend_batch(users, k=k, exclude_seen=True)
    rank_metrics = evaluate_ranking(recommendations, relevant_by_user, k=k)
    elapsed = time.time() - t0

    result: dict[str, Any] = {"model": model.name, "eval_users": len(users), "eval_time_s": round(elapsed, 1)}
    result.update(rank_metrics)

    # RMSE for rating-prediction models (SVD).
    if hasattr(model, "predict"):
        preds = [model.predict(int(r.userId), int(r.movieId)) for r in test.itertuples()]
        actual = test["rating"].tolist()
        result["rmse"] = round(rmse(preds, actual), 4)
    else:
        result["rmse"] = None

    return result


def to_markdown(rows: list[dict[str, Any]]) -> str:
    cols = ["model", "precision@K", "recall@K", "ndcg@K", "hit_ratio@K", "map@K", "rmse"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c.replace("@K", f"@{r.get('k', 10)}") if c.endswith("@K") else c)
            vals.append("-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sample", type=int, default=5000, help="Cap on users evaluated (speed)")
    parser.add_argument("--models", nargs="*", default=["popularity", "svd", "ncf"])
    args = parser.parse_args()

    bundle = load_data()
    relevant_by_user = build_relevant_by_user(bundle.test, IMPLICIT_RATING_THRESHOLD)
    print(f"Test users with positive interactions: {len(relevant_by_user):,}")

    rows: list[dict[str, Any]] = []
    for name in args.models:
        path = ARTIFACTS_DIR / f"{name}.joblib"
        if not path.exists():
            print(f"  [skip] {name}: artifact not found at {path}")
            if name == "ncf":
                print("         Train NCF on Colab: notebooks/03_ncf_training.ipynb")
                print("         Then: python -m src.models.train --load-ncf artifacts/ncf.pth")
            else:
                print(f"         Run: make train")
            continue
        print(f"\n=== Evaluating {name} ===")
        model = load_artifact(path)
        result = evaluate_model(model, relevant_by_user, bundle.test, args.k, args.sample)
        rows.append(result)
        print(f"  {result}")

    if not rows:
        print("\nNo models evaluated. Train first:  make train")
        sys.exit(1)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(ARTIFACTS_DIR / "results.csv", index=False)
    md = to_markdown(rows)
    (ARTIFACTS_DIR / "results.md").write_text(md)
    print("\n=== Results ===")
    print(md)
    print(f"\nSaved to {ARTIFACTS_DIR / 'results.csv'} and results.md")


if __name__ == "__main__":
    main()
