"""Train baseline + SVD models and persist artifacts.

NCF is trained separately in `notebooks/03_ncf_training.ipynb` on Google Colab
(GPU). This entrypoint covers the CPU-friendly models only.

Usage:
    make train
    python -m src.models.train            # train all
    python -m src.models.train --models popularity svd
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from src.data.loader import load_data
from src.models.baseline import PopularityRecommender
from src.models.svd import SVDRecommender
from src.utils.config import ARTIFACTS_DIR
from src.utils.io import save_artifact

MODELS = {
    "popularity": lambda: PopularityRecommender(),
    "svd": lambda: SVDRecommender(),
}


def train_one(name: str, bundle) -> float:
    factory = MODELS[name]
    model = factory()
    print(f"\n=== Training {name} ===")
    t0 = time.time()
    if name == "popularity":
        model.fit(bundle.train, movies=bundle.movies)
    else:
        model.fit(bundle.train)
    elapsed = time.time() - t0
    save_artifact(model, ARTIFACTS_DIR / f"{name}.joblib")
    print(f"Trained {name} in {elapsed:.1f}s -> {ARTIFACTS_DIR / (name + '.joblib')}")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all", action="store_true", help="Train all models (default if no --models)"
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS), help="Subset of models to train"
    )
    args = parser.parse_args()

    names = args.models if args.models else list(MODELS)
    print(f"Models to train: {names}")

    bundle = load_data()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    for name in names:
        timings[name] = train_one(name, bundle)

    print("\n=== Training summary ===")
    for name, secs in timings.items():
        print(f"  {name:12s} {secs:7.1f}s")
    print("\nNext:  make evaluate")


if __name__ == "__main__":
    sys.exit(main())
