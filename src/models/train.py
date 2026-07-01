"""Train/load baseline + SVD + NCF models and persist artifacts.

NCF is trained in `notebooks/03_ncf_training.ipynb` on Google Colab GPU.
Locally we only LOAD the pretrained `.pth` checkpoint (placed at
`artifacts/ncf.pth` after the Colab run). Popularity and SVD train on CPU.

Usage:
    make train                              # train popularity + svd
    python -m src.models.train --models popularity svd
    python -m src.models.train --models ncf --load artifacts/ncf.pth
"""
from __future__ import annotations

import argparse
import sys
import time

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


def load_ncf(path) -> float:
    """Load a pretrained NCF checkpoint (.pth) instead of training locally."""
    from src.models.ncf import NCFRecommender

    print(f"\n=== Loading NCF from {path} ===")
    t0 = time.time()
    model = NCFRecommender()
    model.load(path)
    elapsed = time.time() - t0
    save_artifact(model, ARTIFACTS_DIR / "ncf.joblib")
    print(f"Loaded NCF in {elapsed:.1f}s -> {ARTIFACTS_DIR / 'ncf.joblib'}")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS), help="CPU-trainable models"
    )
    parser.add_argument(
        "--load-ncf", metavar="PATH", help="Load a pretrained NCF .pth checkpoint"
    )
    args = parser.parse_args()

    if args.load_ncf:
        load_ncf(args.load_ncf)
        return

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
    print("\nNext:  make evaluate   (or load NCF:  python -m src.models.train --load-ncf artifacts/ncf.pth)")


if __name__ == "__main__":
    sys.exit(main())
