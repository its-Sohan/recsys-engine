"""Train/load all models and persist artifacts.

Usage:
    make train                              # train popularity + svd + content
    python -m src.models.train --models popularity svd content
    python -m src.models.train --load-ncf artifacts/ncf.pth   # then:
    python -m src.models.train --build-hybrid                  # build hybrid from ncf+content
    make evaluate
"""
from __future__ import annotations

import argparse
import sys
import time

from src.data.loader import load_data
from src.models.baseline import PopularityRecommender
from src.models.content import ContentRecommender
from src.models.svd import SVDRecommender
from src.utils.config import ARTIFACTS_DIR
from src.utils.io import load_artifact, save_artifact

MODELS = {
    "popularity": lambda: PopularityRecommender(),
    "content": lambda: ContentRecommender(),
    "svd": lambda: SVDRecommender(),
}


def train_one(name: str, bundle) -> float:
    factory = MODELS[name]
    model = factory()
    print(f"\n=== Training {name} ===")
    t0 = time.time()
    if name == "popularity":
        model.fit(bundle.train, movies=bundle.movies)
    elif name == "content":
        model.fit(bundle.train, movies=bundle.movies, tags=bundle.tags)
    else:
        model.fit(bundle.train)
    elapsed = time.time() - t0
    save_artifact(model, ARTIFACTS_DIR / f"{name}.joblib")
    print(f"Trained {name} in {elapsed:.1f}s -> {ARTIFACTS_DIR / (name + '.joblib')}")
    return elapsed


def load_ncf(path) -> float:
    from src.models.ncf import NCFRecommender
    print(f"\n=== Loading NCF from {path} ===")
    t0 = time.time()
    model = NCFRecommender()
    model.load(path)
    elapsed = time.time() - t0
    save_artifact(model, ARTIFACTS_DIR / "ncf.joblib")
    print(f"Loaded NCF in {elapsed:.1f}s -> {ARTIFACTS_DIR / 'ncf.joblib'}")
    return elapsed


def build_hybrid() -> None:
    from src.models.hybrid import HybridRecommender
    ncf = load_artifact(ARTIFACTS_DIR / "ncf.joblib")
    content = load_artifact(ARTIFACTS_DIR / "content.joblib")
    print("\n=== Building hybrid from ncf + content ===")
    model = HybridRecommender(ncf_model=ncf, content_model=content, alpha=0.7)
    save_artifact(model, ARTIFACTS_DIR / "hybrid.joblib")
    print(f"Built hybrid -> {ARTIFACTS_DIR / 'hybrid.joblib'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="CPU-trainable models")
    parser.add_argument("--load-ncf", metavar="PATH", help="Load a pretrained NCF .pth")
    parser.add_argument("--build-hybrid", action="store_true", help="Build hybrid from ncf + content artifacts")
    args = parser.parse_args()

    if args.load_ncf:
        load_ncf(args.load_ncf)
        return

    if args.build_hybrid:
        build_hybrid()
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

    print("\nNext steps:")
    print("  make evaluate")
    print("  python -m src.models.train --load-ncf artifacts/ncf.pth")
    print("  python -m src.models.train --build-hybrid")


if __name__ == "__main__":
    sys.exit(main())
