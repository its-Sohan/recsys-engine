"""Build processed data artifacts (time-based split, parquet dumps).

Run via:  make data   (or)   python -m src.data.build
"""
from __future__ import annotations

from src.data.loader import load_data


def main() -> None:
    bundle = load_data()
    print(
        f"users: {bundle.n_users:,}  items: {bundle.n_items:,}  "
        f"train: {len(bundle.train):,}  test: {len(bundle.test):,}"
    )
    print("Done. Next steps:  make train   |   make evaluate")


if __name__ == "__main__":
    main()
