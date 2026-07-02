"""Configuration and path constants for recsys-engine."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Set to "small" or "large" via env var; default small for quick iteration.
DATASET_SIZE = os.environ.get("ML_SIZE", "small")

if DATASET_SIZE == "large":
    MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    DIR_NAME = "ml-25m"
else:
    MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    DIR_NAME = "ml-latest-small"

MOVIELENS_ZIP = RAW_DIR / f"{DIR_NAME}.zip"
MOVIELENS_DIR = RAW_DIR / DIR_NAME

RATINGS_PATH = MOVIELENS_DIR / "ratings.csv"
MOVIES_PATH = MOVIELENS_DIR / "movies.csv"
TAGS_PATH = MOVIELENS_DIR / "tags.csv"
LINKS_PATH = MOVIELENS_DIR / "links.csv"

# ml-25m has genome files; ml-latest-small does not.
GENOME_SCORES_PATH = MOVIELENS_DIR / "genome-scores.csv"
GENOME_TAGS_PATH = MOVIELENS_DIR / "genome-tags.csv"

RANDOM_SEED = 42
IMPLICIT_RATING_THRESHOLD = 4.0
TEST_FRACTION = 0.2
