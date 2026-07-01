"""Configuration and path constants for recsys-engine."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
MOVIELENS_ZIP = RAW_DIR / "ml-25m.zip"
MOVIELENS_DIR = RAW_DIR / "ml-25m"

RATINGS_PATH = MOVIELENS_DIR / "ratings.csv"
MOVIES_PATH = MOVIELENS_DIR / "movies.csv"
TAGS_PATH = MOVIELENS_DIR / "tags.csv"
GENOME_SCORES_PATH = MOVIELENS_DIR / "genome-scores.csv"
GENOME_TAGS_PATH = MOVIELENS_DIR / "genome-tags.csv"
LINKS_PATH = MOVIELENS_DIR / "links.csv"

RANDOM_SEED = 42
IMPLICIT_RATING_THRESHOLD = 4.0
TEST_FRACTION = 0.2
