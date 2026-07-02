"""Download and extract the MovieLens 25M dataset from GroupLens.

Usage:
    python -m src.data.download

The dataset (~250MB compressed, ~1.5GB extracted) is placed in data/raw/ml-25m/.
Subsequent runs skip the download if files already exist and pass an integrity check.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile

from src.utils.config import (
    DATA_DIR,
    DATASET_SIZE,
    DIR_NAME,
    MOVIELENS_DIR,
    MOVIELENS_URL,
    MOVIELENS_ZIP,
    RAW_DIR,
)

EXPECTED_FILES = [
    "ratings.csv",
    "movies.csv",
    "tags.csv",
    "links.csv",
]
if DATASET_SIZE == "large":
    EXPECTED_FILES += ["genome-scores.csv", "genome-tags.csv", "README.txt"]


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(url, dest):
    """Stream-download `url` to `dest` with a simple byte counter."""
    print(f"Downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "recsys-engine/0.1"})
    with urllib.request.urlopen(req) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        last_report = 0
        while True:
            chunk = response.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            pct = (downloaded / total * 100) if total else 0
            if downloaded - last_report >= 50 * 1024 * 1024 or downloaded == total:
                print(f"  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MiB ({pct:.1f}%)")
                last_report = downloaded
    print(f"Download complete: {dest}")


def _extract(zip_path, dest_dir):
    print(f"Extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    print("Extraction complete.")


def _verify_extracted(dest_dir):
    missing = [name for name in EXPECTED_FILES if not (dest_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing expected files after extraction: {missing}")
    print(f"Verified {len(EXPECTED_FILES)} expected files in {dest_dir}.")


def download_movielens(force: bool = False) -> None:
    """Download and extract MovieLens 25M. No-op if already present and valid."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if MOVIELENS_DIR.exists() and not force:
        try:
            _verify_extracted(MOVIELENS_DIR)
            print("MovieLens 25M already present and verified. Skipping download.")
            return
        except RuntimeError:
            print("Existing extraction is incomplete. Re-downloading.")

    if not MOVIELENS_ZIP.exists() or force:
        _download_with_progress(MOVIELENS_URL, MOVIELENS_ZIP)

    if not MOVIELENS_DIR.exists():
        _extract(MOVIELENS_ZIP, RAW_DIR)

    _verify_extracted(MOVIELENS_DIR)

    actual_md5 = _md5(MOVIELENS_ZIP)
    print(f"zip md5: {actual_md5}")

    print("\nDone. Next step:  make data   (build processed splits)")


if __name__ == "__main__":
    download_movielens(force="--force" in sys.argv)
