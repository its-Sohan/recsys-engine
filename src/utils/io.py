"""Tiny artifact I/O helpers (joblib persistence)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def save_artifact(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: Path) -> Any:
    return joblib.load(path)
