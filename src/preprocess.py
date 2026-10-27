"""Data loading and preprocessing for obesity classification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw dataset. Raise a clear error if the file is missing."""
    path = Path(path) if path else DATA_DIR / "obesity.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place or link data under data/ first."
        )
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode features. Fill in once the schema is known."""
    return df.copy()


if __name__ == "__main__":
    print(f"Expect data under: {DATA_DIR}")
