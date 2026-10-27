"""Data loading and preprocessing for obesity classification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "ObesityDataSet_raw_and_data_sinthetic.csv"
TARGET_COL = "NObeyesdad"


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw dataset. Raise a clear error if the file is missing."""
    path = Path(path) if path else DEFAULT_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run: python -m src.download_data  (see data/README.md)."
        )
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode features. Fill in once the schema is known."""
    return df.copy()


if __name__ == "__main__":
    print(f"Expect data under: {DATA_DIR}")
    if DEFAULT_CSV.exists():
        df = load_raw()
        print(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
        if TARGET_COL in df.columns:
            print(f"Target `{TARGET_COL}` classes: {df[TARGET_COL].nunique()}")
    else:
        print("No CSV yet — run: python -m src.download_data")
