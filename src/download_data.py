"""Download the Obesity Levels dataset into data/ via kagglehub."""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

DATASET = "fatemehmehrparvar/obesity-levels"
FILENAME = "ObesityDataSet_raw_and_data_sinthetic.csv"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def main() -> Path:
    cache_dir = Path(kagglehub.dataset_download(DATASET))
    src = cache_dir / FILENAME
    if not src.exists():
        raise FileNotFoundError(f"Expected {FILENAME} under {cache_dir}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / FILENAME
    alias = DATA_DIR / "obesity.csv"
    shutil.copy2(src, dest)
    shutil.copy2(src, alias)
    print(f"Cached at: {cache_dir}")
    print(f"Copied to: {dest}")
    print(f"Alias: {alias}")
    return dest


if __name__ == "__main__":
    main()
