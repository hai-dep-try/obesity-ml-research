"""Data loading and preprocessing (Maulana et al. 2024, Table 2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "ObesityDataSet_raw_and_data_sinthetic.csv"
TARGET_COL = "NObeyesdad"
RANDOM_STATE = 42
TEST_SIZE = 0.2

ONEHOT_COLS = ["Gender", "CAEC", "CALC", "MTRANS"]
BINARY_COLS = ["FAVC", "SCC", "SMOKE", "family_history_with_overweight"]
CONTINUOUS_COLS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_names: list[str]
    class_names: list[str]


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else DEFAULT_CSV
    if not path.exists():
        alias = DATA_DIR / "obesity.csv"
        if alias.exists():
            path = alias
        else:
            raise FileNotFoundError(
                f"Dataset not found at {path}. Run: python -m src.download_data"
            )
    return pd.read_csv(path)


def _encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in BINARY_COLS:
        if col not in out.columns:
            continue
        out[col] = (
            out[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0})
            .astype("float64")
        )
    return out


def build_preprocessor() -> ColumnTransformer:
    numeric_cols = CONTINUOUS_COLS + BINARY_COLS
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, ONEHOT_COLS),
        ],
        remainder="drop",
    )


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    missing_frac = work.isna().mean(axis=1)
    work = work.loc[missing_frac <= 0.05].reset_index(drop=True)
    work = _encode_binary(work)
    y = work[TARGET_COL].astype(str)
    feature_cols = CONTINUOUS_COLS + BINARY_COLS + ONEHOT_COLS
    X = work[feature_cols].copy()
    # sklearn+joblib on Windows mishandles pandas StringDtype in GridSearchCV workers
    for col in ONEHOT_COLS:
        if col in X.columns:
            X[col] = X[col].astype(object)
    return X, y


def load_prepared(path: Path | str | None = None) -> PreparedData:
    """Load → encode → stratified 80:20. Fits preprocessor on train only."""
    df = load_raw(path)
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    pre = build_preprocessor()
    pre.fit(X_train)
    feature_names = list(pre.get_feature_names_out())
    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=pre,
        feature_names=feature_names,
        class_names=sorted(y.unique()),
    )


def load_and_split(path: Path | str | None = None):
    data = load_prepared(path)
    return data.X_train, data.X_test, data.y_train, data.y_test, data.preprocessor


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    missing_frac = work.isna().mean(axis=1)
    work = work.loc[missing_frac <= 0.05].reset_index(drop=True)
    return _encode_binary(work)


if __name__ == "__main__":
    print(f"Expect data under: {DATA_DIR}")
    if DEFAULT_CSV.exists() or (DATA_DIR / "obesity.csv").exists():
        data = load_prepared()
        print(f"Train: {len(data.X_train)} | Test: {len(data.X_test)}")
        print(f"Raw features: {data.X_train.shape[1]} | After encode: {len(data.feature_names)}")
        print("Class distribution (train):")
        print(data.y_train.value_counts().to_string())
    else:
        print("CSV missing — run: python -m src.download_data")
