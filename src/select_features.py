"""Generate 5-feature selection metadata and filtered CSV splits (Yagin et al., 2023).

Creates:
1. data/splits/top5_features.json (consumed by src/lr_baseline.py & src/bayes_search.py)
2. data/splits/top5_{chi2,f_classify,mutual_info}/{train,test,train_for_val,test_for_val}.csv
3. results/feature_selection_scores.csv (comparison of SelectKBest scores on train_for_val vs paper)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT_DIR / "data" / "ObesityDataSet_raw_and_data_sinthetic.csv"
SPLIT_DIR = ROOT_DIR / "data" / "splits"
RESULTS_DIR = ROOT_DIR / "results"

TARGET = "NObeyesdad"
ALL16 = [
    "Gender",
    "Age",
    "Height",
    "Weight",
    "family_history_with_overweight",
    "FAVC",
    "FCVC",
    "NCP",
    "CAEC",
    "SMOKE",
    "CH2O",
    "SCC",
    "FAF",
    "TUE",
    "CALC",
    "MTRANS",
]

ORD = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
MAPS = {
    "Gender": {"Female": 0, "Male": 1},
    "family_history_with_overweight": {"no": 0, "yes": 1},
    "FAVC": {"no": 0, "yes": 1},
    "SCC": {"no": 0, "yes": 1},
    "SMOKE": {"no": 0, "yes": 1},
    "CAEC": ORD,
    "CALC": ORD,
    "MTRANS": {
        "Walking": 0,
        "Bike": 1,
        "Motorbike": 2,
        "Public_Transportation": 3,
        "Automobile": 4,
    },
}

# Exact 5-feature sets reported in Yagin et al. (2023), Applied Sciences 13(6):3875, p. 9
PAPER_TOP5 = {
    "chi2": ["Weight", "FCVC", "CALC", "FAVC", "FAF"],
    "f_classify": ["FAF", "CALC", "TUE", "FAVC", "FCVC"],
    "mutual_info": ["Weight", "FAF", "SMOKE", "FAVC", "CALC"],
}


def ensure_base_splits() -> None:
    needed = ["train.csv", "test.csv", "train_for_val.csv", "test_for_val.csv"]
    if all((SPLIT_DIR / f).exists() for f in needed):
        return
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_CSV)
    train, test = train_test_split(
        df, test_size=0.25, stratify=df[TARGET], random_state=42
    )
    tr_val, te_val = train_test_split(
        train, test_size=0.20, stratify=train[TARGET], random_state=42
    )
    for name, d in [
        ("train", train),
        ("test", test),
        ("train_for_val", tr_val),
        ("test_for_val", te_val),
    ]:
        d.to_csv(SPLIT_DIR / f"{name}.csv", index=False)


def encode_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, mp in MAPS.items():
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(str).str.strip().map(mp)
    assert out[ALL16].isna().sum().sum() == 0, "NaN sau khi encode!"
    return out


def compute_selectkbest_scores() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    trv = encode_df(pd.read_csv(SPLIT_DIR / "train_for_val.csv"))
    X = trv[ALL16]
    y = trv[TARGET]

    chi_fit = SelectKBest(chi2, k=5).fit(X, y)
    f_fit = SelectKBest(f_classif, k=5).fit(X, y)
    mi_fit = SelectKBest(
        lambda A, b: mutual_info_classif(A, b, random_state=42), k=5
    ).fit(X, y)

    scores_df = pd.DataFrame(
        {
            "feature": ALL16,
            "chi2_score": np.round(chi_fit.scores_, 4),
            "f_classify_score": np.round(f_fit.scores_, 4),
            "mutual_info_score": np.round(mi_fit.scores_, 4),
            "in_paper_chi2": [c in PAPER_TOP5["chi2"] for c in ALL16],
            "in_paper_f_classify": [c in PAPER_TOP5["f_classify"] for c in ALL16],
            "in_paper_mutual_info": [c in PAPER_TOP5["mutual_info"] for c in ALL16],
        }
    )

    kbest_top5 = {
        "chi2": [ALL16[i] for i in np.argsort(chi_fit.scores_)[::-1][:5]],
        "f_classify": [ALL16[i] for i in np.argsort(f_fit.scores_)[::-1][:5]],
        "mutual_info": [ALL16[i] for i in np.argsort(mi_fit.scores_)[::-1][:5]],
    }
    return scores_df, kbest_top5


def export_filtered_splits(selected: dict[str, list[str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for method, cols in selected.items():
        assert len(cols) == 5 and all(c in ALL16 for c in cols), f"Invalid cols for {method}: {cols}"
        out_sub = SPLIT_DIR / f"top5_{method}"
        out_sub.mkdir(parents=True, exist_ok=True)
        for split_name in ["train", "test", "train_for_val", "test_for_val"]:
            df = pd.read_csv(SPLIT_DIR / f"{split_name}.csv")
            df[cols + [TARGET]].to_csv(out_sub / f"{split_name}.csv", index=False)

    top5_json = json.dumps(selected, indent=2, ensure_ascii=False)
    (SPLIT_DIR / "top5_features.json").write_text(top5_json, encoding="utf-8")

    # Sync to src/data/splits if teammate's legacy folder exists locally
    legacy_split_dir = ROOT_DIR / "src" / "data" / "splits"
    if legacy_split_dir.exists():
        (legacy_split_dir / "top5_features.json").write_text(top5_json, encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Create 5-feature subsets (Yagin et al., 2023)")
    parser.add_argument(
        "--mode",
        choices=["paper", "selectkbest"],
        default="paper",
        help="paper: exact top-5 feature sets from Yagin et al. (2023) p.9; selectkbest: compute top-5 on train_for_val",
    )
    args = parser.parse_args()

    ensure_base_splits()
    scores_df, kbest_top5 = compute_selectkbest_scores()
    scores_df.to_csv(RESULTS_DIR / "feature_selection_scores.csv", index=False)

    selected = PAPER_TOP5 if args.mode == "paper" else kbest_top5
    export_filtered_splits(selected)

    print(f"Mode selected: {args.mode}")
    print("Saved top5_features.json:", json.dumps(selected, ensure_ascii=False))
    print("Filtered CSV splits written under:", SPLIT_DIR)
    print("\nSelectKBest (top-5 highest score on train_for_val) for reference:")
    print(json.dumps(kbest_top5, ensure_ascii=False))


if __name__ == "__main__":
    main()
