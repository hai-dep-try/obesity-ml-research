"""Train and evaluate XGBoost on the standardized obesity pipeline (CPU).

Integrates HoTuAn's XGBoost experiment into the shared preprocessing and
evaluation framework (src/preprocess.py, src/evaluate.py) for an exact
apples-to-apples comparison against CatBoost and the paper baselines.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.evaluate import (
    RESULTS_DIR,
    compute_metrics,
    metrics_to_percent_row,
    plot_confusion_matrix,
    print_classification_report,
    save_metrics_table,
)
from src.preprocess import RANDOM_STATE, load_prepared

HOTUAN_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
}


def train_xgboost(tune: bool = True) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_prepared()

    X_train = np.asarray(data.preprocessor.transform(data.X_train))
    X_test = np.asarray(data.preprocessor.transform(data.X_test))

    le = LabelEncoder().fit(sorted(data.y_train.unique()))
    y_train = le.transform(data.y_train)
    y_test = le.transform(data.y_test)
    labels = list(le.classes_)

    rows: list[dict] = []

    # 1) HoTuAn baseline configuration on standardized split
    base_model = XGBClassifier(
        **HOTUAN_PARAMS,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    base_model.fit(X_train, y_train)
    base_pred = base_model.predict(X_test)
    base_metrics = compute_metrics(y_test, base_pred)
    base_row = metrics_to_percent_row("XGBoost_HoTuAn", base_metrics)
    base_row["params"] = json.dumps(HOTUAN_PARAMS)
    base_row["cv_accuracy_pct"] = None
    rows.append(base_row)
    print(
        f"XGBoost (HoTuAn params): Acc={base_row['accuracy_pct']}% | "
        f"F1={base_row['f1_pct']}% | Errors={int((base_pred != y_test).sum())}/{len(y_test)}"
    )

    best_model = base_model
    best_pred = base_pred
    best_params = dict(HOTUAN_PARAMS)

    # 2) Stratified 5-Fold GridSearchCV tuning (consistent with baselines)
    if tune:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6, 8],
        }
        search = GridSearchCV(
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="mlogloss",
                n_jobs=-1,
            ),
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_pred = best_model.predict(X_test)
        best_params = dict(search.best_params_)
        tuned_metrics = compute_metrics(y_test, best_pred)
        tuned_row = metrics_to_percent_row("XGBoost_Tuned", tuned_metrics)
        tuned_row["params"] = json.dumps(best_params)
        tuned_row["cv_accuracy_pct"] = round(float(search.best_score_) * 100, 2)
        rows.append(tuned_row)
        print(
            f"XGBoost (Tuned CV={tuned_row['cv_accuracy_pct']}%): "
            f"Acc={tuned_row['accuracy_pct']}% | F1={tuned_row['f1_pct']}% | "
            f"Errors={int((best_pred != y_test).sum())}/{len(y_test)} | params={best_params}"
        )

    print("\n=== Classification Report — Best XGBoost ===")
    print_classification_report(
        le.inverse_transform(y_test),
        le.inverse_transform(best_pred),
        labels=labels,
    )

    out_csv = save_metrics_table(rows, "xgboost_results.csv")
    fig_path = plot_confusion_matrix(
        le.inverse_transform(y_test),
        le.inverse_transform(best_pred),
        labels=labels,
        title="Confusion matrix — XGBoost Tuned (test)",
        filename="confusion_matrix_xgboost.png",
    )
    print(f"Saved: {out_csv} and {fig_path}")
    return {"rows": rows, "best_params": best_params}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost on obesity dataset")
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip GridSearchCV and only run HoTuAn's default parameters",
    )
    args = parser.parse_args()
    train_xgboost(tune=not args.no_tune)
