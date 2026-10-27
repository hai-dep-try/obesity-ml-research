"""Train baselines + CatBoost (Maulana et al. 2024) and write results/."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from src.evaluate import (
    FIGURES_DIR,
    RESULTS_DIR,
    compute_metrics,
    metrics_to_percent_row,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    print_classification_report,
    save_metrics_table,
)
from src.preprocess import RANDOM_STATE, load_prepared

CATBOOST_BEST = {
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 5,
    "iterations": 300,
}

PAPER_ACC = {
    "CatBoost": 95.98,
    "LogisticRegression": 65.48,
    "KNN": 88.18,
    "RandomForest": 95.04,
    "NaiveBayes": 94.07,
}

PAPER_TABLE4 = [
    {"model": "CatBoost", "accuracy": 95.98, "precision": 96.08, "recall": 95.98, "f1": 96.00},
    {"model": "LogisticRegression", "accuracy": 65.48, "precision": 68.60, "recall": 65.48, "f1": 66.60},
    {"model": "KNN", "accuracy": 88.18, "precision": 91.25, "recall": 88.18, "f1": 89.07},
    {"model": "RandomForest", "accuracy": 95.04, "precision": 95.04, "recall": 95.04, "f1": 95.00},
    {"model": "NaiveBayes", "accuracy": 94.07, "precision": 72.13, "recall": 64.07, "f1": 66.17},
]

BASELINE_GRIDS: dict[str, tuple] = {
    "LogisticRegression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs"]},
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        },
    ),
    "NaiveBayes": (
        GaussianNB(),
        {"var_smoothing": [1e-9, 1e-8, 1e-7]},
    ),
}


def _cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def tune_baseline(name: str, X_train, y_train):
    estimator, param_grid = BASELINE_GRIDS[name]
    search = GridSearchCV(
        estimator,
        param_grid,
        cv=_cv(),
        scoring="accuracy",
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def train_catboost(X_train, y_train, tune: bool = False):
    if tune:
        param_grid = {
            "learning_rate": [0.05],
            "depth": [6, 8],
            "l2_leaf_reg": [5],
            "iterations": [300],
        }
        base = CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=-1,
            allow_writing_files=False,
        )
        search = GridSearchCV(
            base,
            param_grid,
            cv=_cv(),
            scoring="accuracy",
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_, float(search.best_score_)

    model = CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=False,
        thread_count=-1,
        allow_writing_files=False,
        **CATBOOST_BEST,
    )
    model.fit(X_train, y_train)
    return model, dict(CATBOOST_BEST), None


def train(tune_catboost: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = load_prepared()
    # Fit on train only (done in load_prepared); transform both splits
    X_train = np.asarray(data.preprocessor.transform(data.X_train))
    X_test = np.asarray(data.preprocessor.transform(data.X_test))
    y_train, y_test = data.y_train, data.y_test
    feature_names = data.feature_names

    baseline_rows: list[dict] = []
    all_rows: list[dict] = []
    best_params: dict = {}

    print(f"Train={len(y_train)} Test={len(y_test)} features={X_train.shape[1]}")
    print("=== Baselines (GridSearchCV, stratified 5-fold) ===")
    for name in BASELINE_GRIDS:
        print(f"Tuning {name}...")
        model, params, cv_score = tune_baseline(name, X_train, y_train)
        pred = model.predict(X_test)
        metrics = compute_metrics(y_test, pred)
        row = metrics_to_percent_row(name, metrics)
        row["cv_accuracy_pct"] = round(cv_score * 100, 2)
        row["paper_accuracy_pct"] = PAPER_ACC.get(name)
        if row["paper_accuracy_pct"] is not None:
            row["delta_accuracy_pp"] = round(
                row["accuracy_pct"] - row["paper_accuracy_pct"], 2
            )
        baseline_rows.append(row)
        all_rows.append(row)
        best_params[name] = params
        print(f"  {name}: test acc={row['accuracy_pct']}%  best_params={params}")

    save_metrics_table(baseline_rows, "baseline_results.csv")

    print("=== CatBoost (main) ===")
    cat_model, cat_params, cat_cv = train_catboost(
        X_train, y_train, tune=tune_catboost
    )
    cat_pred = cat_model.predict(X_test)
    if hasattr(cat_pred, "shape") and len(getattr(cat_pred, "shape", ())) == 2:
        cat_pred = np.asarray(cat_pred).ravel()
    cat_metrics = compute_metrics(y_test, cat_pred)
    cat_row = metrics_to_percent_row("CatBoost", cat_metrics)
    if cat_cv is not None:
        cat_row["cv_accuracy_pct"] = round(cat_cv * 100, 2)
    cat_row["paper_accuracy_pct"] = PAPER_ACC["CatBoost"]
    cat_row["delta_accuracy_pp"] = round(
        cat_row["accuracy_pct"] - PAPER_ACC["CatBoost"], 2
    )
    all_rows.append(cat_row)
    best_params["CatBoost"] = cat_params
    print(f"  CatBoost: test acc={cat_row['accuracy_pct']}%  params={cat_params}")
    print_classification_report(y_test, cat_pred)

    save_metrics_table([cat_row], "main_results.csv")
    save_metrics_table(all_rows, "comparison_results.csv")
    save_metrics_table(all_rows, "all_results.csv")
    save_metrics_table(all_rows, "all_models_results.csv")

    paper_ref = pd.DataFrame(PAPER_TABLE4)
    paper_ref.to_csv(RESULTS_DIR / "paper_table4_reference.csv", index=False)

    # Side-by-side for notebooks / report
    ours = {r["model"]: r for r in all_rows}
    compare_rows = []
    for pref in PAPER_TABLE4:
        name = pref["model"]
        o = ours.get(name, {})
        compare_rows.append(
            {
                "model": name,
                "paper_accuracy_pct": pref["accuracy"],
                "repo_accuracy_pct": o.get("accuracy_pct"),
                "paper_f1": pref["f1"],
                "repo_f1_pct": o.get("f1_pct"),
                "delta_accuracy_pp": o.get("delta_accuracy_pp"),
            }
        )
    save_metrics_table(compare_rows, "paper_comparison.csv")

    labels = sorted(y_test.unique())
    plot_confusion_matrix(
        y_test,
        cat_pred,
        labels=labels,
        title="Confusion matrix — CatBoost (test)",
        filename="confusion_matrix_catboost.png",
    )
    # Alias name used in some docs
    plot_confusion_matrix(
        y_test,
        cat_pred,
        labels=labels,
        title="Confusion matrix — CatBoost (test)",
        filename="catboost_confusion_matrix.png",
    )

    importances = np.asarray(cat_model.get_feature_importance(), dtype=float)
    n_feat = len(importances)
    names = list(feature_names)
    if len(names) != n_feat:
        names = [f"f{i}" for i in range(n_feat)]
    plot_feature_importance(
        names,
        importances,
        title="Feature importance — CatBoost",
        filename="feature_importance_catboost.png",
    )
    plot_feature_importance(
        names,
        importances,
        title="Feature importance — CatBoost",
        filename="catboost_feature_importance.png",
    )
    plot_model_comparison(all_rows, filename="model_comparison.png")

    (RESULTS_DIR / "best_params.json").write_text(
        json.dumps(best_params, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (RESULTS_DIR / "paper_comparison.json").write_text(
        json.dumps(
            {
                "paper": "Maulana et al. (2024) Predicting Obesity Levels with CatBoost",
                "paper_catboost_accuracy_pct": 95.98,
                "ours_catboost_accuracy_pct": cat_row["accuracy_pct"],
                "delta_pp": cat_row["delta_accuracy_pp"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote metrics under {RESULTS_DIR}")
    print(f"Paper CatBoost acc: 95.98% | Ours: {cat_row['accuracy_pct']}%")
    return {"all_rows": all_rows, "best_params": best_params, "cat_metrics": cat_metrics}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train obesity classifiers (CPU)")
    parser.add_argument(
        "--tune-catboost",
        action="store_true",
        help="Small CatBoost grid; default uses paper best params",
    )
    args = parser.parse_args()
    train(tune_catboost=args.tune_catboost)
