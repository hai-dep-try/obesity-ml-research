"""Huan luyen va danh gia mo hinh KNN tren bo du lieu beo phi (CPU).

Quy trinh huan luyen chuyen sau tuong tu train_xgboost.py:
1. Chay mo hinh co so (baseline k=5)
2. Tinh chinh sieu tham so bang GridSearchCV (Stratified 5-Fold)
3. So sanh voi bai bao khoa hoc goc (Maulana et al. 2024, KNN dat 88.18%)
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from src.evaluate import (
    FIGURES_DIR,
    RESULTS_DIR,
    compute_metrics,
    metrics_to_percent_row,
    plot_confusion_matrix,
    print_classification_report,
    save_metrics_table,
)
from src.preprocess import RANDOM_STATE, load_prepared

# Do chinh xac tham chieu tu bai bao goc (Maulana et al. 2024, Bang 4)
PAPER_KNN_ACC = 88.18

# Tham so mac dinh cho mo hinh co so
BASELINE_PARAMS = {
    "n_neighbors": 5,
    "weights": "uniform",
    "metric": "minkowski",
    "p": 2,
}


def train_knn(tune: bool = True) -> dict:
    """Huan luyen KNN: gom mo hinh co so va tuy chon tinh chinh bang GridSearchCV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = load_prepared()

    X_train = np.asarray(data.preprocessor.transform(data.X_train))
    X_test = np.asarray(data.preprocessor.transform(data.X_test))
    y_train, y_test = data.y_train, data.y_test
    labels = sorted(y_test.unique())

    rows: list[dict] = []

    # ── 1) Mo hinh co so KNN (k=5, khoang cach euclidean) ──────────────────
    print(f"Tap huan luyen={len(y_train)} | Tap kiem thu={len(y_test)} | So dac trung={X_train.shape[1]}")
    print("=== Mo hinh co so KNN (Baseline) ===")

    base_model = KNeighborsClassifier(**BASELINE_PARAMS, n_jobs=-1)
    base_model.fit(X_train, y_train)
    base_pred = base_model.predict(X_test)
    base_metrics = compute_metrics(y_test, base_pred)
    base_row = metrics_to_percent_row("KNN_Baseline", base_metrics)
    base_row["params"] = json.dumps(BASELINE_PARAMS)
    base_row["cv_accuracy_pct"] = None
    base_row["paper_accuracy_pct"] = PAPER_KNN_ACC
    base_row["delta_accuracy_pp"] = round(
        base_row["accuracy_pct"] - PAPER_KNN_ACC, 2
    )
    rows.append(base_row)
    print(
        f"  KNN Baseline: Acc={base_row['accuracy_pct']}% | "
        f"F1={base_row['f1_pct']}% | "
        f"So mau sai={int((base_pred != y_test).sum())}/{len(y_test)} | "
        f"Bai bao={PAPER_KNN_ACC}% | Chenh lech={base_row['delta_accuracy_pp']}pp"
    )

    best_model = base_model
    best_pred = base_pred
    best_params = dict(BASELINE_PARAMS)
    best_metrics = base_metrics

    # ── 2) Tinh chinh sieu tham so bang GridSearchCV ───────────────────────
    if tune:
        print("\n=== Tinh chinh sieu tham so (GridSearchCV voi Stratified 5-Fold) ===")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        param_grid = {
            "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
            "p": [1, 2, 3],
        }

        search = GridSearchCV(
            KNeighborsClassifier(n_jobs=-1),
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
            refit=True,
            verbose=1,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_pred = best_model.predict(X_test)
        best_params = dict(search.best_params_)
        best_metrics = compute_metrics(y_test, best_pred)

        tuned_row = metrics_to_percent_row("KNN_Tuned", best_metrics)
        tuned_row["params"] = json.dumps(best_params)
        tuned_row["cv_accuracy_pct"] = round(float(search.best_score_) * 100, 2)
        tuned_row["paper_accuracy_pct"] = PAPER_KNN_ACC
        tuned_row["delta_accuracy_pp"] = round(
            tuned_row["accuracy_pct"] - PAPER_KNN_ACC, 2
        )
        rows.append(tuned_row)
        print(
            f"  KNN Tuned (CV={tuned_row['cv_accuracy_pct']}%): "
            f"Acc={tuned_row['accuracy_pct']}% | F1={tuned_row['f1_pct']}% | "
            f"So mau sai={int((best_pred != y_test).sum())}/{len(y_test)} | "
            f"Bai bao={PAPER_KNN_ACC}% | Chenh lech={tuned_row['delta_accuracy_pp']}pp\n"
            f"  Tham so tot nhat: {best_params}"
        )

    # ── 3) Bao cao phan lop ────────────────────────────────────────────────
    print("\n=== Bao cao chi tiet phan lop — KNN tot nhat ===")
    report = print_classification_report(y_test, best_pred, labels=labels)

    # Luu bao cao phan lop
    report_path = RESULTS_DIR / "knn_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    # ── 4) Xuat file ket qua ───────────────────────────────────────────────
    out_csv = save_metrics_table(rows, "knn_results.csv")

    fig_path = plot_confusion_matrix(
        y_test,
        best_pred,
        labels=labels,
        title="Ma tran nham lan — KNN Toi uu (Tap kiem thu)",
        filename="confusion_matrix_knn.png",
    )

    # Luu thong so sieu tham so tot nhat ra json
    (RESULTS_DIR / "knn_best_params.json").write_text(
        json.dumps(best_params, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"\nDa luu ket qua: {out_csv}")
    print(f"Da luu bieu do: {fig_path}")
    print(f"Do chinh xac bai bao: {PAPER_KNN_ACC}% | Cua nhom: {rows[-1]['accuracy_pct']}%")

    return {"rows": rows, "best_params": best_params, "best_metrics": best_metrics}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Huan luyen KNN tren bo du lieu beo phi")
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Bo qua GridSearchCV va chi chay KNN co so k=5",
    )
    args = parser.parse_args()
    train_knn(tune=not args.no_tune)
