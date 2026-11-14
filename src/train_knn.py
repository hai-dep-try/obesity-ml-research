"""Huan luyen va danh gia mo hinh KNN tren bo du lieu beo phi (CPU).

Phien ban co so (baseline) su dung pipeline tien xu ly chung cua du an.
"""

from __future__ import annotations

import json

import numpy as np
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

# Cac tham so mac dinh cho mo hinh co so
BASELINE_PARAMS = {
    "n_neighbors": 5,
    "weights": "uniform",
    "metric": "minkowski",
    "p": 2,
}


def train_knn() -> dict:
    """Huan luyen KNN co so (k=5) va danh gia tren tap kiem thu."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = load_prepared()

    # Ap dung pipeline chuan hoa StandardScaler va OneHotEncoder
    X_train = np.asarray(data.preprocessor.transform(data.X_train))
    X_test = np.asarray(data.preprocessor.transform(data.X_test))
    y_train, y_test = data.y_train, data.y_test
    labels = sorted(y_test.unique())

    print(f"Tap huan luyen={len(y_train)} | Tap kiem thu={len(y_test)} | So dac trung={X_train.shape[1]}")
    print("=== Mo hinh co so KNN (Baseline) ===")

    model = KNeighborsClassifier(**BASELINE_PARAMS, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # Tinh toan cac chi so danh gia (Accuracy, Precision, Recall, F1)
    metrics = compute_metrics(y_test, pred)
    row = metrics_to_percent_row("KNN_Baseline", metrics)
    row["params"] = json.dumps(BASELINE_PARAMS)
    row["paper_accuracy_pct"] = PAPER_KNN_ACC
    row["delta_accuracy_pp"] = round(row["accuracy_pct"] - PAPER_KNN_ACC, 2)

    print(
        f"  KNN Baseline: Acc={row['accuracy_pct']}% | "
        f"F1={row['f1_pct']}% | "
        f"So mau doan sai={int((pred != y_test).sum())}/{len(y_test)} | "
        f"Bai bao goc={PAPER_KNN_ACC}% | Chenh lech={row['delta_accuracy_pp']}pp"
    )

    print("\n=== Bao cao chi tiet phan lop (Classification Report) ===")
    print_classification_report(y_test, pred, labels=labels)

    # Luu ket qua va bieu do
    out_csv = save_metrics_table([row], "knn_results.csv")
    fig_path = plot_confusion_matrix(
        y_test,
        pred,
        labels=labels,
        title="Ma tran nham lan — KNN Baseline (Tap kiem thu)",
        filename="confusion_matrix_knn.png",
    )

    print(f"\nDa luu ket qua: {out_csv}")
    print(f"Da luu bieu do: {fig_path}")
    print(f"Do chinh xac bai bao: {PAPER_KNN_ACC}% | Cua nhom: {row['accuracy_pct']}%")

    return {"row": row, "metrics": metrics}


if __name__ == "__main__":
    train_knn()
