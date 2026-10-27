"""Evaluation helpers and figure export for obesity models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Weighted multiclass metrics (paper section 2.4)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate(y_true, y_pred, labels: Iterable[str] | None = None) -> dict[str, float]:
    return compute_metrics(y_true, y_pred)


def metrics_to_percent_row(model_name: str, metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "model": model_name,
        "accuracy": round(metrics["accuracy"] * 100, 2),
        "precision": round(metrics["precision"] * 100, 2),
        "recall": round(metrics["recall"] * 100, 2),
        "f1": round(metrics["f1"] * 100, 2),
        "accuracy_pct": round(metrics["accuracy"] * 100, 2),
        "precision_pct": round(metrics["precision"] * 100, 2),
        "recall_pct": round(metrics["recall"] * 100, 2),
        "f1_pct": round(metrics["f1"] * 100, 2),
    }


def save_metrics_table(rows: list[dict[str, Any]], name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / name
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def print_classification_report(
    y_true, y_pred, labels: Sequence[str] | None = None
) -> str:
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(report)
    return report


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str],
    title: str = "CatBoost — confusion matrix (test)",
    filename: str = "confusion_matrix_catboost.png",
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / Path(filename).name
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_feature_importance(
    names: Sequence[str],
    importances: np.ndarray | Sequence[float],
    title: str = "CatBoost — feature importance",
    filename: str = "feature_importance_catboost.png",
    top_n: int = 15,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / Path(filename).name
    importances = np.asarray(importances, dtype=float)
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [names[i] for i in order][::-1],
        importances[order][::-1],
        color="#2a6f97",
    )
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # Persist CSV (canonical + alias used in notebooks)
    fi_df = pd.DataFrame({"feature": list(names), "importance": importances}).sort_values(
        "importance", ascending=False
    )
    fi_df.to_csv(RESULTS_DIR / "feature_importance_catboost.csv", index=False)
    fi_df.to_csv(RESULTS_DIR / "catboost_feature_importance.csv", index=False)
    return out


def plot_model_comparison(
    rows: list[dict[str, Any]],
    filename: str = "model_comparison.png",
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / Path(filename).name
    df = pd.DataFrame(rows)
    y = df["accuracy_pct"] if "accuracy_pct" in df.columns else df["accuracy"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["model"], y, color="#2a6f97")
    ax.axhline(
        95.98,
        color="#d62828",
        linestyle="--",
        linewidth=1,
        label="Paper CatBoost 95.98%",
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model comparison (test accuracy)")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Figures dir: {FIGURES_DIR}")
