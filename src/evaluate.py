"""Evaluation helpers and figure export for obesity models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def evaluate(y_true, y_pred) -> dict:
    """Return a metrics dict. Expand once labels/task type are fixed."""
    from sklearn.metrics import accuracy_score

    return {"accuracy": float(accuracy_score(y_true, y_pred))}


def save_metrics(metrics: dict, name: str = "main_results.csv") -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / name
    pd.DataFrame([metrics]).to_csv(out, index=False)
    return out


if __name__ == "__main__":
    print(f"Results dir: {RESULTS_DIR}")
