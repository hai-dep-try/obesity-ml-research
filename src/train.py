"""Train baseline / main models and write metrics under results/."""

from __future__ import annotations

from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def train() -> None:
    """Train models and persist metrics/artifacts. Implement after EDA."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Wire preprocess → model fit → results CSV here.")


if __name__ == "__main__":
    train()
