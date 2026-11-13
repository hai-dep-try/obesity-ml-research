"""Comprehensive model error analysis, prediction correlation, and BMI ablation study."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.evaluate import (
    FIGURES_DIR,
    RESULTS_DIR,
    compute_metrics,
    metrics_to_percent_row,
)
from src.preprocess import BINARY_COLS, ONEHOT_COLS, RANDOM_STATE, load_prepared, load_raw
from src.train import PAPER_ACC


def _who_bmi_rule(bmi: float) -> str:
    if bmi < 18.5:
        return "Insufficient_Weight"
    if bmi < 25.0:
        return "Normal_Weight"
    if bmi < 27.5:
        return "Overweight_Level_I"
    if bmi < 30.0:
        return "Overweight_Level_II"
    if bmi < 35.0:
        return "Obesity_Type_I"
    if bmi < 40.0:
        return "Obesity_Type_II"
    return "Obesity_Type_III"


def _load_best_params() -> dict:
    bp_path = RESULTS_DIR / "best_params.json"
    if bp_path.exists():
        return json.loads(bp_path.read_text(encoding="utf-8"))
    return {
        "LogisticRegression": {"C": 10.0, "solver": "lbfgs"},
        "KNN": {"n_neighbors": 3, "weights": "distance"},
        "RandomForest": {"max_depth": 20, "min_samples_split": 2, "n_estimators": 200},
        "NaiveBayes": {"var_smoothing": 1e-7},
        "CatBoost": {"learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 5, "iterations": 300},
    }


def run_error_and_correlation_analysis() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = load_prepared()
    X_train = np.asarray(data.preprocessor.transform(data.X_train))
    X_test = np.asarray(data.preprocessor.transform(data.X_test))

    labels = sorted(data.y_test.unique())
    le = LabelEncoder().fit(labels)
    y_train = le.transform(data.y_train)
    y_test = le.transform(data.y_test)

    bp = _load_best_params()
    cb_params = bp.get("CatBoost", {"learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 5, "iterations": 300})
    rf_params = bp.get("RandomForest", {"n_estimators": 200, "max_depth": 20, "min_samples_split": 2})
    lr_params = bp.get("LogisticRegression", {"C": 10.0, "solver": "lbfgs"})
    knn_params = bp.get("KNN", {"n_neighbors": 3, "weights": "distance"})
    nb_params = bp.get("NaiveBayes", {"var_smoothing": 1e-7})

    models = {
        "CatBoost": CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            **cb_params,
        ),
        "XGBoost_Tuned": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            n_jobs=-1,
        ),
        "XGBoost_HoTuAn": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **rf_params,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            **lr_params,
        ),
        "KNN": KNeighborsClassifier(**knn_params),
        "NaiveBayes": GaussianNB(**nb_params),
    }

    preds: dict[str, np.ndarray] = {}
    probas: dict[str, np.ndarray] = {}
    summary_rows: list[dict] = []
    per_class_f1: dict[str, np.ndarray] = {}
    err_pair_rows: list[dict] = []

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        pred = np.asarray(clf.predict(X_test)).ravel().astype(int)
        proba = clf.predict_proba(X_test)
        preds[name] = pred
        probas[name] = proba

        m = compute_metrics(y_test, pred)
        row = metrics_to_percent_row(name, m)
        n_err = int((pred != y_test).sum())
        row["n_errors"] = n_err
        paper_key = name
        row["paper_accuracy_pct"] = PAPER_ACC.get(paper_key)
        row["delta_accuracy_pp"] = (
            round(row["accuracy_pct"] - row["paper_accuracy_pct"], 2)
            if row["paper_accuracy_pct"] is not None
            else None
        )
        summary_rows.append(row)

        per_class_f1[name] = np.round(
            f1_score(y_test, pred, labels=np.arange(len(labels)), average=None, zero_division=0) * 100,
            2,
        )

        cm = confusion_matrix(y_test, pred, labels=np.arange(len(labels)))
        pairs = []
        for i, true_lbl in enumerate(labels):
            for j, pred_lbl in enumerate(labels):
                if i != j and cm[i, j] > 0:
                    pairs.append((int(cm[i, j]), true_lbl, pred_lbl))
        pairs.sort(reverse=True)
        for cnt, true_lbl, pred_lbl in pairs[:6]:
            err_pair_rows.append(
                {
                    "model": name,
                    "true_class": true_lbl,
                    "predicted_class": pred_lbl,
                    "count": cnt,
                    "share_of_model_errors_pct": round(cnt / max(n_err, 1) * 100, 2),
                }
            )

    # Soft Voting Ensemble (CatBoost + XGBoost_Tuned + LogisticRegression)
    sv_proba = (probas["CatBoost"] + probas["XGBoost_Tuned"] + probas["LogisticRegression"]) / 3.0
    sv_pred = sv_proba.argmax(axis=1)
    sv_metrics = compute_metrics(y_test, sv_pred)
    sv_row = metrics_to_percent_row("SoftVoting_CB_XGB_LR", sv_metrics)
    sv_row["n_errors"] = int((sv_pred != y_test).sum())
    sv_row["paper_accuracy_pct"] = None
    sv_row["delta_accuracy_pp"] = None
    summary_rows.append(sv_row)

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "extended_comparison.csv", index=False)
    pd.DataFrame(per_class_f1, index=labels).to_csv(RESULTS_DIR / "per_class_f1_comparison.csv")
    pd.DataFrame(err_pair_rows).to_csv(RESULTS_DIR / "error_analysis_top_pairs.csv", index=False)

    # Pairwise correlation matrices
    model_names = list(models.keys())
    agree_df = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
    kappa_df = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
    err_corr_df = pd.DataFrame(index=model_names, columns=model_names, dtype=float)

    for a in model_names:
        for b in model_names:
            agree_df.loc[a, b] = round(float((preds[a] == preds[b]).mean() * 100), 2)
            kappa_df.loc[a, b] = round(float(cohen_kappa_score(preds[a], preds[b])), 4)
            ea = (preds[a] != y_test).astype(float)
            eb = (preds[b] != y_test).astype(float)
            err_corr_df.loc[a, b] = round(float(np.corrcoef(ea, eb)[0, 1]), 4)

    agree_df.to_csv(RESULTS_DIR / "model_agreement.csv")
    kappa_df.to_csv(RESULTS_DIR / "model_kappa.csv")
    err_corr_df.to_csv(RESULTS_DIR / "model_error_correlation.csv")

    # Extended model comparison figure
    comp_df = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = [
        "#1b4965" if m in {"CatBoost", "SoftVoting_CB_XGB_LR"} else "#2a6f97"
        for m in comp_df["model"]
    ]
    bars = ax.bar(comp_df["model"], comp_df["accuracy_pct"], color=colors)
    ax.axhline(95.98, color="#d62828", linestyle="--", linewidth=1.3, label="Paper CatBoost (95.98%)", zorder=2)
    ax.set_ylim(40, 107)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Extended Model Comparison (Standardized 80:20 Split, N=423)", pad=12)
    for bar, val in zip(bars, comp_df["accuracy_pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.0,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.2),
            zorder=4,
        )
    ax.legend(loc="lower left", framealpha=0.95)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison_extended.png", dpi=150)
    plt.close(fig)

    # Correlation heatmaps figure (Agreement % & Error Correlation)
    short_names = ["CatBoost", "XGB Tuned", "XGB HoTuAn", "RandomForest", "LogisticReg", "KNN", "NaiveBayes"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(
        agree_df.values,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=short_names,
        yticklabels=short_names,
        vmin=45,
        vmax=100,
        ax=axes[0],
    )
    axes[0].set_title("Prediction Agreement (%)")
    axes[0].tick_params(axis="x", rotation=25)

    sns.heatmap(
        err_corr_df.values,
        annot=True,
        fmt=".2f",
        cmap="OrRd",
        xticklabels=short_names,
        yticklabels=short_names,
        vmin=0,
        vmax=1,
        ax=axes[1],
    )
    axes[1].set_title("Binary Error Correlation ($r_{err}$)")
    axes[1].tick_params(axis="x", rotation=25)

    fig.suptitle("Model Prediction Agreement & Error Overlap Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_correlation_heatmaps.png", dpi=150)
    plt.close(fig)

    print("=== EXTENDED COMPARISON ===")
    print(comp_df[["model", "accuracy_pct", "precision_pct", "recall_pct", "f1_pct", "n_errors"]].to_string(index=False))
    print("\n=== PREDICTION AGREEMENT (%) ===")
    print(agree_df.to_string())
    print("\n=== ERROR CORRELATION (r_err) ===")
    print(err_corr_df.to_string())

    return {
        "summary": comp_df,
        "agreement": agree_df,
        "kappa": kappa_df,
        "error_correlation": err_corr_df,
    }


def run_bmi_ablation_study() -> pd.DataFrame:
    df = load_raw()
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    rule_full_pred = df["BMI"].apply(_who_bmi_rule)
    rule_full_acc = accuracy_score(df["NObeyesdad"], rule_full_pred) * 100
    rule_full_err = int((rule_full_pred != df["NObeyesdad"]).sum())

    data = load_prepared()
    test_bmi = data.X_test["Weight"] / (data.X_test["Height"] ** 2)
    rule_test_pred = test_bmi.apply(_who_bmi_rule)
    rule_test_acc = accuracy_score(data.y_test, rule_test_pred) * 100
    rule_test_err = int((rule_test_pred != data.y_test).sum())

    bp = _load_best_params()
    cb_params = bp.get("CatBoost", {"learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 5, "iterations": 300})

    # 1) CatBoost WITHOUT Weight & Height (Lifestyle + Habits + Age + Gender)
    work = df.drop(columns=["NObeyesdad", "Weight", "Height", "BMI"]).copy()
    for col in BINARY_COLS:
        work[col] = work[col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).astype(float)
    y_all = df["NObeyesdad"].astype(str)
    X_tr_nw, X_te_nw, y_tr_nw, y_te_nw = train_test_split(
        work, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
    )
    num_cols_nw = [c for c in work.columns if c not in ONEHOT_COLS]
    pre_nw = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols_nw),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONEHOT_COLS),
        ]
    )
    X_tr_nw_p = pre_nw.fit_transform(X_tr_nw)
    X_te_nw_p = pre_nw.transform(X_te_nw)
    cb_nw = CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        **cb_params,
    )
    cb_nw.fit(X_tr_nw_p, y_tr_nw)
    p_nw = np.asarray(cb_nw.predict(X_te_nw_p)).ravel()
    m_nw = compute_metrics(y_te_nw, p_nw)

    # 2) CatBoost Standard (16 raw features -> 27 encoded)
    X_tr_std = np.asarray(data.preprocessor.transform(data.X_train))
    X_te_std = np.asarray(data.preprocessor.transform(data.X_test))
    cb_std = CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        **cb_params,
    )
    cb_std.fit(X_tr_std, data.y_train)
    p_std = np.asarray(cb_std.predict(X_te_std)).ravel()
    m_std = compute_metrics(data.y_test, p_std)

    # 3) CatBoost WITH explicit BMI feature
    tr_bmi = (data.X_train["Weight"] / (data.X_train["Height"] ** 2)).to_numpy().reshape(-1, 1)
    te_bmi = (data.X_test["Weight"] / (data.X_test["Height"] ** 2)).to_numpy().reshape(-1, 1)
    cb_bmi = CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        **cb_params,
    )
    cb_bmi.fit(np.hstack([X_tr_std, tr_bmi]), data.y_train)
    p_bmi = np.asarray(cb_bmi.predict(np.hstack([X_te_std, te_bmi]))).ravel()
    m_bmi = compute_metrics(data.y_test, p_bmi)

    rows = [
        {
            "experiment": "CatBoost (No Weight & Height)",
            "description": "14 features: eating habits, physical activity, age, gender only",
            "eval_scope": "Test (N=423)",
            "accuracy_pct": round(m_nw["accuracy"] * 100, 2),
            "f1_pct": round(m_nw["f1"] * 100, 2),
            "n_errors": int((p_nw != y_te_nw).sum()),
        },
        {
            "experiment": "WHO BMI Rule (Test Set)",
            "description": "Deterministic cutoffs on Weight/Height^2 (no ML)",
            "eval_scope": "Test (N=423)",
            "accuracy_pct": round(rule_test_acc, 2),
            "f1_pct": round(f1_score(data.y_test, rule_test_pred, average="weighted") * 100, 2),
            "n_errors": rule_test_err,
        },
        {
            "experiment": "WHO BMI Rule (Full Dataset)",
            "description": "Deterministic cutoffs on Weight/Height^2 across all rows",
            "eval_scope": "Full (N=2111)",
            "accuracy_pct": round(rule_full_acc, 2),
            "f1_pct": round(f1_score(df["NObeyesdad"], rule_full_pred, average="weighted") * 100, 2),
            "n_errors": rule_full_err,
        },
        {
            "experiment": "CatBoost (Standard Pipeline)",
            "description": "16 features (Maulana et al. 2024 setup, no explicit BMI)",
            "eval_scope": "Test (N=423)",
            "accuracy_pct": round(m_std["accuracy"] * 100, 2),
            "f1_pct": round(m_std["f1"] * 100, 2),
            "n_errors": int((p_std != data.y_test).sum()),
        },
        {
            "experiment": "CatBoost (+ Explicit BMI)",
            "description": "16 features + engineered BMI = Weight/Height^2",
            "eval_scope": "Test (N=423)",
            "accuracy_pct": round(m_bmi["accuracy"] * 100, 2),
            "f1_pct": round(m_bmi["f1"] * 100, 2),
            "n_errors": int((p_bmi != data.y_test).sum()),
        },
    ]

    ab_df = pd.DataFrame(rows)
    ab_df.to_csv(RESULTS_DIR / "bmi_ablation_results.csv", index=False)

    # Plot Ablation Study
    plot_rows = [rows[0], rows[2], rows[3], rows[4]]
    p_df = pd.DataFrame(plot_rows)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    palette = ["#6c757d", "#e76f51", "#2a6f97", "#2a9d8f"]
    bars = ax.bar(p_df["experiment"], p_df["accuracy_pct"], color=palette)
    ax.set_ylim(70, 103)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ablation Study: Why ML Outperforms Deterministic WHO BMI Rule")
    for bar, val in zip(bars, p_df["accuracy_pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.7,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bmi_ablation_study.png", dpi=150)
    plt.close(fig)

    print("\n=== BMI ABLATION STUDY ===")
    print(ab_df[["experiment", "eval_scope", "accuracy_pct", "f1_pct", "n_errors"]].to_string(index=False))
    return ab_df


if __name__ == "__main__":
    run_error_and_correlation_analysis()
    run_bmi_ablation_study()
