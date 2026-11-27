"""Generate additional publication-grade figures from verified CSVs and dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT_DIR / "data" / "ObesityDataSet_raw_and_data_sinthetic.csv"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

CLASS_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

CLASS_SHORT = {
    "Insufficient_Weight": "Insufficient",
    "Normal_Weight": "Normal",
    "Overweight_Level_I": "Overweight I",
    "Overweight_Level_II": "Overweight II",
    "Obesity_Type_I": "Obesity I",
    "Obesity_Type_II": "Obesity II",
    "Obesity_Type_III": "Obesity III",
}


def plot_dataset_composition_and_bmi() -> Path:
    """Plot 2,111 dataset composition (498 original + 1,613 SMOTE) and BMI spread."""
    df = pd.read_csv(DATA_CSV)
    df["source"] = np.where(np.arange(len(df)) < 498, "Gốc (N=498)", "SMOTE (N=1.613)")
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    orig_counts = [
        int(((df["NObeyesdad"] == c) & (np.arange(len(df)) < 498)).sum())
        for c in CLASS_ORDER
    ]
    smote_counts = [
        int(((df["NObeyesdad"] == c) & (np.arange(len(df)) >= 498)).sum())
        for c in CLASS_ORDER
    ]
    labels = [CLASS_SHORT[c] for c in CLASS_ORDER]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    y = np.arange(len(CLASS_ORDER))
    ax1.barh(y, orig_counts, color="#1b4965", label="Khảo sát gốc (498 mẫu)")
    ax1.barh(y, smote_counts, left=orig_counts, color="#62b6cb", label="Tổng hợp SMOTE (1.613 mẫu)")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9.5)
    ax1.set_ylim(len(CLASS_ORDER) - 0.35, -1.25)
    ax1.set_xlim(0, 480)
    ax1.set_xlabel("Số lượng mẫu (N = 2.111)")
    ax1.set_title("Phân bố 7 lớp: Dữ liệu gốc vs. SMOTE", fontweight="bold")
    ax1.legend(loc="upper center", ncol=2, fontsize=8.3, framealpha=0.95)
    ax1.grid(axis="x", linestyle=":", alpha=0.5)

    for idx, (o, s) in enumerate(zip(orig_counts, smote_counts)):
        tot = o + s
        ax1.text(tot + 6, idx, f"{tot} ({o} gốc)", va="center", fontsize=8.5, fontweight="bold")

    box_data = [df.loc[df["NObeyesdad"] == c, "BMI"].values for c in CLASS_ORDER]
    bp = ax2.boxplot(
        box_data,
        vert=False,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=2.8, alpha=0.45, markerfacecolor="#e76f51", markeredgecolor="none"),
        medianprops=dict(color="#1d3557", linewidth=1.6),
    )
    palette = ["#8ecae6", "#219ebc", "#ffb703", "#fb8500", "#e76f51", "#d62828", "#9b2226"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    cutoffs = [18.5, 25.0, 27.5, 30.0, 35.0, 40.0]
    for cut in cutoffs:
        ax2.axvline(cut, color="#495057", linestyle="--", linewidth=0.9, alpha=0.7)
        ax2.text(cut, 0.42, f"{cut:g}", ha="center", va="bottom", fontsize=7.5, color="#343a40")

    ax2.set_yticks(np.arange(1, len(CLASS_ORDER) + 1))
    ax2.set_yticklabels(labels, fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_ylim(len(CLASS_ORDER) + 0.6, 0.1)
    ax2.set_xlim(11, 53)
    ax2.set_xlabel("Chỉ số BMI = Weight / Height² (kg/m²)")
    ax2.set_title("Phân bố BMI theo lớp & Ngưỡng WHO", fontweight="bold")
    ax2.grid(axis="x", linestyle=":", alpha=0.4)

    fig.suptitle("Giải phẫu Bộ dữ liệu Obesity Levels (2.111 mẫu, 7 lớp mục tiêu)", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "dataset_composition_bmi.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_ann_vs_lr_comparison() -> Path:
    """Plot ANN (10 trials mean ± SD) vs LR vs Yagin et al. (2023) across 4 feature sets."""
    ann_df = pd.read_csv(RESULTS_DIR / "ann_summary.csv").set_index("feature_set")
    lr_df = pd.read_csv(RESULTS_DIR / "lr_results.csv").set_index("feature_set")

    order = ["full", "chi2", "mutual_info", "f_classify"]
    labels = ["Full\n(16 biến)", "Chi-Square\n(Top 5)", "Mutual Info\n(Top 5)", "F-Classify\n(Top 5)"]
    paper_acc = {"full": 93.06, "chi2": 89.04, "mutual_info": 86.52, "f_classify": 90.32}
    paper_brier = {"full": 0.094, "chi2": 0.147, "mutual_info": 0.194, "f_classify": 0.122}

    ann_acc = [float(ann_df.loc[k, "accuracy"]) for k in order]
    ann_sd = [float(ann_df.loc[k, "sd_accuracy"]) for k in order]
    lr_acc = [float(lr_df.loc[k, "accuracy"]) for k in order]
    p_acc = [paper_acc[k] for k in order]

    ann_br = [float(ann_df.loc[k, "brier"]) for k in order]
    lr_br = [float(lr_df.loc[k, "brier"]) for k in order]
    p_br = [paper_brier[k] for k in order]

    x = np.arange(len(order))
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    b1 = ax1.bar(x - w, ann_acc, width=w, yerr=ann_sd, capsize=3, color="#2a6f97", label="ANN Thực nghiệm (10 trials)")
    b2 = ax1.bar(x, lr_acc, width=w, color="#2a9d8f", label="Logistic Regression (Đối chứng)")
    b3 = ax1.bar(x + w, p_acc, width=w, color="#e76f51", alpha=0.88, label="Bài báo Yagin et al. (2023)")

    for bars, is_ann in [(b1, True), (b2, False), (b3, False)]:
        for idx, bar in enumerate(bars):
            val = bar.get_height()
            offset = (ann_sd[idx] if is_ann else 0.0) + 1.2
            ax1.text(bar.get_x() + bar.get_width() / 2, val + offset, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylim(20, 116)
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Độ chính xác trên tập Test (N = 528, Split 75:25)", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.3, framealpha=0.95)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    c1 = ax2.bar(x - w, ann_br, width=w, color="#2a6f97", label="ANN Thực nghiệm")
    c2 = ax2.bar(x, lr_br, width=w, color="#2a9d8f", label="Logistic Regression")
    c3 = ax2.bar(x + w, p_br, width=w, color="#e76f51", alpha=0.88, label="Bài báo Yagin et al. (2023)")

    for bars in [c1, c2, c3]:
        for bar in bars:
            val = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylim(0, 0.88)
    ax2.set_ylabel("Brier Score (thấp hơn = tốt hơn)")
    ax2.set_title("Hiệu chuẩn xác suất (Brier Score đa lớp)", fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8.3, framealpha=0.95)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Pipeline B (Yagin et al., 2023): ANN tối ưu Bayesian vs. Logistic Regression", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "ann_vs_lr_comparison.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_f_classify_critique() -> Path:
    """Visualize true ANOVA F-scores vs the 5 lowest-scoring features chosen in Yagin (2023)."""
    fs = pd.read_csv(RESULTS_DIR / "feature_selection_scores.csv")
    fs = fs.sort_values("f_classify_score", ascending=True).reset_index(drop=True)

    colors = []
    for _, r in fs.iterrows():
        if bool(r["in_paper_f_classify"]):
            colors.append("#d62828")
        elif r["feature"] in {"Weight", "Gender", "family_history_with_overweight", "CAEC"}:
            colors.append("#2a9d8f")
        else:
            colors.append("#8d99ae")

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    y = np.arange(len(fs))
    bars = ax.barh(y, fs["f_classify_score"], color=colors)
    ax.set_xscale("log")
    ax.set_xlim(2, 4200)
    ax.set_yticks(y)
    ax.set_yticklabels(fs["feature"], fontsize=9)
    ax.set_xlabel("Điểm thống kê ANOVA F trên tập train_for_val (thang log)")
    ax.set_title(
        "Phản biện F-Classify: 5 biến bài báo Yagin (2023) chọn\nnằm ở nhóm có điểm thống kê F thấp",
        fontweight="bold",
        fontsize=11,
        pad=10,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    for bar, val in zip(bars, fs["f_classify_score"]):
        ax.text(val * 1.12, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=8.2, fontweight="bold")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#d62828", label="Top 5 theo bài báo Yagin (2023): FAF, CALC, TUE, FAVC, FCVC"),
        Patch(facecolor="#2a9d8f", label="Biến có điểm F cao thực tế bị bài báo bỏ sót (Weight #1 = 1152.66)"),
        Patch(facecolor="#8d99ae", label="Các đặc trưng còn lại"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.2, framealpha=0.95)

    fig.tight_layout()
    out = FIGURES_DIR / "f_classify_critique.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_per_class_f1_heatmap() -> Path:
    """Heatmap of F1-score (%) per obesity class across all 8 models in Pipeline A (N=423)."""
    df = pd.read_csv(RESULTS_DIR / "per_class_f1_comparison.csv", index_col=0)
    df = df.loc[CLASS_ORDER]
    col_rename = {
        "CatBoost": "CatBoost",
        "XGBoost_Tuned": "XGB Tuned",
        "XGBoost_HoTuAn": "XGB HoTuAn",
        "RandomForest": "RandomForest",
        "LogisticRegression": "LogisticReg",
        "KNN_Tuned": "KNN Tuned",
        "KNN": "KNN Base",
        "NaiveBayes": "NaiveBayes",
    }
    df = df.rename(columns=col_rename)
    df.index = [CLASS_SHORT[i] for i in df.index]

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    sns.heatmap(
        df.values,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=list(df.columns),
        yticklabels=list(df.index),
        vmin=20,
        vmax=100,
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("F1-Score (%) theo từng lớp béo phì — Pipeline A (Split 80:20, N = 423)", fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    out = FIGURES_DIR / "per_class_f1_heatmap.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_sp1_literature_benchmark() -> Path:
    """Plot published accuracy benchmarks of surveyed literature for Product 1."""
    items = [
        ("Maulana et al. (2024) — CatBoost", 95.98, "#1b4965", "2.111 mẫu (80:20)"),
        ("Maulana et al. (2024) — Random Forest", 95.04, "#2a6f97", "2.111 mẫu (80:20)"),
        ("Yagin et al. (2023) — ANN + Bayes (Full)", 93.06, "#2a9d8f", "Bài báo công bố (75:25)"),
        ("Yagin et al. (2023) — ANN + F-Classify", 90.32, "#4ea8de", "Top 5 biến (nghi vấn F-score)"),
        ("Maulana et al. (2024) — KNN", 88.18, "#2a6f97", "2.111 mẫu (80:20)"),
        ("Kıvrak (2021) — Deep Learning DNN", 82.00, "#6c757d", "10-fold CV, chưa tune tự động"),
        ("De-La-Hoz-Correa (2019) — Cây J48", 97.80, "#e9c46a", "Lưu ý: F1 = 0% ở Obesity Type III"),
        ("Maulana et al. (2024) — Logistic Reg.", 65.48, "#f4a261", "Nghi vấn chưa hội tụ (max_iter=100)"),
        ("Maulana et al. (2024) — Naive Bayes*", 64.07, "#e76f51", "Recall=64.07% vs Acc ghi nhầm 94.07%"),
        ("Pang et al. (2021) — XGBoost (EHR trẻ em)", 66.14, "#adb5bd", "Biên độ 30.90%–66.14% (dữ liệu EHR)"),
    ]
    names = [x[0] for x in items]
    vals = [x[1] for x in items]
    cols = [x[2] for x in items]
    notes = [x[3] for x in items]

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    y = np.arange(len(items))
    bars = ax.barh(y, vals, color=cols, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 148)
    ax.set_xlabel("Độ chính xác công bố trong bài báo gốc (%)")
    ax.set_title("Tổng hợp hiệu năng công bố & các điểm cần kiểm chứng từ tài liệu khảo sát", fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    for bar, val, note in zip(bars, vals, notes):
        ax.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%  ({note})", va="center", fontsize=8.1)

    fig.tight_layout()
    out = FIGURES_DIR / "sp1_literature_benchmark.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_dataset_composition_and_bmi(),
        plot_ann_vs_lr_comparison(),
        plot_f_classify_critique(),
        plot_per_class_f1_heatmap(),
        plot_sp1_literature_benchmark(),
    ]
    for path in outputs:
        print("Saved:", path)


if __name__ == "__main__":
    main()
