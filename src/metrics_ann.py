import argparse, glob, os, re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix

# Dùng lại đúng công thức đã dùng cho Logistic Regression
from lr_baseline import specificity_weighted, brier_multiclass


def trial_metrics(path):
    d = np.load(path, allow_pickle=True)
    y_true, y_pred = d["y_true"], d["y_pred"]
    proba, classes = d["proba"], [str(c) for c in d["classes"]]
    idx = np.arange(len(classes))

    # Kiểm tra định dạng
    assert proba.shape == (len(y_true), len(classes)), f"Sai kích thước proba: {path}"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4), f"proba không tổng bằng 1: {path}"

    cm = confusion_matrix(y_true, y_pred, labels=idx)
    f1_cls = f1_score(y_true, y_pred, labels=idx, average=None, zero_division=0)

    row = {
        "seed": int(re.search(r"seed(\d+)", os.path.basename(path)).group(1)),
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "sensitivity": recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "specificity": specificity_weighted(cm) * 100,
        "brier": brier_multiclass(y_true, proba, idx),
    }
    for c, v in zip(classes, f1_cls):
        row[f"f1_{c}"] = v * 100
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True,
                    choices=["full", "chi2", "f_classify", "mutual_info"])
    ap.add_argument("--dir", default="results/ann_trials")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.dir}/{args.set}_seed*.npz"))
    assert files, f"Không thấy file nào trong {args.dir} cho bộ '{args.set}'"
    df = pd.DataFrame([trial_metrics(f) for f in files]).sort_values("seed")
    df.insert(0, "trial", range(1, len(df) + 1))

    # Bảng giống Bảng 4: từng trial + mean + SD
    # SD dạng quần thể (ddof=0): khớp bài báo (SD của Obesity Type I = 40 chỉ ra được cách này)
    num = df.drop(columns=["trial", "seed"])
    mean_row = {"trial": "mean", "seed": "", **num.mean().to_dict()}
    sd_row = {"trial": "SD", "seed": "", **num.std(ddof=0).to_dict()}
    table4 = pd.concat([df, pd.DataFrame([mean_row, sd_row])], ignore_index=True)

    os.makedirs(args.out_dir, exist_ok=True)
    table4.round(4).to_csv(f"{args.out_dir}/ann_trials_{args.set}.csv", index=False)

    # Dòng tóm tắt giống Bảng 7
    summary = {
        "feature_set": args.set, "n_trials": len(df),
        "accuracy": num["accuracy"].mean(), "f1": num["f1_weighted"].mean(),
        "sd_accuracy": num["accuracy"].std(ddof=0), "sd_f1": num["f1_weighted"].std(ddof=0),
        "sensitivity": num["sensitivity"].mean(), "specificity": num["specificity"].mean(),
        "brier": num["brier"].mean(),
    }
    sp = f"{args.out_dir}/ann_summary.csv"
    old = pd.read_csv(sp) if os.path.exists(sp) else pd.DataFrame()
    if len(old):
        old = old[old["feature_set"] != args.set]          # ghi đè dòng cùng bộ đặc trưng
    pd.concat([old, pd.DataFrame([summary])], ignore_index=True).round(4).to_csv(sp, index=False)

    print(table4.round(2).to_string(index=False))
    print("\nTóm tắt:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()})

    # So sánh nhanh với Logistic Regression (cùng bộ đặc trưng)
    lr_path = "results/lr_results.csv"
    if os.path.exists(lr_path):
        lr = pd.read_csv(lr_path)
        lr = lr[lr["feature_set"] == args.set]
        if len(lr):
            print("\nLR cùng bộ đặc trưng:",
                  lr[["accuracy", "f1_weighted", "sensitivity", "specificity", "brier"]]
                  .iloc[0].to_dict())


if __name__ == "__main__":
    main()