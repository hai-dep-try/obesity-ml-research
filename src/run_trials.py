import os
import json
import argparse
import numpy as np
from ann_model import train_and_eval, load, ALL16

PAPER_TOP5 = {
    "full": ALL16,
    "chi2": ["Weight", "FCVC", "CALC", "FAVC", "FAF"],
    "f_classify": ["FAF", "CALC", "TUE", "FAVC", "FCVC"],
    "mutual_info": ["Weight", "FAF", "SMOKE", "FAVC", "CALC"],
}

OUT_DIR = "results/ann_trials"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+",
                    default=["full", "chi2", "f_classify", "mutual_info"],
                    choices=list(PAPER_TOP5),
                    help="Các bộ đặc trưng cần chạy")
    ap.add_argument("--n_trials", type=int, default=10)
    args = ap.parse_args()

    with open("results/bayes_best_params.json", encoding="utf-8") as f:
        best_params = json.load(f)

    # Huấn luyện trên toàn bộ train, đánh giá trên test (đúng giai đoạn 3 của bài báo)
    tr, te = load("train"), load("test")
    os.makedirs(OUT_DIR, exist_ok=True)

    for name in args.sets:
        if name not in best_params:
            print(f"[Cảnh báo] Bỏ qua '{name}' vì không có trong bayes_best_params.json")
            continue
        cols = PAPER_TOP5[name]

        for i in range(1, args.n_trials + 1):
            res = train_and_eval(cols, best_params[name], tr, te, seed=i, verbose=0)
            np.savez(f"{OUT_DIR}/{name}_seed{i}.npz",
                     y_true=res["y_true"], y_pred=res["y_pred"],
                     proba=res["proba"], classes=res["classes"])
            print(f"[{name}] Trial {i}/{args.n_trials} - Acc: {res['acc']*100:.2f}%")
        print(f"-> Đã lưu {args.n_trials} file vào {OUT_DIR}/{name}_seed*.npz\n")


if __name__ == "__main__":
    main()