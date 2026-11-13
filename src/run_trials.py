import os
import json
import numpy as np
import pandas as pd
from ann_model import train_and_eval, load

PAPER_TOP5 = {
    # "full": ALL16,  # Đã chạy và lưu kết quả trước đó
    "chi2": ["Weight", "FCVC", "CALC", "FAVC", "FAF"],
    "f_classify": ["FAF", "CALC", "TUE", "FAVC", "FCVC"],
    "mutual_info": ["Weight", "FAF", "SMOKE", "FAVC", "CALC"],
}

def main():
    with open("results/bayes_best_params.json", encoding="utf-8") as f:
        best_params = json.load(f)

    trv, tev = load("train_for_val"), load("test_for_val")
    os.makedirs("results", exist_ok=True)

    for name, cols in PAPER_TOP5.items():
        if name not in best_params:
            print(f"[Cảnh báo] Bỏ qua '{name}' vì không có trong bayes_best_params.json")
            continue

        accs, preds, probas = [], [], []
        for i in range(1, 11):
            res = train_and_eval(cols, best_params[name], trv, tev, seed=i, verbose=0)
            accs.append(res["acc"] * 100)
            preds.append(res["y_pred"])
            probas.append(res["proba"])
            print(f"[{name}] Trial {i}/10 - Acc: {accs[-1]:.2f}%")

        df = pd.DataFrame({
            "trial": range(1, 11),
            "seed": range(1, 11),
            "accuracy": [round(a, 2) for a in accs],
        })
        df.loc[len(df)] = ["Mean", "-", round(np.mean(accs), 2)]
        df.loc[len(df)] = ["SD", "-", round(np.std(accs, ddof=1), 2)]

        df.to_csv(f"results/ann_trials_{name}.csv", index=False)
        np.savez(f"results/{name}_preds.npz", y_pred=np.array(preds), y_proba=np.array(probas))
        print(f"-> Đã lưu: results/ann_trials_{name}.csv\n")

if __name__ == "__main__":
    main()