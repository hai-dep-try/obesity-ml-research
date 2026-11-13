import os
import numpy as np
from ann_model import train_and_eval, load, ALL16

tr, te = load("train"), load("test")
params = {"units": 30, "lr": 0.013, "epoch": 50, "batch": 16}   # nhẹ, chỉ để thử
os.makedirs("results/ann_trials_test", exist_ok=True)

for seed in [1, 2, 3]:
    r = train_and_eval(ALL16, params, tr, te, seed=seed)
    np.savez(f"results/ann_trials_test/full_seed{seed}.npz",
             y_true=r["y_true"], y_pred=r["y_pred"],
             proba=r["proba"], classes=r["classes"])
    print("seed", seed, "acc", round(r["acc"] * 100, 2))