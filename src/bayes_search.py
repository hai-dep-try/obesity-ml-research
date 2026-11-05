import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # ẩn bớt cảnh báo TensorFlow

import argparse
import json
import time
import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from ann_model import train_and_eval, load, ALL16

SPLIT_DIR = "data/splits"
OUT_DIR = "results/bayes"
BEST_JSON = "results/bayes_best_params.json"
SEED_GP = 42     # seed của Bayesian optimizer
SEED_NN = 1      # seed huấn luyện mạng (cố định để hàm mục tiêu ổn định)

# D1. Không gian tìm kiếm
SPACE = [
    Integer(20, 5000, name="units"),
    Real(1e-5, 1e-1, prior="log-uniform", name="lr"),
    Integer(20, 800, name="epoch"),
    Integer(4, 32, name="batch"),
]


def get_cols(name):
    if name == "full":
        return ALL16
    top5 = json.load(open(f"{SPLIT_DIR}/top5_features.json"))
    cols = top5[name]
    assert len(cols) == 5 and all(c in ALL16 for c in cols), f"Cột lạ trong {name}: {cols}"
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True,
                    choices=["full", "chi2", "f_classify", "mutual_info"])  # Chọn bộ đặc trưng, thay required=True thành default="full" nếu muốn mặc định là full
    ap.add_argument("--n_calls", type=int, default=60)
    args = ap.parse_args()

    cols = get_cols(args.set)
    trv, tev = load("train_for_val"), load("test_for_val")

    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = f"{OUT_DIR}/{args.set}_log.csv"
    if os.path.exists(log_path):
        os.replace(log_path, log_path + ".bak")       # giữ bản log cũ, không xóa
    call_no = [0]
    t_start = time.time()

    # D2. Hàm mục tiêu: huấn luyện trên train_for_val, trả về 1 - accuracy(test_for_val)
    @use_named_args(SPACE)
    def objective(units, lr, epoch, batch):
        call_no[0] += 1
        params = {"units": int(units), "lr": float(lr),
                  "epoch": int(epoch), "batch": int(batch)}
        t0 = time.time()
        acc = train_and_eval(cols, params, trv, tev, seed=SEED_NN)["acc"]
        dt = time.time() - t0

        row = {"call": call_no[0], **params,
               "val_acc": round(acc * 100, 2), "sec": round(dt, 1)}
        pd.DataFrame([row]).to_csv(log_path, mode="a", index=False,
                                   header=not os.path.exists(log_path))
        print(f"[{args.set}] {call_no[0]:>3}/{args.n_calls} | "
              f"units={params['units']:>4} lr={params['lr']:.5f} "
              f"epoch={params['epoch']:>3} batch={params['batch']:>2} | "
              f"val_acc={acc*100:.2f}% | {dt:.0f}s | tổng {(time.time()-t_start)/60:.1f} phút",
              flush=True)
        return 1.0 - acc

    # D3. Tối ưu Bayesian (Gaussian Process, Expected Improvement)
    res = gp_minimize(objective, SPACE, acq_func="EI",
                      n_calls=args.n_calls, random_state=SEED_GP)

    best = {"units": int(res.x[0]), "lr": float(res.x[1]),
            "epoch": int(res.x[2]), "batch": int(res.x[3]),
            "val_acc": round((1 - res.fun) * 100, 2),
            "n_calls": args.n_calls, "cols": cols}
    all_best = json.load(open(BEST_JSON)) if os.path.exists(BEST_JSON) else {}
    all_best[args.set] = best
    with open(BEST_JSON, "w", encoding="utf-8") as f:
        json.dump(all_best, f, indent=2, ensure_ascii=False)

    print("\n=== TỐI ƯU ===")
    print(best)
    print(f"Tổng thời gian: {(time.time() - t_start) / 60:.1f} phút")


if __name__ == "__main__":
    main()