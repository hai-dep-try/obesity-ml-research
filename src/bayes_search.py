import os
import sys

# 1. Tắt toàn bộ cảnh báo oneDNN và log rác từ TensorFlow trước khi import bất kỳ thư viện nào
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import json
import time
import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

# Đảm bảo nhận diện các module cùng thư mục src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ann_model import train_and_eval, load

ALL16 = [
    "Age", "Gender", "Height", "Weight", "family_history_with_overweight",
    "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS"
]

PAPER_TOP5 = {
    "chi2": ["Weight", "FCVC", "CALC", "FAVC", "FAF"],
    "f_classify": ["FAF", "CALC", "TUE", "FAVC", "FCVC"],
    "mutual_info": ["Weight", "FAF", "SMOKE", "FAVC", "CALC"],
}

SPLIT_DIR = "data/splits"
OUT_DIR = "results/bayes"
BEST_JSON = os.path.join("results", "bayes_best_params.json")
SEED_GP = 42     # seed của Bayesian optimizer
SEED_NN = 1      # seed huấn luyện mạng

# Không gian tìm kiếm 
SPACE = [
    Integer(16, 256, name="units"),                  # Giảm xuống mức vừa phải thay vì 5000
    Real(1e-4, 1e-2, prior="log-uniform", name="lr"),# Thu hẹp khoảng learning rate tối ưu
    Integer(20, 150, name="epoch"),                  # Giới hạn epoch quanh 150 
    Integer(8, 32, name="batch"),                    # Batch size chuẩn từ 8 đến 32
]


def get_cols(name):
    if name == "full":
        return ALL16
    cols = PAPER_TOP5[name]
    assert len(cols) == 5 and all(c in ALL16 for c in cols), f"Cột lạ trong {name}: {cols}"
    return cols


def save_best_json(best_dict, set_name):
    """Lưu kết quả tối ưu vào file JSON an toàn."""
    os.makedirs(os.path.dirname(BEST_JSON), exist_ok=True)
    all_best = {}
    if os.path.exists(BEST_JSON):
        try:
            with open(BEST_JSON, "r", encoding="utf-8") as f:
                all_best = json.load(f)
        except Exception:
            all_best = {}
            
    all_best[set_name] = best_dict
    with open(BEST_JSON, "w", encoding="utf-8") as f:
        json.dump(all_best, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True,
                    choices=["full", "chi2", "f_classify", "mutual_info"])  
    ap.add_argument("--n_calls", type=int, default=60)
    args = ap.parse_args()

    cols = get_cols(args.set)
    trv, tev = load("train_for_val"), load("test_for_val")

    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = f"{OUT_DIR}/{args.set}_log.csv"
    if os.path.exists(log_path):
        os.replace(log_path, log_path + ".bak")       # Giữ bản log cũ dự phòng
        
    call_no = [0]
    t_start = time.time()

    # Hàm mục tiêu: tối thiểu hóa (1.0 - accuracy)
    @use_named_args(SPACE)
    def objective(units, lr, epoch, batch):
        call_no[0] += 1
        params = {
            "units": int(units),
            "lr": float(lr),
            "epoch": int(epoch),
            "batch": int(batch)
        }
        t0 = time.time()
        
        try:
            result = train_and_eval(cols, params, trv, tev, seed=SEED_NN)
            acc = float(result["acc"])
        except Exception as e:
            # Nếu gặp lỗi NaN gradient hoặc tràn bộ nhớ bất thường ở một lần thử
            print(f"\n[Cảnh báo] Lỗi tại lần thử {call_no[0]} với params {params}: {e}")
            acc = 0.0

        dt = time.time() - t0

        row = {
            "call": call_no[0],
            **params,
            "val_acc": round(acc * 100, 2),
            "sec": round(dt, 1)
        }
        
        pd.DataFrame([row]).to_csv(
            log_path,
            mode="a",
            index=False,
            header=not os.path.exists(log_path)
        )
        
        print(
            f"[{args.set}] {call_no[0]:>3}/{args.n_calls} | "
            f"units={params['units']:>4} lr={params['lr']:.5f} "
            f"epoch={params['epoch']:>3} batch={params['batch']:>2} | "
            f"val_acc={acc*100:.2f}% | {dt:.0f}s | "
            f"tổng {(time.time()-t_start)/60:.1f} phút",
            flush=True
        )
        return 1.0 - acc

    # Tối ưu hóa Bayesian (Gaussian Process với Expected Improvement)
    res = gp_minimize(
        objective,
        SPACE,
        acq_func="EI",
        n_calls=args.n_calls,
        random_state=SEED_GP
    )

    best = {
        "units": int(res.x[0]),
        "lr": float(res.x[1]),
        "epoch": int(res.x[2]),
        "batch": int(res.x[3]),
        "val_acc": round((1.0 - res.fun) * 100, 2),
        "n_calls": args.n_calls,
        "cols": cols
    }
    
    save_best_json(best, args.set)

    print("\n" + "=" * 30 + " KẾT QUẢ TỐI ƯU NHẤT " + "=" * 30)
    print(json.dumps(best, indent=2, ensure_ascii=False))
    print(f"Tổng thời gian: {(time.time() - t_start) / 60:.1f} phút")
    print(f"Đã lưu kết quả vào: {BEST_JSON}")


if __name__ == "__main__":
    main()