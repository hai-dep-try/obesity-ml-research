import time
from ann_model import train_and_eval, load, ALL16

trv, tev = load("train_for_val"), load("test_for_val")

# Chạy khởi động để loại thời gian nạp TensorFlow
train_and_eval(ALL16, {"units": 30, "lr": 0.013, "epoch": 2, "batch": 16}, trv, tev)

print("--- Đo theo batch size (units = 30) ---")
for batch in [32, 16, 8, 1]:
    ep = 5 if batch == 1 else 20      # batch=1 rất chậm nên đo ít epoch hơn
    t0 = time.time()
    train_and_eval(ALL16, {"units": 30, "lr": 0.013, "epoch": ep, "batch": batch},
                   trv, tev)
    per_ep = (time.time() - t0) / ep
    print(f"batch={batch:>2}: {per_ep:.3f}s/epoch "
          f"-> 1000 epoch ≈ {per_ep*1000/60:.1f} phút")

print("--- Đo theo số nơ-ron (batch = 16) ---")
for units in [30, 500, 2000, 5000]:
    t0 = time.time()
    train_and_eval(ALL16, {"units": units, "lr": 0.013, "epoch": 20, "batch": 16},
                   trv, tev)
    print(f"units={units:>4}: {(time.time() - t0) / 20:.3f}s/epoch")