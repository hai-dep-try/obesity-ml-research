from ann_model import train_and_eval, load, ALL16

trv, tev = load("train_for_val"), load("test_for_val")
params = {"units": 30, "lr": 0.013, "epoch": 50, "batch": 16}

# Lặp lại: cùng seed hai lần
a = train_and_eval(ALL16, params, trv, tev, seed=1)["acc"]
b = train_and_eval(ALL16, params, trv, tev, seed=1)["acc"]
print(f"Cùng seed: {a:.4f} vs {b:.4f}")

# Khác seed
c = train_and_eval(ALL16, params, trv, tev, seed=2)["acc"]
print(f"Seed khác: {c:.4f}")

# 5 đặc trưng
cols5 = ["FAF", "CALC", "TUE", "FAVC", "FCVC"]
print("5 đặc trưng:", train_and_eval(cols5, params, trv, tev)["acc"])