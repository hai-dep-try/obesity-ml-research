import json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

SPLIT_DIR, OUT_DIR = "data/splits", "results"
TARGET, SEED = "NObeyesdad", 42
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
ALL16 = ["Gender", "Age", "Height", "Weight", "family_history_with_overweight",
         "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE",
         "CALC", "MTRANS"]

ORD = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
MAPS = {
    "Gender": {"Female": 0, "Male": 1},
    "family_history_with_overweight": {"no": 0, "yes": 1},
    "FAVC": {"no": 0, "yes": 1}, "SCC": {"no": 0, "yes": 1},
    "SMOKE": {"no": 0, "yes": 1},
    "CAEC": ORD, "CALC": ORD,
    "MTRANS": {"Walking": 0, "Bike": 1, "Motorbike": 2,
               "Public_Transportation": 3, "Automobile": 4},
}

def load(name):
    df = pd.read_csv(f"{SPLIT_DIR}/{name}.csv")
    for col, mp in MAPS.items():
        if df[col].dtype == object:
            df[col] = df[col].map(mp)
    assert df[ALL16].isna().sum().sum() == 0, f"NaN trong {name}: kiểm tra giá trị lạ"
    return df

def specificity_weighted(cm):
    spec = []
    for i in range(len(cm)):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp; tn = cm.sum() - tp - fp - fn
        spec.append(tn / (tn + fp))
    return float(np.average(spec, weights=cm.sum(axis=1)))

def brier_multiclass(y_true, proba, classes):
    onehot = (np.asarray(y_true)[:, None] == classes[None, :]).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))

def run(feature_set, cols, d):
    trv, tev, tr, te = d["train_for_val"], d["test_for_val"], d["train"], d["test"]

    # 1) Chọn C: train trên train_for_val, chấm trên test_for_val
    best_c, best_acc = None, -1
    for c in C_GRID:
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=c, max_iter=5000, random_state=SEED))
        m.fit(trv[cols], trv[TARGET])
        acc = accuracy_score(tev[TARGET], m.predict(tev[cols]))
        if acc > best_acc:
            best_c, best_acc = c, acc

    # 2) Train lại trên toàn bộ train với C tốt nhất, đo trên test
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=best_c, max_iter=5000, random_state=SEED))
    model.fit(tr[cols], tr[TARGET])
    pred = model.predict(te[cols])
    proba = model.predict_proba(te[cols])
    classes = model.classes_
    cm = confusion_matrix(te[TARGET], pred, labels=classes)

    # 3) CV 5-fold trên train (thể hiện độ dao động)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(model, tr[cols], tr[TARGET], cv=cv, scoring="accuracy")

    row = {
        "feature_set": feature_set, "n_features": len(cols), "best_C": best_c,
        "val_acc": round(best_acc * 100, 2),
        "accuracy": round(accuracy_score(te[TARGET], pred) * 100, 2),
        "f1_weighted": round(f1_score(te[TARGET], pred, average="weighted") * 100, 2),
        "sensitivity": round(recall_score(te[TARGET], pred, average="weighted") * 100, 2),
        "specificity": round(specificity_weighted(cm) * 100, 2),
        "brier": round(brier_multiclass(te[TARGET], proba, classes), 4),
        "cv_acc_mean": round(cv_scores.mean() * 100, 2),
        "cv_acc_sd": round(cv_scores.std() * 100, 2),
    }
    f1_cls = f1_score(te[TARGET], pred, labels=classes, average=None)
    per_class = {"feature_set": feature_set,
                 **{c: round(v * 100, 2) for c, v in zip(classes, f1_cls)}}
    return row, per_class

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    d = {n: load(n) for n in ["train", "test", "train_for_val", "test_for_val"]}

    sets = {"full": ALL16}
    top5_path = f"{SPLIT_DIR}/top5_features.json"
    if os.path.exists(top5_path):
        top5 = json.load(open(top5_path))
        sets.update({"chi2": top5["chi2"], "f_classify": top5["f_classify"],
                     "mutual_info": top5["mutual_info"]})
    else:
        print("Chưa có top5_features.json, chỉ chạy bộ full")

    rows, per_cls = [], []
    for name, cols in sets.items():
        r, p = run(name, cols, d)
        rows.append(r); per_cls.append(p)
        print(r)

    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/lr_results.csv", index=False)
    pd.DataFrame(per_cls).to_csv(f"{OUT_DIR}/lr_per_class_f1.csv", index=False)
    print("Đã lưu results/lr_results.csv và results/lr_per_class_f1.csv")