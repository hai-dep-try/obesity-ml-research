import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from lr_baseline import load, ALL16, TARGET   # dùng lại phần xử lý dữ liệu

N_CLASSES = 7


def build_model(n_features, units, lr, n_classes=N_CLASSES):
    """ANN 3 tầng theo Yagin et al. (2023): input -> Dense(ReLU) -> BN -> softmax."""
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(units, activation="relu",
                     kernel_initializer=initializers.GlorotUniform(seed=1)),
        layers.BatchNormalization(),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, beta_1=0.95, beta_2=0.99),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare(train_df, eval_df, cols):
    """Fit scaler + label encoder CHỈ trên train_df, áp dụng cho eval_df."""
    scaler = StandardScaler().fit(train_df[cols])
    le = LabelEncoder().fit(train_df[TARGET])      # classes_ theo thứ tự chữ cái
    X_tr = scaler.transform(train_df[cols])
    X_ev = scaler.transform(eval_df[cols])
    y_tr = le.transform(train_df[TARGET])
    y_ev = le.transform(eval_df[TARGET])
    return X_tr, y_tr, X_ev, y_ev, le.classes_


def train_and_eval(cols, params, train_df, eval_df, seed=1, verbose=0):
    """params = {"units":..., "lr":..., "epoch":..., "batch":...}.
    Trả về dict: acc, proba, y_true, y_pred, classes."""
    tf.keras.backend.clear_session()        # tránh tràn RAM khi lặp hàng trăm lần
    tf.keras.utils.set_random_seed(seed)

    X_tr, y_tr, X_ev, y_ev, classes = prepare(train_df, eval_df, cols)
    model = build_model(len(cols), int(params["units"]), float(params["lr"]))
    model.fit(X_tr, to_categorical(y_tr, N_CLASSES),
              epochs=int(params["epoch"]), batch_size=int(params["batch"]),
              shuffle=True, verbose=verbose)

    proba = model.predict(X_ev, verbose=0)
    y_pred = proba.argmax(axis=1)
    return {"acc": accuracy_score(y_ev, y_pred), "proba": proba,
            "y_true": y_ev, "y_pred": y_pred, "classes": classes}


if __name__ == "__main__":
    trv, tev = load("train_for_val"), load("test_for_val")

    # In kiến trúc, đối chiếu số tham số
    m = build_model(len(ALL16), 30, 0.013)
    m.summary()

    # Smoke test: ít epoch để chạy nhanh
    params = {"units": 30, "lr": 0.013, "epoch": 50, "batch": 16}
    t0 = time.time()
    r = train_and_eval(ALL16, params, trv, tev, seed=1)
    dt = time.time() - t0
    print(f"val acc = {r['acc']*100:.2f}% | thời gian = {dt:.1f}s "
          f"| ~{dt/params['epoch']:.3f}s/epoch")