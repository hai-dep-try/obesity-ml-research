import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score

from lr_baseline import load, ALL16, TARGET

N_CLASSES = 7


def build_model(n_features, units, lr, n_classes=N_CLASSES):
    """ANN 3 tầng: input -> Dense(ReLU) -> BN -> softmax."""
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
    """Mã hóa toàn bộ cột dạng chữ sang số, sau đó chuẩn hóa bằng StandardScaler."""
    X_tr = train_df[cols].copy()
    X_ev = eval_df[cols].copy()

    # Bắt chính xác TẤT CẢ cột không phải số (string, object, category...)
    cat_cols = X_tr.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        # Ép kiểu rõ ràng sang float sau khi encode để tránh pandas giữ kiểu string/object
        encoded_tr = oe.fit_transform(X_tr[cat_cols])
        encoded_ev = oe.transform(X_ev[cat_cols])
        
        for idx, col in enumerate(cat_cols):
            X_tr[col] = encoded_tr[:, idx].astype(float)
            X_ev[col] = encoded_ev[:, idx].astype(float)

    # Đảm bảo toàn bộ bảng là mảng số thực float
    X_tr_values = X_tr.astype(float).values
    X_ev_values = X_ev.astype(float).values

    # Chuẩn hóa StandardScaler
    scaler = StandardScaler().fit(X_tr_values)
    X_tr_scaled = scaler.transform(X_tr_values)
    X_ev_scaled = scaler.transform(X_ev_values)

    # Mã hóa nhãn (Target)
    le = LabelEncoder().fit(train_df[TARGET])
    y_tr = le.transform(train_df[TARGET])
    y_ev = le.transform(eval_df[TARGET])

    return X_tr_scaled, y_tr, X_ev_scaled, y_ev, le.classes_


def train_and_eval(cols, params, train_df, eval_df, seed=1, verbose=0):
    """params = {"units":..., "lr":..., "epoch":..., "batch":...}"""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    X_tr, y_tr, X_ev, y_ev, classes = prepare(train_df, eval_df, cols)
    
    model = build_model(X_tr.shape[1], int(params["units"]), float(params["lr"]))
    model.fit(
        X_tr, to_categorical(y_tr, N_CLASSES),
        epochs=int(params["epoch"]),
        batch_size=int(params["batch"]),
        shuffle=True,
        verbose=verbose
    )

    proba = model.predict(X_ev, verbose=0)
    y_pred = proba.argmax(axis=1)
    acc = float(accuracy_score(y_ev, y_pred))
    
    return {
        "acc": acc,
        "proba": proba,
        "y_true": y_ev,
        "y_pred": y_pred,
        "classes": classes
    }


