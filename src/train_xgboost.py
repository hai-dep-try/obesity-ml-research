import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

print("--- 1. Khai báo thư viện và kiểm tra file dữ liệu ---")
csv_path = 'data/ObesityDataSet_raw_and_data_sinthetic.csv'
df = pd.read_csv(csv_path)
print(f"Đã load dữ liệu thành công: {df.shape}")

# --- BƯỚC 2: Mã hóa & Huấn luyện mô hình ---
target_col = 'NObeyesdad' if 'NObeyesdad' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("--- 2. Đang huấn luyện mô hình XGBoost ---")
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)

# --- BƯỚC 3: Dự đoán và Tính chỉ số để kiểm tra ---
print("--- 3. Đang đánh giá kết quả mô hình ---")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
_, _, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

cm = confusion_matrix(y_test, y_pred)
fp = cm.sum(axis=0) - np.diag(cm)
fn = cm.sum(axis=1) - np.diag(cm)
tp = np.diag(cm)
tn = cm.sum() - (fp + fn + tp)

sensitivity = np.mean(tp / (tp + fn + 1e-10))
specificity = np.mean(tn / (tn + fp + 1e-10))

print("\n========== KẾT QUẢ ĐÁNH GIÁ XGBOOST ==========")
print(f"Accuracy    : {acc:.4f}")
print(f"F1-Macro    : {f1:.4f}")
print(f"Sensitivity : {sensitivity:.4f}")
print(f"Specificity : {specificity:.4f}")
print("==============================================")