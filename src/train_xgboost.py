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