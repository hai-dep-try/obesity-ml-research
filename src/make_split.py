import os
import pandas as pd
from sklearn.model_selection import train_test_split

CSV = "data/ObesityDataSet_raw_and_data_sinthetic.csv"
OUT = "data/splits"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)

train, test = train_test_split(df, test_size=0.25,
                               stratify=df["NObeyesdad"], random_state=42)
tr_val, te_val = train_test_split(train, test_size=0.20,
                                  stratify=train["NObeyesdad"], random_state=42)

for name, d in [("train", train), ("test", test),
                ("train_for_val", tr_val), ("test_for_val", te_val)]:
    d.to_csv(f"{OUT}/{name}.csv", index=False)
    print(name, d.shape)