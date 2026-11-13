import os
import json
import numpy as np
import pandas as pd
from skopt.space import Integer, Real

# --- D1. Định nghĩa không gian tìm kiếm (Search Space) ---
space = [
    Integer(20, 500, name="units"),                       # Số lượng nơ-ron ẩn
    Real(1e-4, 1e-1, prior="log-uniform", name="lr"),     # Tốc độ học (Learning Rate)
    Integer(20, 200, name="epochs"),                     # Số vòng lặp huấn luyện
    Integer(8, 32, name="batch")                         # Kích thước batch
]
