# data/

Dataset **Obesity Levels** (Kaggle: `fatemehmehrparvar/obesity-levels`).

- File cục bộ (gitignore): `ObesityDataSet_raw_and_data_sinthetic.csv` và alias `obesity.csv`
- Shape: 2111 dòng × 17 cột; target: `NObeyesdad` (7 lớp)
- **Không** commit dump thô — tải lại khi cần

## Tải dữ liệu

```bash
pip install kagglehub
python -m src.download_data
```

Hoặc:

```python
import kagglehub
path = kagglehub.dataset_download("fatemehmehrparvar/obesity-levels")
print("Path to dataset files:", path)
```

Lần đầu có thể cần đăng nhập Kaggle API (`kagglehub` / credentials).
