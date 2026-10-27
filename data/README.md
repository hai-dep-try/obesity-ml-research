# data/

Dataset **Obesity Levels** (Kaggle: `fatemehmehrparvar/obesity-levels`).

- File cục bộ (gitignore): `obesity.csv` và `ObesityDataSet_raw_and_data_sinthetic.csv`
- Shape: 2111 dòng × 17 cột; nhãn: `NObeyesdad` (7 lớp)
- **Không** commit dump raw lớn — tải lại local khi cần

## Tải dữ liệu

```bash
pip install -r requirements.txt
python -m src.download_data
```

Hoặc thủ công:

```python
import kagglehub
from pathlib import Path
import shutil

path = kagglehub.dataset_download("fatemehmehrparvar/obesity-levels")
print("Path to dataset files:", path)

src = Path(path) / "ObesityDataSet_raw_and_data_sinthetic.csv"
dst = Path("data")  # chạy từ repo root
shutil.copy2(src, dst / "ObesityDataSet_raw_and_data_sinthetic.csv")
shutil.copy2(src, dst / "obesity.csv")
```

Lần đầu Kaggle có thể yêu cầu xác thực (`kagglehub` / Kaggle API credentials).
