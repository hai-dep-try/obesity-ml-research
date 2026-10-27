# data/

## Dataset

- Source: [Obesity Levels (Kaggle)](https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels)
- Local files (gitignored):
  - `obesity.csv` — canonical path used by `src/preprocess.py`
  - `ObesityDataSet_raw_and_data_sinthetic.csv` — original Kaggle filename
- ~2111 rows × 17 columns; target: `NObeyesdad` (7 classes)

## Download

From repo root:

```bash
pip install -r requirements.txt
python -m src.download_data
```

Equivalent:

```python
import kagglehub

path = kagglehub.dataset_download("fatemehmehrparvar/obesity-levels")
print("Path to dataset files:", path)
```

Do **not** commit raw CSV dumps — regenerate with the script if missing.
