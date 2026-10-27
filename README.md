# obesity-ml-research

Đề tài giữa kỳ (**Research Track**): phân loại mức độ béo phì từ dữ liệu Kaggle, tái lập thí nghiệm theo bài báo CatBoost (Maulana et al., 2024).

- Đề bài khóa học: `docs/AI_Midterm_Project.pdf`
- Bài báo chính: `docs/Predicting Obesity Levels with High Accuracy Insights from a CatBoost .pdf`

## Cấu trúc

```
data/          # CSV cục bộ (gitignore) — xem data/README.md
docs/          # PDF đề bài + bài báo
notebooks/     # 01_eda → 02_baseline → 03_main_model
src/           # download_data, preprocess, train, evaluate
results/       # CSV metrics + figures/
reports/       # san_pham_1_khao_sat.md, san_pham_2_nghien_cuu.md
ai_log/        # Nhật ký dùng AI (Sản phẩm 3)
```

## Cài đặt

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.download_data
```

Dataset: [Obesity Levels](https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels) → `data/ObesityDataSet_raw_and_data_sinthetic.csv` (alias `data/obesity.csv`).

## Chạy thí nghiệm (CPU local)

```bash
# Baselines (grid + 5-fold) + CatBoost (hyperparam Table 3), ghi results/
python -m src.train

# (Tuỳ chọn) grid nhỏ thêm cho CatBoost
python -m src.train --tune-catboost
```

Kết quả mong đợi (paper Table 4, test set): CatBoost ~95.98% accuracy; so sánh LR / KNN / RF / Naive Bayes.

Output chính:

- `results/baseline_results.csv`, `results/main_results.csv`
- `results/all_models_results.csv`, `results/paper_comparison.csv`
- `results/figures/catboost_confusion_matrix.png`
- `results/figures/catboost_feature_importance.png`

## Workflow

1. Tải data nếu thiếu CSV trong `data/`
2. Chạy `python -m src.train` (hoặc notebooks `01` → `02` → `03`)
3. Ưu tiên `src/` cho train/eval cuối; ghi output vào `results/`
4. Điền `reports/` và `ai_log/` theo yêu cầu đề bài

Xem thêm `AGENTS.md` cho quy ước làm việc trong repo.
