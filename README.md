# obesity-ml-research

Đồ án giữa kỳ **Research Track**: phân loại mức độ béo phì (Kaggle), tái lập thí nghiệm bài báo CatBoost (Maulana et al., 2024) trên **CPU local**.

- Đề bài: `docs/AI_Midterm_Project.pdf`
- Bài báo: `docs/Predicting Obesity Levels with High Accuracy Insights from a CatBoost .pdf`
- Dataset: [Obesity Levels](https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels) (2111×17, target `NObeyesdad`)

## Cấu trúc

```
data/          # CSV cục bộ (gitignore) — xem data/README.md
docs/          # PDF đề bài + bài báo
notebooks/     # 01_eda → 02_baseline → 03_main_model
src/           # download_data, preprocess, train, evaluate
results/       # CSV metrics + figures/
reports/       # san_pham_1_khao_sat.md, san_pham_2_nghien_cuu.md
ai_log/        # Nhật ký sử dụng AI (Sản phẩm 3)
```

## Cài đặt

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.download_data
```

## Chạy thí nghiệm (CPU)

Pipeline theo bài báo (Table 2–4):

1. Impute missing (mean/mode)
2. One-hot: `Gender`, `CAEC`, `CALC`, `MTRANS`; binary yes/no → 0/1
3. `StandardScaler` cho biến liên tục
4. Split **80:20 stratified** theo `NObeyesdad`
5. Baseline (GridSearchCV + Stratified 5-fold): LR, KNN, RF, Naive Bayes
6. Main: **CatBoost** với tham số paper (`lr=0.05`, `depth=8`, `l2_leaf_reg=5`, `iterations=300`)

```bash
python -m src.preprocess
python -m src.train
# tuỳ chọn (chậm): python -m src.train --tune-catboost
```

Kết quả ghi vào `results/` (`baseline_results.csv`, `main_results.csv`, `all_results.csv`, `paper_comparison.csv`, `figures/`).

## Kết quả local (so với paper Table 4)

| Model | Local Acc (%) | Paper Acc (%) |
|-------|---------------|---------------|
| CatBoost | 95.51 | 95.98 |
| Random Forest | 94.56 | 95.04 |
| KNN | 87.71 | 88.18 |
| Logistic Regression | 93.14 | 65.48 |
| Naive Bayes | 52.01 | 94.07 |

CatBoost / RF / KNN bám sát paper (~0.5 điểm phần trăm). Chênh LR và NB có thể do khác cách encode/tune; xem `results/paper_comparison.csv`.

## Notebook

Chạy theo thứ tự `01_eda` → `02_baseline` → `03_main_model`. Logic tái sử dụng ưu tiên nằm trong `src/`.

## Quy ước agent

Xem `AGENTS.md`. **Không** commit CSV thô trong `data/`.
