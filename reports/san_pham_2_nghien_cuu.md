# Sản phẩm 2 — Nghiên cứu

## Bài toán
Phân loại đa lớp mức độ béo phì (`NObeyesdad`) trên dataset Obesity Levels; tái lập pipeline và bảng kết quả của Maulana et al. (2024) trên CPU local.

## Phương pháp
- Preprocess theo Table 2 bài báo (`src/preprocess.py`)
- Baseline: LR, KNN, RF, Naive Bayes (GridSearchCV, Stratified 5-fold)
- Main: CatBoost với siêu tham số Table 3 (`lr=0.05`, `depth=8`, `l2_leaf_reg=5`, `iterations=300`)
- Đánh giá: accuracy / precision / recall / F1 (weighted), confusion matrix, feature importance

## Kết quả local (test set)

| Model | Acc (%) | Paper Acc (%) | Δ (pp) |
|-------|---------|---------------|--------|
| CatBoost | 95.51 | 95.98 | -0.47 |
| Random Forest | 94.56 | 95.04 | -0.48 |
| KNN | 87.71 | 88.18 | -0.47 |
| Logistic Regression | 93.14 | 65.48 | +27.66 |
| Naive Bayes | 52.01 | 94.07 | -42.06 |

Chi tiết: `results/all_results.csv`, `results/paper_comparison.csv`, `results/figures/`.

## Nhận xét
- CatBoost vẫn là mô hình tốt nhất trong thí nghiệm local, sát paper.
- RF và KNN cũng sát Table 4.
- LR cao hơn paper; NB thấp hơn nhiều — cần ghi nhận khác biệt preprocess/hyperparam khi viết thảo luận.

## Cách tái chạy
```bash
python -m src.download_data
python -m src.train
```
