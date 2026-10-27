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
|-------|--------:|--------------:|-------:|
| **CatBoost** | **95.51** | 95.98 | −0.47 |
| Random Forest | 94.33 | 95.04 | −0.71 |
| Logistic Regression | 93.14 | 65.48 | +27.66 |
| KNN | 84.40 | 88.18 | −3.78 |
| Naive Bayes | 52.25 | 94.07* | −41.82 |

\* Table 4 paper ghi NB Acc 94.07% nhưng Precision/Recall/F1 (~72/64/66) không nhất quán với accuracy.

Chi tiết: `results/all_results.csv`, `results/main_results.csv`, `results/figures/`.

## Nhận xét
- CatBoost vẫn tốt nhất local, sát paper (lệch −0.47 pp).
- RF gần Table 4; KNN hơi thấp hơn paper.
- LR cao hơn paper; NB thấp hơn nhiều trên feature đã scale/one-hot — ghi nhận khi thảo luận.
- Chạy CPU local; không dùng Kaggle kernel/GPU.

## Cách tái chạy
```bash
python -m src.download_data
python -m src.train
```

## Tài liệu
1. Maulana et al. (2024) — `docs/Predicting Obesity Levels with High Accuracy Insights from a CatBoost .pdf`
2. Đề bài: `docs/AI_Midterm_Project.pdf`
