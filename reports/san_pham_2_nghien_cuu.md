# Sản phẩm 2 — Nghiên cứu

## Mục tiêu

Tái lập thí nghiệm phân loại mức độ béo phì theo Maulana et al. (2024) trên **CPU local**, dùng dataset Kaggle Obesity Levels, lưu metrics/figures vào `results/`, rồi đẩy code lên GitHub.

## Thiết lập thí nghiệm

- Preprocess: `src/preprocess.py` (Table 2 paper) — impute, one-hot Gender/CAEC/CALC/MTRANS, StandardScaler, split **80:20 stratified** (`random_state=42`)
- Train/eval: `src/train.py`, `src/evaluate.py`
- Lệnh: `python -m src.train`
- CatBoost dùng hyperparameters tốt nhất paper: `lr=0.05`, `depth=8`, `l2_leaf_reg=5`, `iterations=300`
- Baseline: GridSearchCV + stratified 5-fold (LR, KNN, RF, Naive Bayes)

## Kết quả local (CPU)

| Model | Acc local (%) | Acc paper (%) | Δ (pp) |
|---|---:|---:|---:|
| **CatBoost** | **95.51** | 95.98 | −0.47 |
| Random Forest | 94.56 | 95.04 | −0.48 |
| Logistic Regression | 93.14 | 65.48 | +27.66 |
| KNN | 87.71 | 88.18 | −0.47 |
| Naive Bayes | 52.01 | 94.07* | −42.06 |

\* Table 4 paper ghi NB accuracy 94.07% nhưng precision/recall/F1 (~72/64/66) không nhất quán với accuracy — kết quả local phản ánh NB yếu trên feature đã scale/one-hot.

Nguồn: `results/model_comparison.csv`, `results/main_results.csv`, `results/baseline_results.csv`.

## Phân tích

- Confusion matrix: `results/figures/catboost_confusion_matrix.png`
- Feature importance: `results/figures/catboost_feature_importance.png` / `results/feature_importance_catboost.csv`
- CatBoost vẫn là model tốt nhất local, sát paper (~0.5 điểm phần trăm)
- So sánh biểu đồ: `results/figures/model_comparison.png`

## Giới hạn

- Random seed / phiên bản thư viện / lưới hyperparameter baseline khác paper → lệch nhẹ
- Dataset ~77% synthetic (SMOTE) — generalization thực tế cần thận trọng
- Chạy CPU local; không dùng Kaggle kernel/GPU

## Tài liệu tham chiếu

1. Maulana et al. (2024), Infolitika Journal of Data Science — `docs/Predicting Obesity Levels with High Accuracy Insights from a CatBoost .pdf`
2. Dataset: https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels
3. Đề bài môn: `docs/AI_Midterm_Project.pdf`
