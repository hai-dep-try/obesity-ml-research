# Sản phẩm 2 — Nghiên cứu

## Mục tiêu
Tái lập thí nghiệm Maulana et al. (2024): CatBoost vs các baseline trên dataset Obesity Levels; chạy trên **CPU local**, lưu kết quả vào `results/`.

## Dữ liệu & preprocess
- Nguồn: Kaggle `fatemehmehrparvar/obesity-levels` (`python -m src.download_data`)
- Pipeline: `src/preprocess.py` (Table 2): impute mean/mode, one-hot Gender/CAEC/CALC/MTRANS, binary yes/no→0/1, StandardScaler, split 80:20 stratified, `random_state=42`
- Train 1688 / Test 423 mẫu; 27 feature sau encode

## Mô hình
| Vai trò | Model |
|--------|--------|
| Main | CatBoost (`lr=0.05, depth=8, l2_leaf_reg=5, iterations=300`) |
| Baseline | Logistic Regression, KNN, Random Forest, Naive Bayes (GridSearchCV + Stratified 5-fold) |

Chạy: `python -m src.train`  
(Tuỳ chọn lưới nhỏ CatBoost: `python -m src.train --tune-catboost`)

## Kết quả (CPU local)

Nguồn: `results/all_results.csv`, `results/paper_comparison.csv` (sau `python -m src.train`).

| Model | Paper Acc (%) | Repo Acc (%) | Δ (pp) | Repo F1 (%) |
|-------|---------------|--------------|--------|-------------|
| CatBoost | 95.98 | **95.51** | −0.47 | 95.53 |
| Logistic Regression | 65.48 | 93.14 | +27.66 | 93.10 |
| KNN | 88.18 | 84.40 | −3.78 | 83.52 |
| Random Forest | 95.04 | 94.33 | −0.71 | 94.44 |
| Naive Bayes | 94.07 | 52.25 | −41.82 | 48.53 |

Hình: `results/figures/confusion_matrix_catboost.png`, `feature_importance_catboost.png`, `model_comparison.png`.

### Feature importance (top)
Weight ≫ Height ≫ FCVC, Age, Gender — khớp hướng paper (Weight/Height/Gender nổi bật). Chi tiết: `results/feature_importance_catboost.csv`.

## Thảo luận
- **CatBoost** gần paper (−0.47 pp): tái lập thành công trên CPU với hyperparam Table 3.
- **Random Forest** cũng sát paper (−0.71 pp).
- **Logistic Regression** cao hơn paper rõ: có thể do encode/scale hoặc lưới `C` khác paper (paper không công bố đủ chi tiết baseline).
- **Naive Bayes** thấp hơn paper nhiều: GaussianNB trên vector one-hot + scaled liên tục thường kém; paper ghi Acc 94.07 nhưng Precision/Recall/F1 ~66% — có thể khác metric/setup.
- Dataset ~77% synthetic (SMOTE) — hạn chế khái quát hóa.

## Kết luận
Trên setup local này, **CatBoost dẫn đầu (95.51%)**, sát mức 95.98% của paper, và vượt các baseline trừ khi so với LR đã được tune mạnh trên feature đã scale. Pipeline + số liệu đủ để nộp Research Track kèm `ai_log/`.

## Tài liệu tham chiếu
1. Maulana et al. (2024) — `docs/Predicting Obesity Levels with High Accuracy Insights from a CatBoost .pdf`
2. Đề bài — `docs/AI_Midterm_Project.pdf`
3. Dataset Kaggle Obesity Levels
