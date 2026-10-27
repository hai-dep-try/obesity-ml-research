# Sản phẩm 2 — Nghiên cứu

## 1. Bài toán

Phân loại 7 mức độ béo phì (`NObeyesdad`) từ đặc trưng nhân khẩu + thói quen ăn uống / vận động (dataset Obesity Levels, 2111×17).

## 2. Phương pháp tái lập

Theo Maulana et al. (2024):

1. Preprocess (`src/preprocess.py`): impute, one-hot, binary encode, StandardScaler, split 80:20 stratified (`RANDOM_STATE=42`)
2. Baselines: Logistic Regression, KNN, Random Forest, Naive Bayes (grid search + Stratified 5-fold)
3. Main: CatBoost với hyperparam Table 3 (`lr=0.05`, `depth=8`, `l2_leaf_reg=5`, `iterations=300`)
4. Đánh giá: accuracy / precision / recall / F1 (weighted); confusion matrix; feature importance → `results/`

Chạy (CPU local): `python -m src.train`

## 3. Kết quả thực nghiệm (local CPU)

| Model | Acc repo (%) | Acc paper (%) | Δ (pp) | F1 repo (%) |
|---|---:|---:|---:|---:|
| **CatBoost** | **95.51** | 95.98 | −0.47 | 95.53 |
| Random Forest | 94.33 | 95.04 | −0.71 | 94.44 |
| Logistic Regression | 93.14 | 65.48 | +27.66 | 93.10 |
| KNN | 84.40 | 88.18 | −3.78 | 83.52 |
| Naive Bayes | 52.25 | 94.07 | −41.82 | 48.53 |

Nguồn CSV: `results/paper_comparison.csv`, `results/main_results.csv`, `results/baseline_results.csv`.

Hình:

- `results/figures/catboost_confusion_matrix.png`
- `results/figures/catboost_feature_importance.png`
- `results/figures/model_comparison.png`

## 4. Thảo luận

- **CatBoost** gần paper (−0.47 pp): cùng hyperparam Table 3; lệch nhỏ có thể do seed / chi tiết encode binary / thứ tự cột one-hot.
- **Random Forest** cũng gần paper (−0.71 pp).
- **Logistic Regression** cao hơn paper nhiều (+27.66): có thể paper dùng cấu hình/solver khác hoặc preprocess khác trên biến binary.
- **Naive Bayes** thấp hơn rõ (−41.82): GaussianNB trên features đã one-hot + scale thường kém; paper có thể dùng biến thể / encoding khác.
- Feature importance: Weight / Height / Gender thường đứng đầu (liên quan BMI), phù hợp Figure 3 của paper.
- Hạn chế dataset: ~77% synthetic (SMOTE); cần thận trọng khi tổng quát hóa.

## 5. Tài liệu tham chiếu

Maulana, A., Afidh, R. P. F., Maulydia, N. B., Idroes, G. M., & Rahimah, S. (2024). Predicting Obesity Levels with High Accuracy: Insights from a CatBoost Machine Learning Model. *Infolitika Journal of Data Science*, 2(1). DOI: 10.60084/ijds.v2i1.195
