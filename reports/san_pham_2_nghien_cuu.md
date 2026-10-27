# Sản phẩm 2 — Báo cáo nghiên cứu và kết quả thực nghiệm

## 1. Giới thiệu

Nghiên cứu tái lập và kiểm chứng mô hình **CatBoost** để dự đoán mức độ béo phì theo Maulana et al. (2024), đồng thời so sánh với Logistic Regression, KNN, Random Forest và Naive Bayes trên tập Obesity Levels.

## 2. Phát biểu bài toán

Cho vector đặc trưng nhân khẩu học / lối sống \(x\), dự đoán nhãn \(y \in\) {Insufficient Weight, Normal Weight, Overweight Level I/II, Obesity Type I/II/III}.

**Mục tiêu:** đạt accuracy test gần mức paper (~95.98% với CatBoost) và phân tích đặc trưng quan trọng.

## 3. Dữ liệu và tiền xử lý

- 2111 mẫu, 17 cột; target `NObeyesdad`.
- Theo Table 2: impute mean/mode, one-hot categorical, StandardScaler, split **80:20 stratified** (`random_state=42`) → train 1688 / test 423.
- Triển khai: `src/preprocess.py`.

## 4. Phương pháp

- **CatBoost** (main): `learning_rate=0.05`, `depth=8`, `l2_leaf_reg=5`, `iterations=300` (CPU).
- **Baseline:** GridSearchCV + StratifiedKFold(5) trong `src/train.py`.
- **Đánh giá:** accuracy / precision / recall / F1 weighted; confusion matrix; feature importance.

## 5. Thiết kế thực nghiệm

```bash
python -m src.download_data
python -m src.train
```

Chạy trên **CPU local**. Kết quả: `results/`.

## 6. Kết quả (CPU local)

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Paper Acc (%) |
|-------|--------------|---------------|------------|--------|---------------|
| **CatBoost** | **95.51** | **95.60** | **95.51** | **95.53** | 95.98 |
| Random Forest | 94.56 | 95.07 | 94.56 | 94.67 | 95.04 |
| Logistic Regression | 93.14 | 93.12 | 93.14 | 93.10 | 65.48 |
| KNN | 87.71 | 87.29 | 87.71 | 87.24 | 88.18 |
| Naive Bayes | 53.66 | 57.13 | 53.66 | 50.41 | 94.07 |

Nguồn: `results/comparison_results.csv`, `results/main_results.csv`.

Hình: `results/figures/catboost_confusion_matrix.png`, `catboost_feature_importance.png`, `model_comparison.png`.

## 7. Thảo luận

- CatBoost của nhóm đạt **95.51%**, lệch **−0.47 pp** so với paper (95.98%) — cùng bậc độ chính xác; khác biệt có thể do seed, chi tiết encode binary, hoặc grid baseline.
- Thứ tự mô hình giống paper: CatBoost > RF > … ; KNN gần paper (88.18%).
- Logistic Regression của nhóm (~93%) cao hơn số paper (65.48%) — có thể paper dùng cấu hình/encode khác.
- Naive Bayes paper ghi Acc 94% nhưng Precision/Recall thấp (~64–72%); kết quả local (~54%) hợp lý hơn với GaussianNB trên one-hot + scaled features.
- Feature importance: xem `results/feature_importance_catboost.csv` (kỳ vọng Weight/Height/Gender nổi bật như Figure 3).

## 8. Hạn chế và hướng phát triển

- Dataset có tỷ lệ dữ liệu tổng hợp cao (~77% SMOTE).
- Chưa validate trên quần thể thực tế Việt Nam.
- Hướng tiếp: engineered BMI, cân bằng lớp, so sánh LightGBM/XGBoost, hoặc full grid Table 3 (`python -m src.train --tune-catboost`).

## 9. Tài liệu tham khảo

1. Maulana et al. (2024). Predicting Obesity Levels with High Accuracy… *Infolitika Journal of Data Science*. DOI: 10.60084/ijds.v2i1.195
2. Paleiro et al. — nguồn dataset Obesity Levels (UCI / Kaggle).
3. Đề bài AI Midterm — `docs/AI_Midterm_Project.pdf`
