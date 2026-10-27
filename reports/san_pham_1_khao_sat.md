# Sản phẩm 1 — Khảo sát

## Mục tiêu
Khảo sát bài báo và phương pháp liên quan đến dự đoán mức độ béo phì bằng ML, tập trung vào gradient boosting / CatBoost.

## Bài báo chính
Maulana et al. (2024). *Predicting Obesity Levels with High Accuracy: Insights from a CatBoost Machine Learning Model*. Infolitika Journal of Data Science.
File: `docs/Predicting Obesity Levels with High Accuracy Insights from a CatBoost .pdf`

## Điểm chính
- Dataset 2111 mẫu, 17 thuộc tính, target `NObeyesdad` (7 lớp)
- Preprocess: impute, one-hot categorical, StandardScaler, split 80:20 stratified
- So sánh: CatBoost, Logistic Regression, KNN, Random Forest, Naive Bayes
- CatBoost tốt nhất trong paper: Acc 95.98%, Precision 96.08%, Recall 95.98%, F1 96.00%
- Feature importance nổi bật: Weight, Height, Gender (và thói quen ăn uống / vận động)

## Hướng tái lập trong repo
Xem `README.md` và `python -m src.train`. Kết quả local trong `results/`.
