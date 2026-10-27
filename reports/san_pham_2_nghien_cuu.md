# Sản phẩm 2 — Báo cáo nghiên cứu

## 1. Đặt vấn đề
Phân loại mức độ béo phì (7 lớp `NObeyesdad`) từ đặc trưng nhân khẩu học, lối sống và thói quen ăn uống. Repo tái lập thí nghiệm của Maulana et al. (2024) với CatBoost làm mô hình chính và 4 baseline.

## 2. Dữ liệu & tiền xử lý
- Nguồn: Kaggle `fatemehmehrparvar/obesity-levels` (2111 × 17)
- Pipeline (`src/preprocess.py`, theo Table 2 paper):
  - Impute mean (liên tục) / mode (phân loại) nếu thiếu < 5%
  - Binary yes/no → 0/1
  - One-hot: `Gender`, `CAEC`, `CALC`, `MTRANS`
  - `StandardScaler` cho biến liên tục + binary đã mã hóa
  - Split **80:20 stratified**, `random_state=42`

## 3. Mô hình
| Vai trò | Thuật toán | Ghi chú |
|---|---|---|
| Main | CatBoost | Best paper: `lr=0.05`, `depth=8`, `l2_leaf_reg=5`, `iterations=300` |
| Baseline | Logistic Regression, KNN, Random Forest, Naive Bayes | Grid search + Stratified 5-fold (CPU) |

Chạy: `python -m src.train` (local CPU).

## 4. Kết quả
Số liệu chạy local được ghi vào:

- `results/baseline_results.csv` — 4 baseline
- `results/main_results.csv` — CatBoost
- `results/all_models_results.csv` — bảng so sánh kiểu Table 4
- `results/figures/catboost_confusion_matrix.png`
- `results/figures/catboost_feature_importance.png`
- `results/catboost_feature_importance.csv`

**Tham chiếu paper (Table 4):** CatBoost Acc ≈ **95.98%**. Kết quả local có thể lệch nhẹ do seed / triển khai grid baseline.

## 5. Thảo luận ngắn
- CatBoost thường vượt baseline nhờ xử lý tốt đặc trưng phân loại và regularization.
- Feature importance thường nhấn mạnh **Weight / Height / Gender** (gần với BMI) — khớp paper Figure 3.
- Hạn chế: ~77% dữ liệu synthetic; cần thận trọng khi suy diễn lâm sàng.

## 6. Tài liệu tham khảo
1. Maulana, A., et al. (2024). Predicting Obesity Levels with High Accuracy: Insights from a CatBoost Machine Learning Model. *Infolitika Journal of Data Science*, 2(1). DOI: 10.60084/ijds.v2i1.195
2. Đề bài AI Midterm — `docs/AI_Midterm_Project.pdf`
