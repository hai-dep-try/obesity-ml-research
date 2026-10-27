# Sản phẩm 1 — Khảo sát / Literature review

## Mục tiêu
Tóm tắt bối cảnh phân loại mức độ béo phì bằng ML và định vị bài báo tái lập trong repo.

## Nguồn chính
- **Maulana et al. (2024).** *Predicting Obesity Levels with High Accuracy: Insights from a CatBoost Machine Learning Model.* Infolitika Journal of Data Science, 2(1). DOI: [10.60084/ijds.v2i1.195](https://doi.org/10.60084/ijds.v2i1.195)  
  File local: `docs/Predicting Obesity Levels with High Accuracy Insights from a CatBoost .pdf`
- Đề bài khóa học: `docs/AI_Midterm_Project.pdf` (Research Track)

## Dataset
- Kaggle: `fatemehmehrparvar/obesity-levels` (2111 mẫu, 17 thuộc tính, nhãn `NObeyesdad` — 7 lớp)
- ~77% dữ liệu sinh synthetically (SMOTE/Weka); phần còn lại thu thập từ người dùng

## Phương pháp trong paper (tóm tắt)
| Hạng mục | Chi tiết |
|---|---|
| Preprocess | Impute mean/mode; one-hot Gender/CAEC/CALC/MTRANS; StandardScaler; split 80:20 stratified |
| Main model | CatBoost (best: `lr=0.05`, `depth=8`, `l2_leaf_reg=5`, `iterations=300`) |
| Baselines | Logistic Regression, KNN, Random Forest, Naive Bayes |
| Metrics | Accuracy, Precision, Recall, F1 (weighted) |

## Kết quả paper (Table 4, test)
| Model | Acc (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---:|---:|---:|---:|
| CatBoost | 95.98 | 96.08 | 95.98 | 96.00 |
| Random Forest | 95.04 | 95.04 | 95.04 | 95.00 |
| Naive Bayes | 94.07 | 72.13 | 64.07 | 66.17 |
| KNN | 88.18 | 91.25 | 88.18 | 89.07 |
| Logistic Regression | 65.48 | 68.60 | 65.48 | 66.60 |

Feature importance (paper): Weight, Height, Gender nổi bật nhất.

## Gap / hướng tái lập trong repo
- Tái lập preprocess + 5 model trên CPU local (`python -m src.train`)
- Ghi metrics/figures vào `results/`; so sánh với Table 4
- Báo cáo nghiên cứu: `reports/san_pham_2_nghien_cuu.md`
