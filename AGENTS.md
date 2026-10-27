# AGENTS.md

## Dự án
Research Track giữa kỳ: phân loại mức độ béo phì trên dữ liệu Kaggle.
Remote: https://github.com/hai-dep-try/obesity-ml-research
Làm việc trong `obesity-ml-research/` (repo root), không phải thư mục cha `ttnt midterm`.
Đề bài: `docs/AI_Midterm_Project.pdf` (Research Track + AI Usage Log).
Bài báo tái lập: `docs/Predicting Obesity Levels with High Accuracy Insights from a CatBoost .pdf` (Maulana et al., 2024).

## Layout (không tạo thêm thư mục top-level)
- `docs/` — PDF đề bài + bài báo
- `data/` — dataset qua `python -m src.download_data` (Kaggle: `fatemehmehrparvar/obesity-levels`); không commit CSV thô nặng
- `notebooks/` — thứ tự: `01_eda` → `02_baseline` → `03_main_model`
- `src/` — pipeline: `download_data.py`, `preprocess.py`, `train.py`, `evaluate.py`
- `results/` — CSV + `figures/`; regenerate, không sửa tay metrics
- `reports/` — deliverables (`san_pham_1_khao_sat.md`, `san_pham_2_nghien_cuu.md`)
- `ai_log/` — nhật ký AI (bắt buộc); xem `ai_log/README.md` và PDF §II

## Thí nghiệm (paper)
- Target: `NObeyesdad` (7 lớp); split 80:20 stratified; `RANDOM_STATE=42`
- Preprocess (Table 2): impute mean/mode; one-hot `Gender/CAEC/CALC/MTRANS`; binary yes/no → 0/1; `StandardScaler`
- Main: CatBoost (`lr=0.05, depth=8, l2_leaf_reg=5, iterations=300`)
- Baseline: Logistic Regression, KNN, Random Forest, Naive Bayes
- Chạy local CPU: `python -m src.train` (không dùng Kaggle kernel remote trừ khi user yêu cầu lại)

## Quy ước
- Ưu tiên sửa `src/` thay vì nhân đôi logic trong notebook
- Giữ số thứ tự notebook; đừng đổi tên `01_`/`02_`/`03_` mà không cập nhật docs
- Metrics → `results/`; hình → `results/figures/`
- Tên file báo cáo tiếng Việt là cố ý — đừng “chuẩn hóa”
- Cập nhật `ai_log` cho mọi tương tác LLM substantive

## Setup / chạy
- `python -m venv .venv` rồi `pip install -r requirements.txt`
- `python -m src.download_data` trước EDA nếu thiếu CSV
- `python -m src.train` để train + ghi kết quả
- Notebooks theo thứ tự EDA → baseline → main

## Không làm
- Commit file raw lớn dưới `data/`
- Coi CSV trong `results/` là source of truth cho code model
- Thêm cấu trúc dự án song song ở repo root
- Dùng câu trả lời AI thay cho trích dẫn bài báo gốc
