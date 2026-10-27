# AGENTS.md

## Project
Research Track midterm: phân loại béo phì trên dữ liệu Kaggle.
Remote: https://github.com/hai-dep-try/obesity-ml-research
Làm việc trong `obesity-ml-research/` (repo root), không phải thư mục cha `ttnt midterm`.
Đề bài: `docs/AI_Midterm_Project.pdf`. Bài báo tái lập: PDF CatBoost trong `docs/`.

## Layout (không invent thêm top-level dirs)
- `docs/` — PDF đề bài + bài báo
- `data/` — `python -m src.download_data` (Kaggle `fatemehmehrparvar/obesity-levels`); không commit dump nặng
- `notebooks/` — thứ tự: `01_eda` → `02_baseline` → `03_main_model`
- `src/` — `download_data.py`, `preprocess.py`, `train.py`, `evaluate.py`
- `results/` — CSV + `figures/`; regenerate, không sửa tay metrics
- `reports/` — `san_pham_1_khao_sat.md`, `san_pham_2_nghien_cuu.md`
- `ai_log/` — nhật ký AI bắt buộc; xem `ai_log/README.md`

## Conventions
- Ưu tiên sửa `src/` thay vì nhân đôi logic trong notebook
- Giữ số thứ tự notebook; đừng đổi tên `01_`/`02_`/`03_` nếu chưa cập nhật docs
- Metrics → `results/`; hình → `results/figures/`
- Tên report tiếng Việt là cố ý — đừng “normalize”
- Cập nhật `ai_log` cho mọi tương tác LLM substantive
- Target: `NObeyesdad`
- Thí nghiệm chạy **CPU local** rồi push GitHub (không bắt buộc Kaggle kernel)

## Setup / run
- `python -m venv .venv` rồi `pip install -r requirements.txt`
- `python -m src.download_data` nếu thiếu CSV trong `data/`
- `python -m src.train` để train baseline + CatBoost và ghi `results/`
- CatBoost best (paper): `lr=0.05, depth=8, l2_leaf_reg=5, iterations=300`

## Do not
- Commit CSV thô lớn dưới `data/`
- Coi CSV trong `results/` là source of truth cho model code
- Thêm cấu trúc project song song ở repo root
- Dùng câu trả lời AI thay cho citation bài báo gốc
