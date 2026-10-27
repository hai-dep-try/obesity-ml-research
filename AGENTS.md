# AGENTS.md

## Project
ML research repo for obesity classification (course midterm).
Remote: https://github.com/hai-dep-try/obesity-ml-research
Work inside `obesity-ml-research/` (repo root), not the parent `ttnt midterm` folder.

## Layout (do not invent extra top-level dirs)
- `data/` — raw/heavy datasets stay local or linked; do not commit large originals
- `notebooks/` — ordered: `01_eda` → `02_baseline` → `03_main_model`
- `src/` — reusable pipeline: `preprocess.py`, `train.py`, `evaluate.py`
- `results/` — CSVs + `figures/`; regenerate rather than hand-edit metrics
- `reports/` — deliverables (`san_pham_1_khao_sat.md`, `san_pham_2_nghien_cuu.md`)
- `ai_log/` — AI usage log; see `ai_log/README.md`

## Conventions
- Prefer changing `src/` over duplicating logic inside notebooks
- Keep notebook numbering; don’t rename `01_`/`02_`/`03_` without updating docs
- Write metrics to `results/`; figures to `results/figures/`
- Vietnamese report filenames are intentional — don’t “normalize” them
- Update `ai_log` when using AI for substantive work

## Setup / run
- `python -m venv .venv` then `pip install -r requirements.txt`
- Run notebooks in order for EDA → baseline → main
- Prefer `src/` scripts for final train/eval results over one-off notebook training

## Do not
- Commit large raw data files under `data/`
- Treat `results/` CSVs as source of truth for model code
- Add a second parallel project structure at repo root
