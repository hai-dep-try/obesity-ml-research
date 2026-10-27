# AGENTS.md

## Project
Research Track midterm: obesity classification on Kaggle health data.
Remote: https://github.com/hai-dep-try/obesity-ml-research
Work inside `obesity-ml-research/` (repo root), not the parent `ttnt midterm` folder.
Course brief: `docs/AI_Midterm_Project.pdf` (Research Track + AI Usage Log rules).

## Layout (do not invent extra top-level dirs)
- `docs/` — assignment PDF from instructors
- `data/` — dataset via `python -m src.download_data` (Kaggle: `fatemehmehrparvar/obesity-levels`); do not commit heavy raw dumps
- `notebooks/` — ordered: `01_eda` → `02_baseline` → `03_main_model`
- `src/` — reusable pipeline: `download_data.py`, `preprocess.py`, `train.py`, `evaluate.py`
- `results/` — CSVs + `figures/`; regenerate rather than hand-edit metrics
- `reports/` — deliverables (`san_pham_1_khao_sat.md`, `san_pham_2_nghien_cuu.md`)
- `ai_log/` — AI usage log (required product); see `ai_log/README.md` and PDF §II

## Conventions
- Prefer changing `src/` over duplicating logic inside notebooks
- Keep notebook numbering; don’t rename `01_`/`02_`/`03_` without updating docs
- Write metrics to `results/`; figures to `results/figures/`
- Vietnamese report filenames are intentional — don’t “normalize” them
- Update `ai_log` for every substantive LLM interaction (full prompt/response + accept/reject + verification)
- Target label column: `NObeyesdad`

## Setup / run
- `python -m venv .venv` then `pip install -r requirements.txt`
- `python -m src.download_data` before EDA if CSV missing under `data/`
- Run notebooks in order for EDA → baseline → main
- Prefer `src/` scripts for final train/eval results over one-off notebook training

## Do not
- Commit large raw data files under `data/`
- Treat `results/` CSVs as source of truth for model code
- Add a second parallel project structure at repo root
- Use AI answers as a substitute for citing original research papers
