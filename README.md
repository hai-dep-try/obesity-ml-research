# obesity-ml-research

Midterm ML research (**Research Track**): obesity level classification from Kaggle.

Assignment brief: `docs/AI_Midterm_Project.pdf`

## Structure

```
data/          # local CSVs (gitignored) — see data/README.md
docs/          # course PDF requirements
notebooks/     # 01_eda → 02_baseline → 03_main_model
src/           # download_data, preprocess, train, evaluate
results/       # metrics CSVs + figures/
reports/       # san_pham_1_khao_sat.md, san_pham_2_nghien_cuu.md
ai_log/        # AI usage log (Sản phẩm 3)
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\\activate
pip install -r requirements.txt
python -m src.download_data
python -m src.preprocess
```

Dataset: [Obesity Levels](https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels) → `data/ObesityDataSet_raw_and_data_sinthetic.csv` (also aliased as `data/obesity.csv`).

## Workflow

1. Download data if CSV missing under `data/`
2. Run notebooks in order (`01` → `02` → `03`)
3. Prefer `src/` for final train/eval; write outputs to `results/`

See `AGENTS.md` for agent conventions.
