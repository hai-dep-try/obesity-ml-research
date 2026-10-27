# obesity-ml-research

Midterm ML research: obesity classification.

## Structure

```
data/          # local/linked datasets (do not commit heavy raw files)
notebooks/     # 01_eda → 02_baseline → 03_main_model
src/           # preprocess, train, evaluate
results/       # metrics CSVs + figures/
reports/       # san_pham_1_khao_sat.md, san_pham_2_nghien_cuu.md
ai_log/        # AI usage log
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Workflow

1. Put or link data under `data/`
2. Run notebooks in order (`01` → `02` → `03`)
3. Prefer `src/` for final train/eval; write outputs to `results/`

See `AGENTS.md` for agent conventions.
