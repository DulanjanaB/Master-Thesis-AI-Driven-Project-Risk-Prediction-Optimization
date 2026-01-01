# AI-Powered Lessons Learned Risk Analyzer (PoC)

An industry-style, thesis-safe Proof of Concept: convert **Lessons Learned (LL) text** + lightweight project metadata into a **risk prediction** and a short **explainability report**.

This repo uses **synthetic data only** (no company, confidential, or personal data).

## What this demo includes

- Synthetic dataset generator (metadata + LL text + label)
- Baseline model: TF-IDF (LL text) + numeric features → logistic regression risk classifier
- Run report in `reports/` with metrics and top drivers (coefficient-based)

## Quickstart (Windows PowerShell)
1) Create and activate a virtual env:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
pip install -e .
```

3) Generate synthetic data:

```powershell
risk-analyzer generate --out data/synthetic/projects.csv --n 600
```

4) Train + evaluate + write report:

```powershell
risk-analyzer train --data data/synthetic/projects.csv
```

Outputs:

- `data/synthetic/projects.csv`
- `models/model.joblib`
- `reports/run_<timestamp>.md`

## Why this is thesis-safe

- Uses synthetic LL examples and simulated project signals
- Focuses on methodology (NLP → features → model → explainability)
- Avoids organization-specific schemas, KPIs, or any proprietary assumptions

## Optional explainability (SHAP)

This PoC ships with coefficient-based explanations by default. If you want to experiment with SHAP later:

```powershell
pip install -e ".[explain]"
```

## Repo structure

- `src/risk_analyzer/` — library + CLI
- `data/synthetic/` — generated sample data
- `models/` — trained artifacts
- `reports/` — run outputs
- `docs/` — short documentation

## Next steps (easy extensions)

- Swap TF-IDF with embeddings (e.g., sentence-transformers)
- Add multi-class risk levels (Low/Medium/High)
- Add SHAP explanations (optional dependency)

