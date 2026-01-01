# Architecture (PoC)

## Goal

Convert historical *Lessons Learned* style text + simple project metadata into a baseline risk prediction and an explainable summary.

## Pipeline

1) **Data generation (synthetic)**
   - Numeric signals (duration, milestone delay, budget variance)
   - LL text snippets sampled from a small theme library
   - Label generation rule that correlates risk with certain themes + numeric stress

2) **Feature engineering**
   - Numeric features: passed through as-is
   - Text features: TF-IDF vectorization on LL text

3) **Model**
   - Logistic regression classifier

4) **Explainability**
   - Coefficient-based drivers (top positive/negative weights)
   - Optional: SHAP (if installed via `pip install .[explain]`)

5) **Outputs**
   - `models/` trained pipeline artifact
   - `reports/` metrics + top drivers
