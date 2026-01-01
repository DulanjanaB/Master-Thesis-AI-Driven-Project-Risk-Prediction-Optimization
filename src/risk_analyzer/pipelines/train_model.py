from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from risk_analyzer.utils.reporting import utc_timestamp_slug, write_text_report


@dataclass(frozen=True)
class TrainConfig:
    seed: int


_NUMERIC_COLS = ["duration_days", "milestone_delay_days", "budget_variance_pct"]
_TEXT_COL = "lessons_learned_text"
_TARGET_COL = "risk_label"


def _top_coefficients(
    *,
    model: LogisticRegression,
    feature_names: list[str],
    top_k: int = 15,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    coefs = model.coef_.ravel()
    idx_sorted = np.argsort(coefs)

    neg = [(feature_names[i], float(coefs[i])) for i in idx_sorted[:top_k]]
    pos = [(feature_names[i], float(coefs[i])) for i in idx_sorted[-top_k:][::-1]]
    return pos, neg


def train_and_report(*, data_csv: Path, model_out: Path, report_dir: Path, seed: int = 42) -> Path:
    cfg = TrainConfig(seed=seed)

    df = pd.read_csv(data_csv)
    missing = [c for c in (_NUMERIC_COLS + [_TEXT_COL, _TARGET_COL]) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_csv}: {missing}")

    X = df[_NUMERIC_COLS + [_TEXT_COL]]
    y = df[_TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=cfg.seed,
        stratify=y,
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", _NUMERIC_COLS),
            (
                "txt",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
                _TEXT_COL,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=None,
        class_weight="balanced",
        random_state=cfg.seed,
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc = float(roc_auc_score(y_test, y_proba))

    # Build human-readable feature names for a simple coefficient report.
    # ColumnTransformer feature naming is awkward; we create names manually.
    tfidf: TfidfVectorizer = pipe.named_steps["pre"].named_transformers_["txt"]
    vocab = list(tfidf.get_feature_names_out())
    feature_names = [f"num:{c}" for c in _NUMERIC_COLS] + [f"txt:{t}" for t in vocab]

    model: LogisticRegression = pipe.named_steps["clf"]
    pos, neg = _top_coefficients(model=model, feature_names=feature_names, top_k=12)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_out)

    stamp = utc_timestamp_slug()
    report = "\n".join(
        [
            "# Risk Analyzer Run Report",
            "",
            f"- Data: `{data_csv.as_posix()}`",
            f"- Model: TF-IDF + LogisticRegression",
            f"- Seed: `{cfg.seed}`",
            "",
            "## Metrics (test split)",
            "",
            f"- Accuracy: `{acc:.3f}`",
            f"- F1: `{f1:.3f}`",
            f"- ROC-AUC: `{roc:.3f}`",
            "",
            "## Top risk drivers (positive coefficients)",
            "",
            *[f"- `{name}`: `{value:+.4f}`" for name, value in pos],
            "",
            "## Top protective drivers (negative coefficients)",
            "",
            *[f"- `{name}`: `{value:+.4f}`" for name, value in neg],
            "",
            "## Notes",
            "",
            "- This PoC is trained on synthetic data to demonstrate an end-to-end workflow.",
            "- Coefficients are a simple, transparent baseline explanation method.",
        ]
    )

    artifacts = write_text_report(report_dir=report_dir, stem=f"run_{stamp}", content=report)
    return artifacts.report_path
