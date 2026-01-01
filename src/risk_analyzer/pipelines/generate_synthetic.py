from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    n: int
    seed: int


_THEMES = {
    "supplier": [
        "Supplier delays caused schedule overruns",
        "Late component delivery impacted integration milestones",
        "Vendor lead times were underestimated",
    ],
    "requirements": [
        "Unclear requirements led to rework",
        "Scope changes were not controlled early",
        "Ambiguous specifications increased iteration cycles",
    ],
    "tools": [
        "Tool instability affected testing phase",
        "CI pipeline failures slowed validation",
        "Test equipment downtime blocked verification",
    ],
    "resources": [
        "Resource constraints reduced development velocity",
        "Key experts were unavailable during critical phases",
        "Parallel priorities caused context switching",
    ],
    "planning": [
        "Milestone planning was optimistic and lacked buffers",
        "Risk registers were not updated consistently",
        "Dependencies were identified too late",
    ],
    "quality": [
        "Late defect discovery increased rework",
        "Quality gates were skipped under schedule pressure",
        "Verification coverage was insufficient early",
    ],
}


def _sample_ll_text(rng: np.random.Generator) -> tuple[str, dict[str, int]]:
    theme_names = list(_THEMES.keys())

    # Choose 1-3 themes; include some high-signal risk themes more often.
    weights = np.array([1.3, 1.2, 1.0, 1.0, 0.9, 0.8], dtype=float)
    weights = weights / weights.sum()
    k = int(rng.integers(1, 4))
    chosen = rng.choice(theme_names, size=k, replace=False, p=weights)

    parts: list[str] = []
    counts: dict[str, int] = {t: 0 for t in theme_names}
    for t in chosen:
        parts.append(str(rng.choice(_THEMES[t])))
        counts[t] += 1

    # Add a neutral sentence sometimes to reduce trivial separability.
    if rng.random() < 0.4:
        parts.append("Cross-team alignment improved after a mid-project review")

    return "; ".join(parts), counts


def generate_synthetic_dataset(*, out_csv: Path, n: int = 500, seed: int = 42) -> None:
    cfg = SyntheticConfig(n=n, seed=seed)
    rng = np.random.default_rng(cfg.seed)

    rows: list[dict[str, object]] = []

    for i in range(cfg.n):
        ll_text, theme_counts = _sample_ll_text(rng)

        duration_days = int(rng.normal(loc=210, scale=45).clip(90, 420))
        milestone_delay_days = int(rng.normal(loc=12, scale=16).clip(0, 90))
        budget_variance_pct = float(rng.normal(loc=3.0, scale=8.0).clip(-15, 40))

        # Risk signal: numeric stress + specific themes.
        theme_risk = (
            0.9 * theme_counts["supplier"]
            + 0.8 * theme_counts["requirements"]
            + 0.6 * theme_counts["tools"]
            + 0.5 * theme_counts["resources"]
        )
        numeric_risk = 0.015 * milestone_delay_days + 0.02 * max(budget_variance_pct, 0.0)
        base = -1.1
        logit = base + theme_risk + numeric_risk
        prob_risk = float(1.0 / (1.0 + np.exp(-logit)))

        # Add label noise.
        prob_risk = float((0.85 * prob_risk) + 0.15 * rng.random())
        risk_label = int(prob_risk >= 0.5)

        rows.append(
            {
                "project_id": f"P{i:05d}",
                "duration_days": duration_days,
                "milestone_delay_days": milestone_delay_days,
                "budget_variance_pct": budget_variance_pct,
                "lessons_learned_text": ll_text,
                "risk_label": risk_label,
                "risk_probability": prob_risk,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
