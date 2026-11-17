"""
Generate a realistic synthetic loan default dataset.
Designed to mimic the statistical properties of Lending Club data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

RANDOM_SEED = 42
N_SAMPLES = 50_000
DEFAULT_RATE = 0.20


def _generate_fico(n: int, rng: np.random.Generator) -> np.ndarray:
    scores = rng.normal(loc=680, scale=55, size=n)
    return np.clip(scores, 300, 850).astype(int)


def _generate_income(n: int, rng: np.random.Generator) -> np.ndarray:
    income = rng.lognormal(mean=10.8, sigma=0.55, size=n)
    return np.clip(income, 15_000, 350_000).astype(int)


def generate_loan_dataset(
    n_samples: int = N_SAMPLES,
    default_rate: float = DEFAULT_RATE,
    seed: int = RANDOM_SEED,
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    fico = _generate_fico(n_samples, rng)
    annual_inc = _generate_income(n_samples, rng)
    loan_amnt = rng.choice(
        [5_000, 7_500, 10_000, 12_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000],
        size=n_samples,
        p=[0.08, 0.10, 0.15, 0.12, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04],
    )

    base_rate = 0.28 - (fico - 300) / 550 * 0.22
    int_rate = np.clip(base_rate + rng.normal(0, 0.015, n_samples), 0.05, 0.30)

    dti = np.clip(rng.normal(18, 8, n_samples), 1, 45)
    revol_util = np.clip(rng.beta(2, 3, n_samples) * 100, 0, 100)
    term = rng.choice([36, 60], size=n_samples, p=[0.7, 0.3])

    monthly_rate = int_rate / 12
    installment = (
        loan_amnt * monthly_rate * (1 + monthly_rate) ** term
        / ((1 + monthly_rate) ** term - 1)
    )

    grade_map = {
        (0, 580): "G", (580, 620): "F", (620, 660): "E",
        (660, 700): "D", (700, 740): "C", (740, 780): "B", (780, 850): "A",
    }

    def assign_grade(score: int) -> str:
        for (low, high), g in grade_map.items():
            if low <= score < high:
                return g
        return "A"

    grade = np.array([assign_grade(s) for s in fico])

    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], size=n_samples, p=[0.40, 0.45, 0.12, 0.03])
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "other"],
        size=n_samples, p=[0.45, 0.20, 0.12, 0.08, 0.05, 0.10],
    )
    emp_length = rng.choice(
        ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
         "6 years", "7 years", "8 years", "9 years", "10+ years"],
        size=n_samples,
        p=[0.08, 0.07, 0.08, 0.08, 0.07, 0.08, 0.07, 0.07, 0.07, 0.06, 0.27],
    )

    delinq_2yrs = rng.poisson(0.3, n_samples).clip(0, 8)
    pub_rec = rng.poisson(0.15, n_samples).clip(0, 5)
    inq_last_6mths = rng.poisson(0.8, n_samples).clip(0, 10)
    open_acc = rng.poisson(9, n_samples).clip(1, 30)
    total_acc = (open_acc + rng.poisson(8, n_samples)).clip(open_acc, 60)

    risk_score = (
        2.8 * (int_rate - 0.12)
        + 0.045 * (dti - 15)
        + 0.018 * (700 - fico)
        + 0.012 * (revol_util - 40)
        + 0.35 * delinq_2yrs
        + 0.45 * pub_rec
        + 0.18 * inq_last_6mths
        + rng.normal(0, 0.9, n_samples)
    )

    prob_default = 1 / (1 + np.exp(-risk_score))
    threshold = np.quantile(prob_default, 1 - default_rate)
    default = (prob_default > threshold).astype(int)

    df = pd.DataFrame({
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": np.round(int_rate * 100, 2),
        "installment": np.round(installment, 2),
        "grade": grade,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "purpose": purpose,
        "dti": np.round(dti, 2),
        "delinq_2yrs": delinq_2yrs,
        "fico_score": fico,
        "inq_last_6mths": inq_last_6mths,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_util": np.round(revol_util, 1),
        "total_acc": total_acc,
        "default": default,
    })

    for col in ["emp_length", "revol_util", "dti"]:
        mask = rng.random(n_samples) < 0.03
        df.loc[mask, col] = np.nan

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved synthetic dataset → {path}  ({len(df):,} rows)")

    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "data" / "raw" / "loan_data.csv"
    generate_loan_dataset(save_path=out)
    print("Done.")
