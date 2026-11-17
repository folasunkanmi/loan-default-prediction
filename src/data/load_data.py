"""Data loading utilities for the real Lending Club sample and synthetic fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_data(filename: str = "loan_data_lc.csv") -> pd.DataFrame:
    """
    Load the real Lending Club sample (preferred) or fall back to synthetic data.

    The preferred file is the cleaned sample from:
    https://github.com/gedeck/practical-statistics-for-data-scientists
    (originally derived from Lending Club public loan data).
    """
    path = RAW_DIR / filename
    if not path.exists():
        alt = RAW_DIR / "loan_data.csv"
        if alt.exists():
            print(f"Warning: {filename} not found – using synthetic {alt.name}")
            return pd.read_csv(alt)
        raise FileNotFoundError(
            f"Neither {path} nor {alt} found. "
            "Download the real dataset or run `python -m src.data.generate_data`."
        )

    df = pd.read_csv(path)

    if "outcome" in df.columns:
        df = df.copy()
        df["default"] = (df["outcome"] == "default").astype(int)
        df["term"] = df["term"].str.replace(" months", "", regex=False).astype(int)
        keep = [
            "loan_amnt", "term", "annual_inc", "dti", "payment_inc_ratio",
            "revol_bal", "revol_util", "purpose", "home_ownership",
            "delinq_2yrs_zero", "pub_rec_zero", "open_acc", "grade",
            "emp_length", "borrower_score", "default",
        ]
        keep = [c for c in keep if c in df.columns]
        df = df[keep]
    return df


def save_processed_data(df: pd.DataFrame, filename: str = "loan_processed.parquet") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / filename
    df.to_parquet(path, index=False)
    return path


def load_processed_data(filename: str = "loan_processed.parquet") -> pd.DataFrame:
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_parquet(path)
