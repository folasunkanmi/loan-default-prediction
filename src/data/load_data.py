"""Data loading utilities for the real Lending Club sample and synthetic fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

_KNOWN_OUTCOMES = {"default", "paid off"}

_GRADE_MAP = {
    "A": 7.0, "B": 6.0, "C": 5.0, "D": 4.0, "E": 3.0, "F": 2.0, "G": 1.0,
}

_EMP_LENGTH_MAP = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
    "n/a": 0,
    "NA": 0,
}


def _normalize_term(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    s = series.astype(str).str.replace(" months", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _normalize_grade(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    mapped = series.astype(str).str.strip().str.upper().map(_GRADE_MAP)
    if mapped.isna().any():
        numeric = pd.to_numeric(series, errors="coerce")
        mapped = mapped.fillna(numeric)
    return mapped.astype(float)


def _normalize_emp_length(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    s = series.astype(str).str.strip()
    mapped = s.map(_EMP_LENGTH_MAP)
    if mapped.isna().any():
        digits = s.str.extract(r"(\d+)", expand=False)
        mapped = mapped.fillna(pd.to_numeric(digits, errors="coerce"))
    return mapped.fillna(0).astype(float)


def _normalize_outcome(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.lower()
    unknown = set(cleaned.unique()) - _KNOWN_OUTCOMES - {"nan"}
    if unknown:
        raise ValueError(
            f"Unknown outcome label(s) in Lending Club data: {sorted(unknown)}. "
            f"Expected only {_KNOWN_OUTCOMES}."
        )
    return (cleaned == "default").astype(int)


def load_raw_data(filename: str = "loan_data_lc.csv") -> pd.DataFrame:
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
        df["default"] = _normalize_outcome(df["outcome"])
        if "term" in df.columns:
            df["term"] = _normalize_term(df["term"])
        if "grade" in df.columns:
            df["grade"] = _normalize_grade(df["grade"])
        if "emp_length" in df.columns:
            df["emp_length"] = _normalize_emp_length(df["emp_length"])
        keep = [
            "loan_amnt", "term", "annual_inc", "dti", "payment_inc_ratio",
            "revol_bal", "revol_util", "purpose", "home_ownership",
            "delinq_2yrs_zero", "pub_rec_zero", "open_acc", "grade",
            "emp_length", "borrower_score", "default",
        ]
        keep = [c for c in keep if c in df.columns]
        df = df[keep]
        df = df.dropna(subset=["default"]).reset_index(drop=True)
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
