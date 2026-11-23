"""Validated data loading for the Lending Club sample and synthetic fallback."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REAL_NUMERIC_COLUMNS = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "payment_inc_ratio",
    "revol_bal",
    "revol_util",
    "open_acc",
    "grade",
    "emp_length",
]
REAL_CATEGORICAL_COLUMNS = [
    "term",
    "purpose",
    "home_ownership",
    "delinq_2yrs_zero",
    "pub_rec_zero",
]
REAL_REQUIRED_COLUMNS = set(
    REAL_NUMERIC_COLUMNS + REAL_CATEGORICAL_COLUMNS + ["outcome"]
)
SYNTHETIC_TARGETS = {"default"}
OUTCOME_TO_TARGET = {"paid off": 0, "default": 1}


def _validate_target(target: pd.Series, source: Path) -> pd.Series:
    if target.isna().any():
        raise ValueError(f"Dataset target contains missing values: {source}")
    values = set(target.unique())
    if not values.issubset({0, 1}) or values != {0, 1}:
        raise ValueError(
            f"Dataset target must contain both binary classes 0 and 1; "
            f"found {sorted(values, key=str)} in {source}"
        )
    return target.astype("int8")


def load_raw_data(filename: str = "loan_data_lc.csv") -> pd.DataFrame:
    """Load and validate real Lending Club data, or the synthetic fallback."""
    path = RAW_DIR / filename
    if not path.exists():
        fallback = RAW_DIR / "loan_data.csv"
        if not fallback.exists():
            raise FileNotFoundError(
                f"Neither {path} nor {fallback} exists. "
                "Download the Lending Club sample or generate synthetic data."
            )
        path = fallback

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")
    if df.columns.has_duplicates:
        raise ValueError(f"Dataset contains duplicate column names: {path}")

    if "outcome" in df.columns:
        missing = sorted(REAL_REQUIRED_COLUMNS.difference(df.columns))
        if missing:
            raise ValueError(f"Real Lending Club dataset is missing columns: {missing}")
        outcome = df["outcome"].astype("string").str.strip().str.lower()
        unknown = sorted(set(outcome.dropna().unique()) - set(OUTCOME_TO_TARGET))
        if outcome.isna().any() or unknown:
            raise ValueError(
                "Real Lending Club outcome must contain only 'default' and "
                f"'paid off'; found missing={bool(outcome.isna().any())}, values={unknown}"
            )
        df = df[REAL_NUMERIC_COLUMNS + REAL_CATEGORICAL_COLUMNS].copy()
        df["default"] = outcome.map(OUTCOME_TO_TARGET).astype("int8")
        for column in REAL_NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="raise")
        dataset_type = "real"
        dataset_source = (
            "https://github.com/gedeck/practical-statistics-for-data-scientists/"
            "raw/master/data/loan_data.csv.gz"
        )
    else:
        if "default" not in df.columns:
            raise ValueError("Dataset must contain an 'outcome' or 'default' target.")
        df["default"] = _validate_target(df["default"], path)
        dataset_type = "synthetic"
        dataset_source = "src.data.generate_data.generate_loan_dataset"

    df["default"] = _validate_target(df["default"], path)
    df.attrs["provenance"] = {
        "dataset_type": dataset_type,
        "dataset_source": dataset_source,
        "source_file": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
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
