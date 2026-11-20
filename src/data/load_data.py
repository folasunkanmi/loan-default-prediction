"""Data loading utilities for the real Lending Club sample and synthetic fallback."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

_GRADE_MAP = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}


def _normalize_real_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common Lending Club text encodings into the model's numeric schema."""
    df = df.copy()

    if "term" in df.columns:
        term = df["term"].astype(str).str.extract(r"(\d+)")[0]
        df["term"] = pd.to_numeric(term, errors="coerce")

    if "grade" in df.columns:
        grade = df["grade"].astype(str).str.strip().str.upper()
        mapped = grade.map(_GRADE_MAP)
        df["grade"] = mapped.where(mapped.notna(), pd.to_numeric(grade, errors="coerce"))

    if "emp_length" in df.columns:
        emp = df["emp_length"].astype(str).str.strip().str.lower()
        numeric = pd.to_numeric(emp, errors="coerce")
        extracted = pd.to_numeric(emp.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
        extracted = extracted.where(~emp.str.contains(r"<\s*1", regex=True), 0.5)
        extracted = extracted.where(~emp.str.contains(r"10\s*\+", regex=True), 10.0)
        df["emp_length"] = numeric.where(numeric.notna(), extracted)

    numeric_columns = (
        "borrower_score", "payment_inc_ratio", "annual_inc", "loan_amnt",
        "dti", "revol_bal", "revol_util", "open_acc", "grade", "emp_length",
        "delinq_2yrs_zero", "pub_rec_zero", "default",
    )
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _normalize_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Convert recognized outcome labels to binary default; fail on unknown labels."""
    if "outcome" not in df.columns:
        return df
    labels = df["outcome"].astype(str).str.strip().str.lower()
    default_mask = labels.str.contains(r"default|charged.?off|late", regex=True)
    paid_mask = labels.str.contains(r"paid|fully paid", regex=True)
    unknown = ~(default_mask | paid_mask)
    if unknown.any():
        examples = sorted(labels[unknown].unique())[:5]
        raise ValueError(f"Unrecognized loan outcome labels: {examples}")
    df = df.copy()
    df["default"] = default_mask.astype(int)
    return df


def load_raw_data(filename: str = "loan_data_lc.csv") -> pd.DataFrame:
    """Load the real sample (preferred), or fall back to generated synthetic data."""
    path = RAW_DIR / filename
    if not path.exists():
        alt = RAW_DIR / "loan_data.csv"
        if alt.exists():
            print(f"Warning: {filename} not found – using synthetic {alt.name}")
            return pd.read_csv(alt)
        raise FileNotFoundError(
            f"Neither {path} nor {alt} found. Download the real dataset or run "
            "`python -m src.data.generate_data`."
        )

    df = _normalize_outcome(pd.read_csv(path))
    if "outcome" in df.columns:
        keep = [
            "loan_amnt", "term", "annual_inc", "dti", "payment_inc_ratio",
            "revol_bal", "revol_util", "purpose", "home_ownership",
            "delinq_2yrs_zero", "pub_rec_zero", "open_acc", "grade",
            "emp_length", "borrower_score", "default",
        ]
        df = df[[column for column in keep if column in df.columns]]

    if "default" not in df.columns:
        raise ValueError("Dataset must contain a 'default' target or recognized 'outcome' column.")
    df = _normalize_real_features(df)
    if df["default"].isna().any():
        raise ValueError("Target 'default' contains missing or non-numeric values.")
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
