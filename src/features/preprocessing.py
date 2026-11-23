"""
Preprocessing pipeline using scikit-learn ColumnTransformer.

Supports BOTH:
- Real Lending Club sample (loan_data_lc.csv)
- Synthetic generator output (loan_data.csv)

Feature lists are selected dynamically based on available columns.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Canonical feature groups for the REAL Lending Club sample.
# borrower_score is deliberately excluded: it is an issuer-derived score with
# unclear point-in-time availability and could encode the target.
REAL_NUMERIC = [
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
REAL_CATEGORICAL = [
    "term",
    "purpose",
    "home_ownership",
    "delinq_2yrs_zero",
    "pub_rec_zero",
]

# Feature groups for the SYNTHETIC generator
SYNTH_NUMERIC = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_score",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_util",
    "total_acc",
]
SYNTH_CATEGORICAL = [
    "term",
    "grade",
    "emp_length",
    "home_ownership",
    "purpose",
]

TARGET = "default"

# These are updated by detect_schema() / build_preprocessor()
NUMERIC_FEATURES: List[str] = REAL_NUMERIC.copy()
CATEGORICAL_FEATURES: List[str] = REAL_CATEGORICAL.copy()


def detect_schema(df: pd.DataFrame) -> str:
    """Return 'real' or 'synthetic' based on available columns."""
    if "borrower_score" in df.columns or "payment_inc_ratio" in df.columns:
        return "real"
    if "fico_score" in df.columns or "int_rate" in df.columns:
        return "synthetic"
    # Fallback: prefer real if most real columns exist
    real_hits = sum(c in df.columns for c in REAL_NUMERIC + REAL_CATEGORICAL)
    synth_hits = sum(c in df.columns for c in SYNTH_NUMERIC + SYNTH_CATEGORICAL)
    return "real" if real_hits >= synth_hits else "synthetic"


def set_feature_lists(schema: str) -> None:
    """Update module-level feature lists for the chosen schema."""
    global NUMERIC_FEATURES, CATEGORICAL_FEATURES
    if schema == "synthetic":
        NUMERIC_FEATURES = SYNTH_NUMERIC.copy()
        CATEGORICAL_FEATURES = SYNTH_CATEGORICAL.copy()
    else:
        NUMERIC_FEATURES = REAL_NUMERIC.copy()
        CATEGORICAL_FEATURES = REAL_CATEGORICAL.copy()


def build_preprocessor(
    schema: str | None = None,
    feature_columns: List[str] | None = None,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for the given schema.
    If schema is None, assumes feature lists have already been set.
    """
    if schema is not None:
        numeric_features = SYNTH_NUMERIC if schema == "synthetic" else REAL_NUMERIC
        categorical_features = (
            SYNTH_CATEGORICAL if schema == "synthetic" else REAL_CATEGORICAL
        )
    else:
        numeric_features = NUMERIC_FEATURES
        categorical_features = CATEGORICAL_FEATURES

    if feature_columns is not None:
        selected = set(feature_columns)
        numeric_features = [name for name in numeric_features if name in selected]
        categorical_features = [
            name for name in categorical_features if name in selected
        ]

    transformers = []
    if numeric_features:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_features))
    if categorical_features:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_features))
    if not transformers:
        raise ValueError("No supported feature columns are available for preprocessing.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extract final feature names after transformation (useful for SHAP).

    Prefer the fitted transformer's own get_feature_names_out() so that names
    always match the columns actually produced by this preprocessor instance,
    regardless of the current module-level NUMERIC_FEATURES / CATEGORICAL_FEATURES.
    Falls back to module-level lists only if the preprocessor is not yet fitted.
    """
    # Preferred path: derive names from the fitted ColumnTransformer
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            raw = list(preprocessor.get_feature_names_out())
            # Strip sklearn prefixes "num__" / "cat__" for cleaner SHAP labels
            cleaned = []
            for name in raw:
                if name.startswith("num__"):
                    cleaned.append(name[5:])
                elif name.startswith("cat__"):
                    cleaned.append(name[5:])
                else:
                    cleaned.append(name)
            return cleaned
        except Exception:
            pass

    # Fallback for unfitted preprocessor (uses current module-level lists)
    num_names = list(NUMERIC_FEATURES)
    try:
        cat_encoder = preprocessor.named_transformers_["cat"]["onehot"]
        cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    except Exception:
        cat_names = []
    return num_names + cat_names


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Detect schema, set feature lists, and return X, y.
    Only columns that actually exist are used.
    """
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain target column {TARGET!r}.")
    if df[TARGET].isna().any() or not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain non-null binary values 0 and 1.")

    schema = detect_schema(df)
    set_feature_lists(schema)
    num = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    if not num and not cat:
        raise ValueError(f"No supported {schema} feature columns are present.")
    X = df[num + cat].copy()
    y = df[TARGET].astype("int8")
    return X, y
