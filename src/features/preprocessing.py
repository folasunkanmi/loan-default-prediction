"""Preprocessing pipeline using scikit-learn ColumnTransformer."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "loan_amnt", "annual_inc", "dti", "payment_inc_ratio", "revol_bal",
    "revol_util", "open_acc", "grade", "emp_length", "borrower_score",
]
CATEGORICAL_FEATURES = [
    "term", "purpose", "home_ownership", "delinq_2yrs_zero", "pub_rec_zero",
]
TARGET = "default"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    num_names = NUMERIC_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"]["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return num_names + cat_names


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    num = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    X = df[num + cat].copy()
    y = df[TARGET].astype(int)
    return X, y
