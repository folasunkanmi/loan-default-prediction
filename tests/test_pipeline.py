"""Basic smoke tests for the loan default prediction pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generate_data import generate_loan_dataset
from src.features.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
)


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    return generate_loan_dataset(n_samples=500, seed=42)


def test_generate_data_shape(synthetic_df):
    assert len(synthetic_df) == 500
    assert TARGET in synthetic_df.columns or "default" in synthetic_df.columns
    assert synthetic_df["default"].isin([0, 1]).all()


def test_generate_data_default_rate(synthetic_df):
    rate = synthetic_df["default"].mean()
    assert 0.10 < rate < 0.35


def test_no_data_leakage_columns():
    assert TARGET not in NUMERIC_FEATURES
    assert TARGET not in CATEGORICAL_FEATURES


def test_prepare_xy_types(synthetic_df):
    assert "default" in synthetic_df.columns
    y = synthetic_df["default"]
    assert y.dtype in (np.int64, np.int32, int) or str(y.dtype).startswith("int")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
