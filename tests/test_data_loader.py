"""Tests for real Lending Club encoding normalization and outcome handling."""

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import _normalize_real_dataframe


def test_real_schema_normalizes_grade_employment_and_term():
    raw = pd.DataFrame(
        {
            "grade": ["A", "B", "C", "D"],
            "emp_length": ["< 1 year", "5 years", "10+ years", "n/a"],
            "term": [" 36 months", "60 months", "36 months", "60 months"],
            "loan_status": ["Fully Paid", "Charged Off", "Fully Paid", "Default"],
            "loan_amnt": [1000, 2000, 3000, 4000],
        }
    )
    out = _normalize_real_dataframe(raw.copy())
    assert list(out["grade"]) == [7, 6, 5, 4]
    assert list(out["emp_length"]) == [0, 5, 10, 0]
    assert list(out["term"]) == [36, 60, 36, 60]
    assert list(out["default"]) == [0, 1, 0, 1]


def test_real_schema_keeps_numeric_grade_and_employment_values():
    raw = pd.DataFrame(
        {
            "grade": [7, 5],
            "emp_length": [3, 10],
            "term": [36, 60],
            "loan_status": ["Fully Paid", "Charged Off"],
            "loan_amnt": [1000, 2000],
        }
    )
    out = _normalize_real_dataframe(raw.copy())
    assert list(out["grade"]) == [7, 5]
    assert list(out["emp_length"]) == [3, 10]


def test_outcome_normalization_and_unknown_labels():
    good = pd.DataFrame(
        {
            "grade": ["A"],
            "emp_length": ["1 year"],
            "term": ["36 months"],
            "loan_status": ["Current"],
            "loan_amnt": [1000],
        }
    )
    with pytest.raises(ValueError, match="Unknown loan outcome"):
        _normalize_real_dataframe(good)
