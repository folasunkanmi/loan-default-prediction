"""Tests for common Lending Club raw-data encodings."""

import pandas as pd
import pytest

from src.data.load_data import _normalize_outcome, _normalize_real_features


def test_real_schema_normalizes_grade_employment_and_term():
    raw = pd.DataFrame({
        "term": ["36 months", "60 months"],
        "grade": ["A", "G"],
        "emp_length": ["< 1 year", "10+ years"],
        "annual_inc": ["50000", "70000"],
    })
    normalized = _normalize_real_features(raw)
    assert normalized["term"].tolist() == [36, 60]
    assert normalized["grade"].tolist() == [7, 1]
    assert normalized["emp_length"].tolist() == [0.5, 10.0]
    assert normalized["annual_inc"].tolist() == [50000, 70000]


def test_real_schema_keeps_numeric_grade_and_employment_values():
    raw = pd.DataFrame({"term": [36], "grade": [5.0], "emp_length": [7]})
    normalized = _normalize_real_features(raw)
    assert normalized.loc[0, "grade"] == 5
    assert normalized.loc[0, "emp_length"] == 7


def test_outcome_normalization_and_unknown_labels():
    df = pd.DataFrame({"outcome": ["default", "fully paid", "charged off"]})
    normalized = _normalize_outcome(df)
    assert normalized["default"].tolist() == [1, 0, 1]
    with pytest.raises(ValueError, match="Unrecognized"):
        _normalize_outcome(pd.DataFrame({"outcome": ["Current"]}))
