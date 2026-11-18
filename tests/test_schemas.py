"""Pydantic schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.prediction import (
    RealSchemaInput,
    SyntheticSchemaInput,
    validate_and_build_row,
)


def test_synthetic_valid():
    data = {
        "loan_amnt": 12000,
        "term": 36,
        "int_rate": 11.5,
        "installment": 320,
        "grade": "c",
        "emp_length": "3 years",
        "home_ownership": "rent",
        "annual_inc": 55000,
        "purpose": "credit_card",
        "dti": 14.0,
        "delinq_2yrs": 0,
        "fico_score": 690,
        "inq_last_6mths": 2,
        "open_acc": 7,
        "pub_rec": 0,
        "revol_util": 35.0,
        "total_acc": 12,
    }
    obj = SyntheticSchemaInput.model_validate(data)
    assert obj.grade == "C"
    assert obj.home_ownership == "RENT"
    df = validate_and_build_row(data, schema="synthetic")
    assert df.shape == (1, 17)


def test_synthetic_bad_term():
    with pytest.raises(ValidationError):
        SyntheticSchemaInput.model_validate({
            "loan_amnt": 10000, "term": 24, "int_rate": 10, "installment": 200,
            "grade": "A", "emp_length": "1 year", "home_ownership": "OWN",
            "annual_inc": 80000, "purpose": "car", "dti": 10, "delinq_2yrs": 0,
            "fico_score": 750, "inq_last_6mths": 0, "open_acc": 5, "pub_rec": 0,
            "revol_util": 20, "total_acc": 10,
        })


def test_real_valid():
    data = {
        "loan_amnt": 10000,
        "term": 60,
        "annual_inc": 70000,
        "dti": 18,
        "payment_inc_ratio": 9,
        "revol_bal": 8000,
        "revol_util": 45,
        "purpose": "debt_consolidation",
        "home_ownership": "MORTGAGE",
        "delinq_2yrs_zero": 1,
        "pub_rec_zero": 1,
        "open_acc": 11,
        "grade": 5.0,
        "emp_length": 6,
        "borrower_score": 0.6,
    }
    obj = RealSchemaInput.model_validate(data)
    assert obj.term == 60
    df = validate_and_build_row(data, schema="real")
    assert "borrower_score" in df.columns
