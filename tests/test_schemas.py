"""Unit tests for dual-schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.prediction import RealSchemaInput, SyntheticSchemaInput, validate_and_build_row


def test_real_schema_valid():
    data = {
        "loan_amnt": 12000,
        "term": 36,
        "annual_inc": 65000,
        "dti": 16.0,
        "payment_inc_ratio": 8.0,
        "revol_bal": 12000,
        "revol_util": 55.0,
        "purpose": "debt_consolidation",
        "home_ownership": "MORTGAGE",
        "delinq_2yrs_zero": 1,
        "pub_rec_zero": 1,
        "open_acc": 10,
        "grade": 4.5,
        "emp_length": 5,
        "borrower_score": 0.55,
    }
    obj = RealSchemaInput.model_validate(data)
    assert obj.term == 36
    df = validate_and_build_row(data, "real")
    assert df.shape == (1, 15)


def test_real_schema_rejects_bad_term():
    data = {
        "loan_amnt": 12000,
        "term": 48,
        "annual_inc": 65000,
        "dti": 16.0,
        "payment_inc_ratio": 8.0,
        "revol_bal": 12000,
        "revol_util": 55.0,
        "purpose": "debt_consolidation",
        "home_ownership": "MORTGAGE",
        "delinq_2yrs_zero": 1,
        "pub_rec_zero": 1,
        "open_acc": 10,
        "grade": 4.5,
        "emp_length": 5,
        "borrower_score": 0.55,
    }
    with pytest.raises(ValidationError):
        RealSchemaInput.model_validate(data)


def test_synthetic_schema_valid():
    data = {
        "loan_amnt": 15000,
        "term": 60,
        "int_rate": 12.5,
        "installment": 350.0,
        "grade": "C",
        "emp_length": "5 years",
        "home_ownership": "RENT",
        "annual_inc": 72000,
        "purpose": "credit_card",
        "dti": 18.2,
        "delinq_2yrs": 0,
        "fico_score": 710,
        "inq_last_6mths": 1,
        "open_acc": 9,
        "pub_rec": 0,
        "revol_util": 42.0,
        "total_acc": 18,
    }
    obj = SyntheticSchemaInput.model_validate(data)
    assert obj.grade == "C"
    df = validate_and_build_row(data, "synthetic")
    assert "fico_score" in df.columns


def test_synthetic_rejects_bad_fico():
    data = {
        "loan_amnt": 15000,
        "term": 60,
        "int_rate": 12.5,
        "installment": 350.0,
        "grade": "C",
        "emp_length": "5 years",
        "home_ownership": "RENT",
        "annual_inc": 72000,
        "purpose": "credit_card",
        "dti": 18.2,
        "delinq_2yrs": 0,
        "fico_score": 200,
        "inq_last_6mths": 1,
        "open_acc": 9,
        "pub_rec": 0,
        "revol_util": 42.0,
        "total_acc": 18,
    }
    with pytest.raises(ValidationError):
        SyntheticSchemaInput.model_validate(data)
