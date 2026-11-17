"""
Pydantic models for dual-schema loan applicant inputs.

Supports both:
- real  : Lending Club cleaned sample columns
- synthetic : generator columns (fico_score, int_rate, ...)
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


class RealSchemaInput(BaseModel):
    """Applicant features matching the real Lending Club sample."""

    loan_amnt: float = Field(..., ge=500, le=50000, description="Loan amount in USD")
    term: int = Field(..., description="Loan term in months (36 or 60)")
    annual_inc: float = Field(..., ge=5000, le=1_000_000, description="Annual income")
    dti: float = Field(..., ge=0, le=60, description="Debt-to-income ratio")
    payment_inc_ratio: float = Field(..., ge=0, le=50, description="Payment / income ratio")
    revol_bal: float = Field(..., ge=0, le=500_000, description="Revolving balance")
    revol_util: float = Field(..., ge=0, le=150, description="Revolving utilization %")
    purpose: str = Field(..., min_length=1)
    home_ownership: str = Field(..., min_length=1)
    delinq_2yrs_zero: int = Field(..., ge=0, le=1, description="1 if no delinquencies in 2y")
    pub_rec_zero: int = Field(..., ge=0, le=1, description="1 if no public records")
    open_acc: int = Field(..., ge=0, le=80)
    grade: float = Field(..., ge=1.0, le=7.0, description="Numeric grade proxy")
    emp_length: float = Field(..., ge=0, le=15, description="Employment length in years")
    borrower_score: float = Field(..., ge=0.0, le=1.0)

    @field_validator("term")
    @classmethod
    def term_must_be_36_or_60(cls, v: int) -> int:
        if v not in (36, 60):
            raise ValueError("term must be 36 or 60")
        return v

    @field_validator("home_ownership")
    @classmethod
    def normalize_home(cls, v: str) -> str:
        allowed = {"RENT", "MORTGAGE", "OWN", "OTHER"}
        up = v.strip().upper()
        if up not in allowed:
            raise ValueError(f"home_ownership must be one of {sorted(allowed)}")
        return up

    model_config = {"extra": "forbid"}


class SyntheticSchemaInput(BaseModel):
    """Applicant features matching the synthetic generator schema."""

    loan_amnt: float = Field(..., ge=500, le=50000)
    term: int = Field(...)
    int_rate: float = Field(..., ge=0.0, le=40.0, description="Interest rate %")
    installment: float = Field(..., ge=0, le=5000)
    grade: str = Field(..., min_length=1, max_length=2)
    emp_length: str = Field(..., min_length=1)
    home_ownership: str = Field(..., min_length=1)
    annual_inc: float = Field(..., ge=5000, le=1_000_000)
    purpose: str = Field(..., min_length=1)
    dti: float = Field(..., ge=0, le=60)
    delinq_2yrs: int = Field(..., ge=0, le=20)
    fico_score: int = Field(..., ge=300, le=850)
    inq_last_6mths: int = Field(..., ge=0, le=20)
    open_acc: int = Field(..., ge=0, le=80)
    pub_rec: int = Field(..., ge=0, le=20)
    revol_util: float = Field(..., ge=0, le=150)
    total_acc: int = Field(..., ge=1, le=100)

    @field_validator("term")
    @classmethod
    def term_must_be_36_or_60(cls, v: int) -> int:
        if v not in (36, 60):
            raise ValueError("term must be 36 or 60")
        return v

    @field_validator("grade")
    @classmethod
    def normalize_grade(cls, v: str) -> str:
        g = v.strip().upper()
        if g not in {"A", "B", "C", "D", "E", "F", "G"}:
            raise ValueError("grade must be A–G")
        return g

    @field_validator("home_ownership")
    @classmethod
    def normalize_home(cls, v: str) -> str:
        allowed = {"RENT", "MORTGAGE", "OWN", "OTHER"}
        up = v.strip().upper()
        if up not in allowed:
            raise ValueError(f"home_ownership must be one of {sorted(allowed)}")
        return up

    model_config = {"extra": "forbid"}


class PredictionRequest(BaseModel):
    """
    Flexible request: either provide `schema` + matching fields,
    or omit schema and let the server use the trained model's schema.
    """

    schema: Optional[Literal["real", "synthetic"]] = Field(
        default=None,
        description="Force input schema; defaults to the trained model schema",
    )
    features: Dict[str, Any] = Field(..., description="Applicant feature map")

    model_config = {"extra": "forbid"}


class PredictionResponse(BaseModel):
    default_probability: float
    predicted_class: int
    risk_tier: str
    threshold: float
    model_name: str
    schema: str
    top_drivers: Optional[list] = None


def validate_and_build_row(
    features: Dict[str, Any],
    schema: Literal["real", "synthetic"],
) -> pd.DataFrame:
    """Validate features against the chosen schema and return a 1-row DataFrame."""
    if schema == "real":
        validated = RealSchemaInput.model_validate(features)
    else:
        validated = SyntheticSchemaInput.model_validate(features)
    return pd.DataFrame([validated.model_dump()])
