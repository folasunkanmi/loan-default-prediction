"""Risk-tier boundaries and version consistency."""

from __future__ import annotations

from src.config.settings import RISK_LOW, RISK_MEDIUM, VERSION
from api.main import APP_VERSION, _risk_tier
from app import streamlit_app as st_app


def test_version_constant_is_1_2_2():
    assert VERSION == "1.2.2"
    assert APP_VERSION == VERSION
    assert st_app.APP_VERSION == VERSION


def test_risk_tier_boundaries():
    assert RISK_LOW == 0.30
    assert RISK_MEDIUM == 0.55
    assert _risk_tier(0.0) == "Low Risk"
    assert _risk_tier(RISK_LOW - 1e-9) == "Low Risk"
    assert _risk_tier(RISK_LOW) == "Medium Risk"
    assert _risk_tier((RISK_LOW + RISK_MEDIUM) / 2) == "Medium Risk"
    assert _risk_tier(RISK_MEDIUM - 1e-9) == "Medium Risk"
    assert _risk_tier(RISK_MEDIUM) == "High Risk"
    assert _risk_tier(0.99) == "High Risk"
