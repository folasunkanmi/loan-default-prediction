"""Regression coverage for the real Lending Club sample integration."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.features.preprocessing import build_preprocessor, prepare_xy


def test_real_sample_target_provenance_and_leakage_exclusion():
    raw = pd.read_csv(load_data.RAW_DIR / "loan_data_lc.csv")
    data = load_data.load_raw_data()

    assert len(data) == 45_342
    assert data["default"].value_counts().to_dict() == {0: 22_671, 1: 22_671}
    assert data["default"].tolist() == raw["outcome"].eq("default").astype("int8").tolist()
    assert data.attrs["provenance"]["dataset_type"] == "real"
    assert len(data.attrs["provenance"]["source_sha256"]) == 64
    assert "borrower_score" not in data.columns
    assert "outcome" not in data.columns
    assert "status" not in data.columns

    X, y = prepare_xy(data)
    assert len(X) == len(y) == 45_342
    assert "borrower_score" not in X.columns
    assert "default" not in X.columns
    assert build_preprocessor("real", list(X.columns)).fit_transform(X).shape[0] == len(X)


def test_real_sample_rejects_unknown_outcome_labels(tmp_path, monkeypatch):
    rows = []
    for outcome in ("default", "paid off", "unknown"):
        row = {name: 1 for name in load_data.REAL_REQUIRED_COLUMNS}
        row["outcome"] = outcome
        rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / "loan_data_lc.csv", index=False)
    monkeypatch.setattr(load_data, "RAW_DIR", tmp_path)

    with pytest.raises(ValueError, match="outcome must contain only"):
        load_data.load_raw_data()


def test_prepare_xy_rejects_nonbinary_or_missing_targets():
    data = pd.DataFrame(
        {
            "loan_amnt": [1, 2, 3],
            "payment_inc_ratio": [0.1, 0.2, 0.3],
            "default": [0, 1, 2],
        }
    )
    with pytest.raises(ValueError, match="binary"):
        prepare_xy(data)

    data["default"] = [0, 1, None]
    with pytest.raises(ValueError, match="binary"):
        prepare_xy(data)
