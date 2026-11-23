"""Production-readiness regression checks for v1.1.2."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def test_version_consistency_across_manifests():
    """settings, pyproject, CITATION, and CHANGELOG head must agree on 1.1.2."""
    from src.config.settings import VERSION

    assert VERSION == "1.1.2"

    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'version\s*=\s*"1\.1\.2"', pyproject)

    citation = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r'version:\s*"1\.1\.2"', citation)

    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [1.1.2]" in changelog


def test_model_comparison_has_provenance():
    path = REPO / "reports" / "model_comparison.json"
    assert path.exists()
    report = json.loads(path.read_text(encoding="utf-8"))
    assert "provenance" in report
    prov = report["provenance"]
    assert prov.get("dataset_type") in {"synthetic", "real"}
    assert "selected_model" in report or "selected_model" in prov
    # Must not hide synthetic as real
    if prov["dataset_type"] == "synthetic":
        note = (prov.get("note") or "").lower()
        assert "synthetic" in note or "not real" in note


def test_streamlit_uses_artifact_threshold_not_hardcoded_decision_cut():
    """Decision cut-off must come from artifact['threshold'], not a fixed 0.5 literal for decisions."""
    text = (REPO / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert 'artifact.get("threshold"' in text or "artifact['threshold']" in text
    assert "proba >= threshold" in text


def test_artifact_threshold_and_schema_loadable():
    model_path = REPO / "models" / "best_pipeline.joblib"
    if not model_path.exists():
        pytest.skip("No model artefact")
    import joblib
    import sklearn

    def _v(s):
        parts = []
        for x in s.split(".")[:3]:
            parts.append(int("".join(c for c in x if c.isdigit()) or 0))
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts)

    if _v(sklearn.__version__) < (1, 8, 0):
        pytest.skip("sklearn too old for artefact")

    art = joblib.load(model_path)
    assert "pipeline" in art
    assert "threshold" in art
    assert 0.0 < float(art["threshold"]) < 1.0
    assert art.get("schema") in {"synthetic", "real"}
    assert art.get("model_name")
    thr = float(art["threshold"])
    assert thr == pytest.approx(0.68, abs=0.2) or 0.15 <= thr <= 0.75


def test_risk_bands_centralised():
    from src.config.settings import RISK_LOW, RISK_MEDIUM

    assert RISK_LOW == 0.30
    assert RISK_MEDIUM == 0.55
    assert RISK_LOW < RISK_MEDIUM


def test_requires_python_311():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11"' in text
