"""Repository-integrity and model-artifact regression checks."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def test_version_consistency_across_manifests():
    """Version metadata and changelog must agree."""
    from src.config.settings import VERSION

    assert VERSION == "1.1.3"

    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'version\s*=\s*"1\.1\.3"', pyproject)

    citation = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r'version:\s*"1\.1\.3"', citation)

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "version-1.1.3" in readme
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [1.1.3]" in changelog


def test_model_comparison_has_provenance():
    path = REPO / "reports" / "model_comparison.json"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert not any(marker in text for marker in ("<<<<<<<", "=======", ">>>>>>>"))
    report = json.loads(text)
    assert "provenance" in report
    prov = report["provenance"]
    assert prov.get("dataset_type") in {"synthetic", "real"}
    assert report.get("selected_model") == prov.get("selected_model")
    assert report.get("decision_threshold") == prov.get("decision_threshold")
    assert set(report["test"]["metrics"]) >= {
        "roc_auc", "pr_auc", "precision", "recall", "f1", "best_threshold"
    }
    assert set(report["test"]["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}
    assert set(report["artifacts"]) == {"model", "roc_pr_plot", "confusion_matrix_plot"}
    assert report["evaluation_protocol"]["model_selection"] == "validation PR-AUC only"
    assert report["evaluation_protocol"]["threshold_selection"] == "validation F1 only"
    for artifact in report["artifacts"].values():
        assert (REPO / artifact).is_file()
    # Must not hide synthetic as real
    if prov["dataset_type"] == "synthetic":
        note = (prov.get("note") or "").lower()
        assert "synthetic" in note or "not real" in note
    else:
        assert prov["default_rate"] == pytest.approx(0.5)
        assert "balanced curated sample" in prov["note"]
        assert len(prov["source_sha256"]) == 64


def test_streamlit_uses_artifact_threshold_not_hardcoded_decision_cut():
    """Decision cut-off must come from artifact['threshold'], not a fixed 0.5 literal for decisions."""
    text = (REPO / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert 'artifact.get("threshold"' in text or "artifact['threshold']" in text
    assert "proba >= threshold" in text
    assert 'inputs["borrower_score"]' not in text


def test_artifact_threshold_and_real_prediction_compatible():
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
    assert art["schema"] == "real"
    assert "borrower_score" not in art["raw_feature_names"]
    from src.data.load_data import load_raw_data

    sample = load_raw_data().loc[:, art["raw_feature_names"]].iloc[[0]]
    probability = float(art["pipeline"].predict_proba(sample)[0, 1])
    assert 0.0 <= probability <= 1.0
    assert float(art["threshold"]) == pytest.approx(
        float(art["provenance"]["decision_threshold"])
    )


def test_risk_bands_centralised():
    from src.config.settings import RISK_LOW, RISK_MEDIUM

    assert RISK_LOW == 0.30
    assert RISK_MEDIUM == 0.55
    assert RISK_LOW < RISK_MEDIUM


def test_requires_python_311():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11,<3.13"' in text
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert 'python-version: ["3.11", "3.12"]' in ci
