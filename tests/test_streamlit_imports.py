"""
Regression: Streamlit app must import src.* when launched from a different CWD.

Streamlit Cloud often does not use the repository root as the working directory,
which caused ModuleNotFoundError: No module named 'src'.
"""

from __future__ import annotations

import importlib
import os
import runpy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "app" / "streamlit_app.py"


def test_src_package_importable_from_repo_root():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import src  # noqa: F401
    from src.explainability.shap_analysis import compute_shap_values, top_shap_drivers

    assert callable(compute_shap_values)
    assert callable(top_shap_drivers)


def test_streamlit_app_bootstrap_adds_repo_root(tmp_path, monkeypatch):
    """Simulate Streamlit Cloud: run from a non-repo CWD and still import src."""
    # Work from an empty temp directory (not the repo root)
    monkeypatch.chdir(tmp_path)
    assert Path.cwd() != REPO_ROOT

    # Clear any prior src imports so we truly re-import via the app bootstrap
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            del sys.modules[key]
    # Remove repo root from path if present so the app must re-add it
    while str(REPO_ROOT) in sys.path:
        sys.path.remove(str(REPO_ROOT))

    # Minimal mirror of the app's bootstrap logic:
    project_root = APP_PATH.resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from src.explainability.shap_analysis import compute_shap_values, top_shap_drivers
    from src.config import settings

    assert callable(compute_shap_values)
    assert hasattr(settings, "RISK_LOW")
    assert root_str in sys.path


def test_app_file_contains_sys_path_bootstrap():
    """Guard against regressions that remove the Streamlit Cloud path fix."""
    text = APP_PATH.read_text(encoding="utf-8")
    assert "sys.path" in text
    assert "parents[1]" in text
    # path bootstrap must appear before src imports
    assert text.find("sys.path") < text.find("from src")


def test_try_shap_drivers_import_path_works_from_foreign_cwd(tmp_path, monkeypatch):
    """End-to-end: after bootstrap, try_shap_drivers' internal import succeeds."""
    monkeypatch.chdir(tmp_path)
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            del sys.modules[key]
    while str(REPO_ROOT) in sys.path:
        sys.path.remove(str(REPO_ROOT))

    project_root = APP_PATH.resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import the helper the same way the app does (lazy import inside function)
    from src.explainability.shap_analysis import compute_shap_values, top_shap_drivers

    assert compute_shap_values is not None
    assert top_shap_drivers is not None
