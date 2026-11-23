"""Central configuration for reproducibility and production settings."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_SEED = 42
TEST_SIZE = 0.20
N_JOBS = -1

CV_FOLDS = 5
SMOTE_K_NEIGHBORS = 5

# Probability-of-default risk bands used by the Streamlit app
RISK_LOW = 0.30
RISK_MEDIUM = 0.55

VERSION = "1.1.3"
