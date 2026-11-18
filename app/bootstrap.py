"""Optional cold-start bootstrap for Streamlit Cloud when no model artifact exists."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_pipeline.joblib"
RAW_SYNTH = PROJECT_ROOT / "data" / "raw" / "loan_data.csv"


def ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH

    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    if not RAW_SYNTH.exists():
        from src.data.generate_data import generate_loan_dataset

        generate_loan_dataset(n_samples=8_000, save_path=RAW_SYNTH)

    from src.models.train import train_all_models

    train_all_models()
    return MODEL_PATH


if __name__ == "__main__":
    path = ensure_model()
    print(f"Model ready at {path}")
