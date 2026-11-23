# Model Governance & Limitations

**Software version:** 1.1.2  
**Scope:** Loan default *probability* scoring for demonstration, research, and portfolio use.

## What has been measured

| Item | Status |
|------|--------|
| Stratified Train / Validation / Test split | Implemented |
| Model selection on **validation** only | Implemented |
| Threshold tuned on validation (F1 on default class) | Implemented |
| Held-out **test** metrics once per champion model | Implemented |
| Discrimination metrics (ROC-AUC, PR-AUC, precision, recall, F1) | Reported in `reports/model_comparison.json` |
| SHAP / coefficient attributions for individual predictions | Implemented in Streamlit |
| Schema detection (real Lending Club vs synthetic) | Implemented |

Check `reports/model_comparison.json` → `provenance.dataset_type`:

- **`synthetic`** — metrics are **not** real Lending Club performance (typically higher AUC on synthetic data).
- **`real`** — metrics refer to the cleaned Lending Club sample described in the README (~45k loans). Typical real-sample ROC-AUC is about **0.73**.

## What has **not** been claimed or measured

- **Calibration** (e.g. reliability diagrams, Brier score, expected calibration error) is not part of the automated report. Predicted probabilities should not be treated as well-calibrated without further analysis.
- **Fairness / disparate impact** across protected attributes (race, gender, age, etc.) has **not** been audited in this repository. Do not assume equal error rates or equal opportunity across groups.
- **Regulatory compliance** (ECOA, FCRA, GDPR automated-decision rules, local credit statutes) is **not** certified. This project is not a compliance product.
- **Production monitoring, drift detection, automated retraining, and a FastAPI service** are **not** included in this release.
- **Causal** interpretation of SHAP/coefficient drivers is not supported; attributions are associative.

## Appropriate use

1. Use the **saved decision threshold** from the model artefact for accept/reject style decisions in demos; risk **bands** (0.30 / 0.55) are display aids, not the optimised cut-off.
2. Prefer models trained on **real** Lending Club data for any serious credit-risk illustration; regenerate metrics with `python -m src.models.train` after downloading real data.
3. Treat Streamlit outputs as **decision support**, not automated underwriting.
4. Re-validate metrics, calibration, and fairness on the target population before any operational use.

## Risk bands vs decision threshold

| Concept | Source | Purpose |
|---------|--------|---------|
| Decision threshold | `artifact["threshold"]` (validation F1) | Binary "likely default / likely pay" in the app |
| Risk bands LOW/MEDIUM/HIGH | `src.config.settings` (0.30 / 0.55) | Human-readable tier labels only |

These are intentionally separate.
