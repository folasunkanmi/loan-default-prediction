# Model Governance & Limitations

**Software version:** 1.1.3
**Scope:** Demonstration of loan-default classification on a balanced, curated Lending Club sample.

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

The real sample contains 45,342 loans, evenly divided between 22,671 defaults and
22,671 paid-off loans. The target is mapped from the sample's `outcome` field.
`borrower_score` is excluded because its point-in-time availability is unclear
and it may encode target information. The exact source checksum, split sizes,
selected model, threshold, and measured metrics are in `reports/model_comparison.json`.

The balanced, curated sample is not representative of a real lending population.
Its metrics are not estimates of real-world Lending Club performance. Synthetic
metrics are also clearly labeled and must not be interpreted as real-data results.

## What has **not** been claimed or measured

- **Calibration** (e.g. reliability diagrams, Brier score, expected calibration error) is not part of the automated report. Predicted probabilities should not be treated as well-calibrated without further analysis.
- **Fairness / disparate impact** across protected attributes (race, gender, age, etc.) has **not** been audited in this repository. Do not assume equal error rates or equal opportunity across groups.
- **Regulatory compliance** (ECOA, FCRA, GDPR automated-decision rules, local credit statutes) is **not** certified. This project is not a compliance product.
- **Production monitoring, drift detection, automated retraining, and a FastAPI service** are **not** included in this release.
- **Causal** interpretation of SHAP/coefficient drivers is not supported; attributions are associative.

## Appropriate use

1. Use the **saved decision threshold** from the model artefact for accept/reject style decisions in demos; risk **bands** (0.30 / 0.55) are display aids, not the optimised cut-off.
2. Do not treat this balanced curated sample as representative of lending outcomes or as evidence of real-world predictive performance.
3. Treat Streamlit outputs as an educational demonstration, not credit decision support or automated underwriting.
4. Validate performance, calibration, fairness, and data rights independently before considering any use beyond demonstration.

## Risk bands vs decision threshold

| Concept | Source | Purpose |
|---------|--------|---------|
| Decision threshold | `artifact["threshold"]` (validation F1) | Binary "likely default / likely pay" in the app |
| Risk bands LOW/MEDIUM/HIGH | `src.config.settings` (0.30 / 0.55) | Human-readable tier labels only |

These are intentionally separate.
