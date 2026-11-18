"""
Model monitoring and data/prediction drift detection.

- Population Stability Index (PSI) for numeric features and score distributions
- Kolmogorov–Smirnov statistic for continuous shifts
- Categorical frequency divergence
- Alert thresholds aligned with common credit-risk practice
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _psi_from_hist(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    e = np.clip(expected / max(expected.sum(), 1), eps, None)
    a = np.clip(actual / max(actual.sum(), 1), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 20 or len(current) < 20:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(np.nanmin(reference), np.nanmax(reference) + 1e-9, bins + 1)
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    return _psi_from_hist(ref_hist.astype(float), cur_hist.astype(float))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample KS statistic (no p-value dependency on scipy optional)."""
    reference = np.sort(np.asarray(reference, dtype=float))
    current = np.sort(np.asarray(current, dtype=float))
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 10 or len(current) < 10:
        return float("nan")
    # Empirical CDFs evaluated on pooled unique points
    data = np.unique(np.concatenate([reference, current]))
    cdf_r = np.searchsorted(reference, data, side="right") / len(reference)
    cdf_c = np.searchsorted(current, data, side="right") / len(current)
    return float(np.max(np.abs(cdf_r - cdf_c)))


def psi_categorical(ref_freq: Dict[str, float], current: pd.Series) -> float:
    cur = current.astype(str).value_counts(normalize=True).to_dict()
    keys = set(ref_freq) | set(cur)
    eps = 1e-4
    e = np.array([max(ref_freq.get(k, 0.0), eps) for k in keys])
    a = np.array([max(cur.get(k, 0.0), eps) for k in keys])
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def interpret_psi(psi: float) -> str:
    if np.isnan(psi):
        return "insufficient_data"
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate_shift"
    return "significant_drift"


def detect_feature_drift(
    reference_stats: Dict[str, Any],
    current_df: pd.DataFrame,
    psi_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compare a live/batch frame against training reference statistics.

    reference_stats is stored in the model artifact at train time.
    """
    results: Dict[str, Any] = {}
    alerts: List[str] = []

    for col, stats in reference_stats.items():
        if col not in current_df.columns:
            results[col] = {"status": "missing_column"}
            alerts.append(col)
            continue

        if stats.get("type") == "numeric":
            # Rebuild a synthetic reference sample from quantiles for PSI
            # Prefer using stored quantiles as anchors
            ref_proxy = np.array(
                [stats["p10"], stats["p50"], stats["p90"]] * 50, dtype=float
            )
            # Better: use mean/std normal proxy if quantiles thin
            if stats.get("std", 0) > 0:
                rng = np.random.default_rng(0)
                ref_proxy = rng.normal(stats["mean"], stats["std"], size=500)
            cur = current_df[col].astype(float).values
            psi = psi_numeric(ref_proxy, cur, bins=psi_bins)
            status = interpret_psi(psi)
            results[col] = {"psi": psi, "status": status, "type": "numeric"}
            if status == "significant_drift":
                alerts.append(col)
        else:
            psi = psi_categorical(stats.get("frequencies", {}), current_df[col])
            status = interpret_psi(psi)
            results[col] = {"psi": psi, "status": status, "type": "categorical"}
            if status == "significant_drift":
                alerts.append(col)

    overall = "significant_drift" if alerts else "stable"
    if not alerts and any(
        r.get("status") == "moderate_shift" for r in results.values()
    ):
        overall = "moderate_shift"

    return {
        "overall_status": overall,
        "n_features_checked": len(results),
        "alerts": alerts,
        "features": results,
    }


def detect_prediction_drift(
    reference_probas: List[float],
    current_probas: List[float],
    bins: int = 10,
) -> Dict[str, Any]:
    ref = np.asarray(reference_probas, dtype=float)
    cur = np.asarray(current_probas, dtype=float)
    psi = psi_numeric(ref, cur, bins=bins)
    ks = ks_statistic(ref, cur)
    return {
        "psi": psi,
        "ks": ks,
        "status": interpret_psi(psi),
        "n_reference": int(len(ref)),
        "n_current": int(len(cur)),
        "mean_ref": float(np.nanmean(ref)) if len(ref) else None,
        "mean_cur": float(np.nanmean(cur)) if len(cur) else None,
    }
