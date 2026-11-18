from typing import List, Dict, Any
import numpy as np

def detect_prediction_drift(reference_probas: List[float], current_probas: List[float]) -> Dict[str, Any]:
    if not reference_probas or not current_probas:
        return {"drift_score": 0.0, "drift_detected": False, "mean_ref": 0, "mean_curr": 0}
    mean_ref = float(np.mean(reference_probas))
    mean_curr = float(np.mean(current_probas))
    drift_score = abs(mean_curr - mean_ref)
    return {"drift_score": drift_score, "drift_detected": drift_score > 0.1, "mean_ref": mean_ref, "mean_curr": mean_curr}

def score_prediction_drift_psi(reference_probas: List[float], current_probas: List[float], bins: int = 10) -> Dict[str, Any]:
    if not reference_probas or not current_probas:
        return {"psi": 0.0, "drift_detected": False, "level": "no_change"}
    ref = np.array(reference_probas)
    curr = np.array(current_probas)
    breakpoints = np.histogram_bin_edges(ref, bins=bins)
    eps = 1e-6
    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    curr_counts, _ = np.histogram(curr, bins=breakpoints)
    ref_perc = ref_counts / len(ref)
    curr_perc = curr_counts / len(curr)
    ref_perc = np.where(ref_perc == 0, eps, ref_perc)
    curr_perc = np.where(curr_perc == 0, eps, curr_perc)
    psi_values = (curr_perc - ref_perc) * np.log(curr_perc / ref_perc)
    psi = float(np.sum(psi_values))
    level = "no_change" if psi < 0.1 else "small_change" if psi < 0.25 else "large_change"
    return {"psi": psi, "drift_detected": psi >= 0.1, "level": level, "reference_size": len(ref), "current_size": len(curr)}

def detect_feature_drift(reference_features, current_features) -> Dict[str, Any]:
    try:
        ref = np.array(reference_features).flatten()
        curr = np.array(current_features).flatten()
        if len(ref)==0 or len(curr)==0:
            return {"drift_detected": False, "drift_score": 0.0}
        score = float(abs(np.mean(curr) - np.mean(ref)))
        return {"drift_detected": score > 0.1, "drift_score": score}
    except Exception:
        return {"drift_detected": False, "drift_score": 0.0}
