import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List
from src.monitoring.drift import detect_prediction_drift, score_prediction_drift_psi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITOR_DIR = PROJECT_ROOT / "reports" / "monitoring"
LOG_PATH = MONITOR_DIR / "predictions.jsonl"

@dataclass
class PredictionRecord:
    timestamp: float
    probability: float
    predicted_class: int
    threshold: float
    model_name: str
    schema: str
    risk_tier: str

class PredictionMonitor:
    def __init__(self, max_records: int = 5000, persist: bool = True):
        self._lock = threading.Lock()
        self._records: Deque[PredictionRecord] = deque(maxlen=max_records)
        self.persist = persist
        MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    def log_prediction(self, probability: float, predicted_class: int, threshold: float, model_name: str, schema: str, risk_tier: str) -> None:
        rec = PredictionRecord(timestamp=time.time(), probability=probability, predicted_class=predicted_class, threshold=threshold, model_name=model_name, schema=schema, risk_tier=risk_tier)
        with self._lock:
            self._records.append(rec)
            if self.persist:
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(rec)) + "\n")
    # Back-compat alias used by Streamlit/API
    def log(self, probability: float, predicted_class: int, threshold: float, model_name: str, schema: str, risk_tier: str) -> None:
        self.log_prediction(probability, predicted_class, threshold, model_name, schema, risk_tier)
    def recent_probas(self, last_n: int = 500) -> List[float]:
        with self._lock:
            recs = list(self._records)[-last_n:]
        return [r.probability for r in recs]
    def score_drift_vs_reference(self, reference_probas: List[float], last_n: int = 500) -> Dict[str, Any]:
        current = self.recent_probas(last_n)
        return detect_prediction_drift(reference_probas, current)
    def psi_vs_reference(self, reference_probas: List[float], last_n: int = 500) -> Dict[str, Any]:
        current = self.recent_probas(last_n)
        return score_prediction_drift_psi(reference_probas, current)
    def prediction_distribution(self, last_n: int | None = None) -> Dict[str, Any]:
        with self._lock:
            records = list(self._records)
        if last_n is not None:
            records = records[-last_n:]
        if not records:
            return {"count": 0, "mean_prob": None, "std_prob": None, "class_rate": None}
        probs = [r.probability for r in records]
        classes = [r.predicted_class for r in records]
        return {
            "count": len(records),
            "mean_prob": float(sum(probs) / len(probs)),
            "std_prob": float(np_std(probs)),
            "class_rate": float(sum(classes) / len(classes)),
        }
    def summary(self, reference_probas: List[float] | None = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"distribution": self.prediction_distribution(), "log_path": str(LOG_PATH) if self.persist else None}
        if reference_probas:
            out["score_drift"] = self.score_drift_vs_reference(reference_probas)
            out["psi"] = self.psi_vs_reference(reference_probas)
        return out

def np_std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

_monitor: PredictionMonitor | None = None
def get_monitor() -> PredictionMonitor:
    global _monitor
    if _monitor is None:
        _monitor = PredictionMonitor()
    return _monitor
