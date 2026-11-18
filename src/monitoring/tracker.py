"""
Lightweight prediction logging + distribution summaries.
Integrates with drift helpers for score-distribution monitoring.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from src.monitoring.drift import detect_prediction_drift

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
        if persist:
            MONITOR_DIR.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        probability: float,
        predicted_class: int,
        threshold: float,
        model_name: str,
        schema: str,
        risk_tier: str,
    ) -> None:
        rec = PredictionRecord(
            timestamp=time.time(),
            probability=float(probability),
            predicted_class=int(predicted_class),
            threshold=float(threshold),
            model_name=model_name,
            schema=schema,
            risk_tier=risk_tier,
        )
        with self._lock:
            self._records.append(rec)
            if self.persist:
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(rec)) + "\n")

    def recent_probas(self, last_n: Optional[int] = None) -> List[float]:
        with self._lock:
            records = list(self._records)
        if last_n is not None:
            records = records[-last_n:]
        return [r.probability for r in records]

    def prediction_distribution(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            records = list(self._records)
        if last_n is not None:
            records = records[-last_n:]
        if not records:
            return {"count": 0, "mean_prob": None, "std_prob": None, "class_rate": None}

        probs = np.array([r.probability for r in records], dtype=float)
        classes = np.array([r.predicted_class for r in records], dtype=int)
        tiers = [r.risk_tier for r in records]
        tier_counts = {t: tiers.count(t) for t in sorted(set(tiers))}

        return {
            "count": len(records),
            "mean_prob": float(probs.mean()),
            "std_prob": float(probs.std()),
            "p50": float(np.percentile(probs, 50)),
            "p90": float(np.percentile(probs, 90)),
            "class_rate": float(classes.mean()),
            "tier_counts": tier_counts,
            "histogram": np.histogram(probs, bins=10, range=(0, 1))[0].tolist(),
        }

    def score_drift_vs_reference(
        self,
        reference_probas: List[float],
        last_n: int = 500,
    ) -> Dict[str, Any]:
        current = self.recent_probas(last_n)
        return detect_prediction_drift(reference_probas, current)

    def summary(self, reference_probas: Optional[List[float]] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "distribution": self.prediction_distribution(),
            "log_path": str(LOG_PATH) if self.persist else None,
        }
        if reference_probas:
            out["score_drift"] = self.score_drift_vs_reference(reference_probas)
        return out


_monitor: Optional[PredictionMonitor] = None


def get_monitor() -> PredictionMonitor:
    global _monitor
    if _monitor is None:
        _monitor = PredictionMonitor()
    return _monitor
