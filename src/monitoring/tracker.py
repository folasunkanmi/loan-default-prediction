"""
Lightweight model monitoring:
- Log prediction probabilities and metadata
- Summarise prediction distributions
- Simple population-stability / drift signals vs a reference window
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

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
    """In-memory ring buffer + optional JSONL persistence."""

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

    def psi_vs_reference(
        self,
        reference_probs: List[float],
        last_n: int = 500,
        bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Population Stability Index between a reference probability distribution
        and the most recent live predictions. PSI > 0.25 often flags material drift.
        """
        with self._lock:
            recent = list(self._records)[-last_n:]
        if len(recent) < 30 or len(reference_probs) < 30:
            return {"psi": None, "status": "insufficient_data"}

        live = np.array([r.probability for r in recent], dtype=float)
        ref = np.clip(np.asarray(reference_probs, dtype=float), 1e-6, 1 - 1e-6)
        edges = np.linspace(0, 1, bins + 1)

        ref_hist, _ = np.histogram(ref, bins=edges)
        live_hist, _ = np.histogram(live, bins=edges)
        ref_pct = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-4, None)
        live_pct = np.clip(live_hist / max(live_hist.sum(), 1), 1e-4, None)
        psi = float(np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct)))

        if psi < 0.1:
            status = "stable"
        elif psi < 0.25:
            status = "moderate_shift"
        else:
            status = "significant_drift"

        return {"psi": psi, "status": status, "bins": bins, "n_live": len(live)}

    def summary(self) -> Dict[str, Any]:
        dist = self.prediction_distribution()
        return {"distribution": dist, "log_path": str(LOG_PATH) if self.persist else None}


_monitor: Optional[PredictionMonitor] = None


def get_monitor() -> PredictionMonitor:
    global _monitor
    if _monitor is None:
        _monitor = PredictionMonitor()
    return _monitor
