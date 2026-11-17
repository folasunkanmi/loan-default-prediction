"""
FastAPI service exposing the trained loan-default model.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health
    GET  /metrics
    POST /predict
    POST /predict/batch
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import load_best_model
from src.monitoring.tracker import get_monitor
from src.schemas.prediction import PredictionRequest, PredictionResponse, validate_and_build_row

app = FastAPI(
    title="Loan Default Prediction API",
    description="REST API for probability-of-default scoring with dual-schema support.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_artifact: Optional[Dict[str, Any]] = None


def _load_artifact() -> Dict[str, Any]:
    global _artifact
    if _artifact is None:
        try:
            _artifact = load_best_model()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="Model not trained. Run: python -m src.models.train",
            ) from exc
    return _artifact


def _risk_tier(proba: float) -> str:
    if proba < 0.30:
        return "Low Risk"
    if proba < 0.55:
        return "Medium Risk"
    return "High Risk"


@app.on_event("startup")
def startup() -> None:
    try:
        _load_artifact()
    except HTTPException:
        # Allow API to boot without a model so /health still works in CI.
        pass


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        art = _load_artifact()
        return {
            "status": "ok",
            "model_loaded": True,
            "model_name": art["model_name"],
            "schema": art.get("schema", "real"),
            "threshold": art["threshold"],
        }
    except HTTPException:
        return {"status": "degraded", "model_loaded": False}


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    monitor = get_monitor()
    return monitor.summary()


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    art = _load_artifact()
    schema = req.schema or art.get("schema", "real")
    if schema not in ("real", "synthetic"):
        raise HTTPException(status_code=400, detail="schema must be 'real' or 'synthetic'")

    try:
        X = validate_and_build_row(req.features, schema=schema)  # type: ignore[arg-type]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Input validation failed: {exc}") from exc

    # Schema mismatch guard: trained pipeline expects specific columns
    trained_schema = art.get("schema", "real")
    if schema != trained_schema:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Request schema '{schema}' does not match trained model schema "
                f"'{trained_schema}'. Retrain on the desired schema or send matching features."
            ),
        )

    try:
        proba = float(art["pipeline"].predict_proba(X)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    threshold = float(art["threshold"])
    pred = int(proba >= threshold)
    tier = _risk_tier(proba)

    get_monitor().log(
        probability=proba,
        predicted_class=pred,
        threshold=threshold,
        model_name=art["model_name"],
        schema=schema,
        risk_tier=tier,
    )

    return PredictionResponse(
        default_probability=round(proba, 6),
        predicted_class=pred,
        risk_tier=tier,
        threshold=threshold,
        model_name=art["model_name"],
        schema=schema,
    )


class BatchRequest(BaseModel):
    schema: Optional[str] = None
    records: List[Dict[str, Any]] = Field(..., min_length=1, max_length=500)


@app.post("/predict/batch")
def predict_batch(req: BatchRequest) -> Dict[str, Any]:
    results = []
    for i, feats in enumerate(req.records):
        try:
            resp = predict(PredictionRequest(schema=req.schema, features=feats))  # type: ignore[arg-type]
            results.append({"index": i, "ok": True, **resp.model_dump()})
        except HTTPException as exc:
            results.append({"index": i, "ok": False, "error": exc.detail})
    return {"n": len(results), "results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
