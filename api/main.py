"""
FastAPI service for loan-default PD scoring.

    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health
    GET  /metrics
    GET  /model/info
    POST /predict
    POST /predict/batch
    POST /drift/scores

CORS:
    Default allow_origins=["*"] with allow_credentials=False.
    Set CORS_ALLOW_ORIGINS to a comma-separated list for production.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import RISK_LOW, RISK_MEDIUM, VERSION
from src.models.train import load_best_model
from src.monitoring.drift import detect_prediction_drift
from src.monitoring.tracker import get_monitor
from src.schemas.prediction import PredictionRequest, PredictionResponse, validate_and_build_row

APP_VERSION = VERSION

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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    try:
        _load_artifact()
    except HTTPException:
        pass
    yield


app = FastAPI(
    title="Loan Default Prediction API",
    description="REST API for probability-of-default scoring with dual-schema support, SHAP drivers, and monitoring.",
    version=APP_VERSION,
    lifespan=_lifespan,
)

_cors_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
_cors_origins = (
    ["*"]
    if _cors_origins_env == "*"
    else [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
)
_cors_credentials = False if _cors_origins == ["*"] else True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _risk_tier(proba: float) -> str:
    if proba < RISK_LOW:
        return "Low Risk"
    if proba < RISK_MEDIUM:
        return "Medium Risk"
    return "High Risk"


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        art = _load_artifact()
        return {
            "status": "ok",
            "api_version": APP_VERSION,
            "model_loaded": True,
            "model_name": art["model_name"],
            "schema": art.get("schema", "real"),
            "threshold": art["threshold"],
            "calibrated": art.get("calibrated", False),
        }
    except HTTPException:
        return {"status": "degraded", "api_version": APP_VERSION, "model_loaded": False}


@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    art = _load_artifact()
    return {
        "model_name": art["model_name"],
        "schema": art.get("schema"),
        "threshold": art["threshold"],
        "calibrated": art.get("calibrated", False),
        "calibration_method": art.get("calibration_method"),
        "feature_columns": art.get("feature_columns"),
        "test_metrics": art.get("test_metrics"),
        "api_version": APP_VERSION,
    }


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    art = None
    try:
        art = _load_artifact()
    except HTTPException:
        pass
    ref = art.get("reference_probas") if art else None
    return get_monitor().summary(reference_probas=ref)


@app.post("/predict", response_model=PredictionResponse)
def predict(
    req: PredictionRequest,
    include_shap: bool = Query(default=True, description="Attach top SHAP drivers"),
) -> PredictionResponse:
    art = _load_artifact()
    trained_schema = art.get("schema", "real")
    schema = getattr(req, "data_schema", None) or getattr(req, "schema", None) or trained_schema
    if schema not in ("real", "synthetic"):
        raise HTTPException(status_code=400, detail="schema must be 'real' or 'synthetic'")
    if schema != trained_schema:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Request schema '{schema}' does not match trained model schema "
                f"'{trained_schema}'."
            ),
        )

    try:
        X = validate_and_build_row(req.features, schema=schema)  # type: ignore[arg-type]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Input validation failed: {exc}") from expc

    feature_columns = art.get("feature_columns") or []
    if feature_columns:
        missing = [c for c in feature_columns if c not in X.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing columns: {missing}")
        X = X[feature_columns]

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

    drivers = None
    if include_shap:
        try:
            from src.explainability.shap_analysis import explain_instance

            bg = art.get("shap_background")
            drivers = explain_instance(art["pipeline"], X, top_k=5, background=bg)
        except Exception:
            drivers = None

    return PredictionResponse(
        default_probability=round(proba, 6),
        predicted_class=pred,
        risk_tier=tier,
        threshold=threshold,
        model_name=art["model_name"],
        schema=schema,
        top_drivers=drivers,
    )


class BatchRequest(BaseModel):
    schema: Optional[str] = None
    records: List[Dict[str, Any]] = Field(..., min_length=1, max_length=500)
    include_shap: bool = False


@app.post("/predict/batch")
def predict_batch(req: BatchRequest) -> Dict[str, Any]:
    results = []
    for i, feats in enumerate(req.records):
        try:
            resp = predict(
                PredictionRequest(schema=req.schema, features=feats),  # type: ignore[arg-type]
                include_shap=req.include_shap,
            )
            results.append({"index": i, "ok": True, **resp.model_dump()})
        except HTTPException as exc:
            results.append({"index": i, "ok": False, "error": exc.detail})
    return {"n": len(results), "results": results}


class DriftScoreRequest(BaseModel):
    probabilities: List[float] = Field(..., min_length=10, max_length=50_000)


@app.post("/drift/scores")
def drift_scores(req: DriftScoreRequest) -> Dict[str, Any]:
    art = _load_artifact()
    ref = art.get("reference_probas") or []
    if len(ref) < 10:
        raise HTTPException(status_code=503, detail="No reference score distribution in artifact")
    return detect_prediction_drift(ref, req.probabilities)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
