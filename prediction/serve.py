from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prediction.features import make_prediction_features


MODEL_PATH = Path(os.getenv("MODEL_PATH", "prediction/models/qps_quantile_models.joblib"))
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
QPS_QUERY = os.getenv("QPS_QUERY", 'sum(rate(app_requests_total{path="/work"}[1m]))')

app = FastAPI(title="Multi-Step QPS Prediction Service")


class PredictRequest(BaseModel):
    history: list[float] | None = Field(default=None, description="Recent QPS values.")


class PredictResponse(BaseModel):
    forecast: list[float]
    quantiles: dict[str, float] | None = None
    confidence: float | None = None
    window: int
    steps: int
    source: str


def load_artifact() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def naive_forecast(history: list[float], steps: int) -> list[float]:
    if not history:
        return [0.0] * steps
    if len(history) == 1:
        return [max(0.0, history[-1])] * steps
    slope = (history[-1] - history[0]) / max(len(history) - 1, 1)
    return [max(0.0, history[-1] + slope * offset) for offset in range(1, steps + 1)]


def artifact_window_and_steps(artifact: object) -> tuple[int, int]:
    if isinstance(artifact, dict) and "model" in artifact:
        return int(artifact.get("window", 6)), int(artifact.get("steps", 5))
    if isinstance(artifact, dict) and artifact and all(isinstance(key, int) for key in artifact):
        return int(os.getenv("PREDICTION_WINDOW", "6")), len(artifact)
    return int(os.getenv("PREDICTION_WINDOW", "6")), int(os.getenv("PREDICTION_STEPS", "1"))


def predict_from_artifact(artifact: object, features: np.ndarray, steps: int) -> tuple[list[float], dict | None, float | None]:
    quantiles_out = None
    confidence_out = None

    if isinstance(artifact, dict) and "models" in artifact and "quantiles" in artifact:
        models = artifact["models"]
        transform = artifact.get("target_transform")
        results = {}
        for q_key, model in models.items():
            pred = model.predict(features).reshape(-1)
            if transform == "log1p":
                pred = np.expm1(pred)
            results[q_key] = float(np.clip(pred[0], 0, None))
        p10 = results.get("0.1", 0.0)
        p50 = results.get("0.5", 0.0)
        p90 = results.get("0.9", 0.0)
        width = p90 - p10
        confidence_out = 1.0 - min(width / max(p50, 1.0), 2.0) / 2.0
        quantiles_out = {"p10": p10, "p50": p50, "p90": p90}
        forecast = [p10, p50, p90]
        return forecast, quantiles_out, confidence_out

    if isinstance(artifact, dict) and "model" in artifact:
        prediction = artifact["model"].predict(features).reshape(-1)
        if artifact.get("target_transform") == "log1p":
            prediction = np.expm1(prediction)
    elif hasattr(artifact, "predict"):
        prediction = np.asarray(artifact.predict(features)).reshape(-1)
    else:
        raise HTTPException(status_code=503, detail="unsupported model artifact format")
    return [float(value) for value in np.clip(prediction, 0, None)[:steps]], None, None


def artifact_feature_mode(artifact: object) -> str:
    if isinstance(artifact, dict) and "model" in artifact:
        return str(artifact.get("feature_mode", "raw"))
    return "raw"


def fetch_prometheus_history(window: int) -> list[float]:
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": QPS_QUERY},
        timeout=5,
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("data", {}).get("result", [])
    if not result:
        raise HTTPException(status_code=503, detail="Prometheus query returned no QPS data")
    current = float(result[0]["value"][1])
    return [current] * window


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    artifact = load_artifact()
    window, steps = artifact_window_and_steps(artifact) if artifact else (
        int(os.getenv("PREDICTION_WINDOW", "6")),
        int(os.getenv("PREDICTION_STEPS", "5")),
    )

    if request.history is None:
        history = fetch_prometheus_history(window)
        source = "prometheus"
    else:
        history = request.history
        source = "request"

    if len(history) < window:
        raise HTTPException(status_code=400, detail=f"history must contain at least {window} values")

    if artifact:
        feature_mode = artifact_feature_mode(artifact)
        try:
            features = make_prediction_features(np.asarray(history, dtype=float), window, feature_mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        values, quantiles_out, confidence_out = predict_from_artifact(artifact, features, steps)
    else:
        values = naive_forecast(history[-window:], steps)
        quantiles_out = None
        confidence_out = None
        source = f"{source}:naive_fallback"

    return PredictResponse(
        forecast=values,
        quantiles=quantiles_out,
        confidence=confidence_out,
        window=window,
        steps=steps,
        source=source,
    )
