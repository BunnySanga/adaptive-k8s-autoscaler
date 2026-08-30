from __future__ import annotations

import math
import os
import time
from typing import Any

from fastapi import FastAPI, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


APP_NAME = os.getenv("APP_NAME", "adaptive-scaler-app")

REQUEST_COUNT = Counter(
    "app_requests_total",
    "Total HTTP requests handled by the workload app.",
    ["path", "method", "status"],
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Request latency in seconds.",
    ["path"],
    buckets=(0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0),
)
IN_FLIGHT = Gauge("app_in_flight_requests", "Current in-flight requests.")
WORK_UNITS = Counter("app_work_units_total", "Synthetic CPU work units completed.")

app = FastAPI(title="Adaptive Resource Scaling Workload")


def burn_cpu(duration_ms: int, complexity: int) -> dict[str, Any]:
    """Run deterministic floating point work for approximately duration_ms."""
    start = time.perf_counter()
    deadline = start + max(duration_ms, 0) / 1000.0
    iterations = 0
    value = 0.0

    while time.perf_counter() < deadline:
        for i in range(max(complexity, 1)):
            value += math.sin(i + iterations) * math.cos((i + 1) / (iterations + 1))
        iterations += 1

    elapsed = time.perf_counter() - start
    WORK_UNITS.inc(iterations)
    return {
        "duration_seconds": elapsed,
        "iterations": iterations,
        "checksum": round(value, 6),
    }


@app.get("/")
def root() -> dict[str, str]:
    REQUEST_COUNT.labels(path="/", method="GET", status="200").inc()
    return {"service": APP_NAME, "status": "ok"}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    REQUEST_COUNT.labels(path="/healthz", method="GET", status="200").inc()
    return {"status": "healthy"}


@app.get("/work")
def work(
    duration_ms: int = Query(default=80, ge=0, le=5000),
    complexity: int = Query(default=250, ge=1, le=5000),
) -> dict[str, Any]:
    path = "/work"
    IN_FLIGHT.inc()
    started = time.perf_counter()
    try:
        result = burn_cpu(duration_ms=duration_ms, complexity=complexity)
        REQUEST_COUNT.labels(path=path, method="GET", status="200").inc()
        return {
            "service": APP_NAME,
            "requested_duration_ms": duration_ms,
            "complexity": complexity,
            **result,
        }
    finally:
        REQUEST_LATENCY.labels(path=path).observe(time.perf_counter() - started)
        IN_FLIGHT.dec()


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

