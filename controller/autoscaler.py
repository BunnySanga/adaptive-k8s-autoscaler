from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

from controller.qos import pahpa_select_replicas, required_replicas_for_qps
from controller.stability import StabilityFilter


def classify_forecast(forecast: list[float], tolerance: float = 0.05) -> str:
    if len(forecast) < 2:
        return "flat"
    first = forecast[0]
    last = forecast[-1]
    baseline = max(abs(first), 1.0)
    change = (last - first) / baseline
    if change > tolerance and all(a <= b for a, b in zip(forecast, forecast[1:])):
        return "rising"
    if change < -tolerance and all(a >= b for a, b in zip(forecast, forecast[1:])):
        return "declining"
    return "flat"


@dataclass
class AutoscalerConfig:
    mode: str
    min_replicas: int = 1
    max_replicas: int = 15
    service_rate_per_pod: float = 30.0
    slo_seconds: float = 1.0
    prediction_url: str = "http://localhost:5000/predict"
    prometheus_url: str = "http://localhost:9090"
    qps_query: str = 'sum(rate(app_requests_total{path="/work"}[1m]))'
    deployment: str = "adaptive-scaler-app"
    namespace: str = "default"
    stability_enabled: bool = False
    scale_up_consecutive: int = 3
    scale_down_m: int = 5
    scale_down_n: int = 7
    burst_multiplier: float = 2.0


def parse_history(raw: str | None) -> list[float]:
    if not raw:
        return []
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def fetch_current_qps(config: AutoscalerConfig) -> float:
    import requests

    response = requests.get(
        f"{config.prometheus_url}/api/v1/query",
        params={"query": config.qps_query},
        timeout=5,
    )
    response.raise_for_status()
    result = response.json().get("data", {}).get("result", [])
    if not result:
        return 0.0
    return float(result[0]["value"][1])


def fetch_forecast(config: AutoscalerConfig, history: list[float]) -> list[float]:
    import requests

    response = requests.post(config.prediction_url, json={"history": history or None}, timeout=5)
    response.raise_for_status()
    data = response.json()
    return [float(value) for value in data["forecast"]]


def hpa_formula(current_replicas: int, current_qps: float, target_qps_per_pod: float, min_replicas: int, max_replicas: int) -> int:
    if target_qps_per_pod <= 0:
        raise ValueError("target_qps_per_pod must be > 0")
    desired = math.ceil(current_qps / target_qps_per_pod) if current_qps > 0 else min_replicas
    return max(min_replicas, min(max_replicas, desired))


def decide(
    config: AutoscalerConfig,
    current_replicas: int,
    current_qps: float,
    forecast: list[float],
    stability_filter: StabilityFilter | None = None,
) -> tuple[int, dict[str, Any]]:
    if config.mode == "hpa_baseline":
        desired = hpa_formula(
            current_replicas,
            current_qps,
            config.service_rate_per_pod,
            config.min_replicas,
            config.max_replicas,
        )
        return desired, {"mode": config.mode, "current_qps": current_qps}

    if not forecast:
        forecast = [current_qps]

    if config.mode == "pahpa_style":
        desired, details = pahpa_select_replicas(
            predicted_qps=forecast[0],
            current_qps=current_qps,
            current_replicas=current_replicas,
            service_rate_per_pod=config.service_rate_per_pod,
            slo_seconds=config.slo_seconds,
            min_replicas=config.min_replicas,
            max_replicas=config.max_replicas,
        )
        details.update({"mode": config.mode, "forecast": forecast})

    elif config.mode == "confidence_aware":
        if len(forecast) < 3:
            p10 = forecast[0] * 0.8
            p50 = forecast[0]
            p90 = forecast[0] * 1.2
        else:
            p10, p50, p90 = forecast[0], forecast[1], forecast[2]

        width = p90 - p10
        confidence = 1.0 - min(width / max(p50, 1.0), 2.0) / 2.0

        if confidence > 0.7:
            decision_qps = p50
        elif confidence > 0.3:
            decision_qps = 0.5 * p50 + 0.5 * p90
        else:
            decision_qps = p90

        decision_qps = max(decision_qps, current_qps)

        desired = required_replicas_for_qps(
            decision_qps,
            config.service_rate_per_pod,
            config.slo_seconds,
            config.min_replicas,
            config.max_replicas,
        )

        details = {
            "mode": config.mode,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "confidence": confidence,
            "decision_qps": decision_qps,
            "proposed_replicas": desired,
        }

    else:
        raise ValueError(f"unsupported mode: {config.mode}")

    if stability_filter is not None:
        dq = details.get("decision_qps", details.get("predicted_qps", current_qps))
        filtered, stability_details = stability_filter.filter(
            proposed_replicas=desired,
            current_replicas=current_replicas,
            decision_qps=dq,
        )
        details["pre_stability_replicas"] = desired
        details["stability"] = stability_details
        desired = filtered

    return desired, details


def patch_deployment_replicas(config: AutoscalerConfig, replicas: int) -> None:
    from kubernetes import client, config as kube_config

    try:
        kube_config.load_incluster_config()
    except Exception:
        kube_config.load_kube_config()

    api = client.AppsV1Api()
    body = {"spec": {"replicas": replicas}}
    api.patch_namespaced_deployment_scale(config.deployment, config.namespace, body)


def run_once(
    args: argparse.Namespace,
    stability_filter: StabilityFilter | None = None,
) -> dict[str, Any]:
    config = AutoscalerConfig(
        mode=args.mode,
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas,
        service_rate_per_pod=args.service_rate_per_pod,
        slo_seconds=args.slo_seconds,
        prediction_url=args.prediction_url,
        prometheus_url=args.prometheus_url,
        deployment=args.deployment,
        namespace=args.namespace,
        stability_enabled=getattr(args, "stability", False),
    )
    history = parse_history(args.history)
    current_qps = history[-1] if history else fetch_current_qps(config)
    if args.forecast:
        forecast = parse_history(args.forecast)
    elif args.dry_run and history:
        recent = history[-5:] if len(history) >= 5 else [current_qps] * 5
        forecast = recent
    else:
        forecast = fetch_forecast(config, history)
    desired, details = decide(
        config, args.current_replicas, current_qps, forecast,
        stability_filter=stability_filter,
    )

    result = {"desired_replicas": desired, "details": details}
    if not args.dry_run:
        patch_deployment_replicas(config, desired)
        result["patched"] = True
    else:
        result["patched"] = False
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediction-assisted autoscaler controller.")
    parser.add_argument(
        "--mode",
        choices=["hpa_baseline", "pahpa_style", "confidence_aware"],
        default="confidence_aware",
        help="Use confidence_aware for quantile-based risk-adjusted scaling.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true", help="Run one decision interval and exit.")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--history", help="Comma-separated recent QPS values for local testing.")
    parser.add_argument("--forecast", help="Comma-separated forecast values for local testing.")
    parser.add_argument("--current-replicas", type=int, default=int(os.getenv("CURRENT_REPLICAS", "1")))
    parser.add_argument("--min-replicas", type=int, default=1)
    parser.add_argument("--max-replicas", type=int, default=15)
    parser.add_argument("--service-rate-per-pod", type=float, default=30.0)
    parser.add_argument("--slo-seconds", type=float, default=1.0)
    parser.add_argument("--prediction-url", default=os.getenv("PREDICTION_URL", "http://localhost:5000/predict"))
    parser.add_argument("--prometheus-url", default=os.getenv("PROMETHEUS_URL", "http://localhost:9090"))
    parser.add_argument("--deployment", default=os.getenv("DEPLOYMENT_NAME", "adaptive-scaler-app"))
    parser.add_argument("--namespace", default=os.getenv("NAMESPACE", "default"))
    parser.add_argument("--stability", action="store_true", help="Enable prediction-stability filter.")
    parser.add_argument("--scale-up-consecutive", type=int, default=3)
    parser.add_argument("--scale-down-m", type=int, default=5)
    parser.add_argument("--scale-down-n", type=int, default=7)
    parser.add_argument("--burst-multiplier", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    sf: StabilityFilter | None = None
    if parsed.stability:
        sf = StabilityFilter(
            scale_up_consecutive=parsed.scale_up_consecutive,
            scale_down_m=parsed.scale_down_m,
            scale_down_n=parsed.scale_down_n,
            burst_multiplier=parsed.burst_multiplier,
            service_rate_per_pod=parsed.service_rate_per_pod,
        )
    while True:
        print(json.dumps(run_once(parsed, stability_filter=sf), indent=2))
        if parsed.once or parsed.dry_run:
            break
        time.sleep(parsed.interval_seconds)
