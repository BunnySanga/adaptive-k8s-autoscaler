from __future__ import annotations

import math


def mmc_response_time_seconds(arrival_rate: float, service_rate: float, servers: int) -> float:
    """Return M/M/c mean response time in seconds.

    arrival_rate and service_rate are expressed in requests per second.
    """
    if servers < 1:
        raise ValueError("servers must be >= 1")
    if service_rate <= 0:
        raise ValueError("service_rate must be > 0")
    if arrival_rate <= 0:
        return 1.0 / service_rate

    utilization = arrival_rate / (servers * service_rate)
    if utilization >= 1:
        return math.inf

    offered_load = arrival_rate / service_rate
    p0_terms = sum((offered_load**k) / math.factorial(k) for k in range(servers))
    tail = (offered_load**servers) / (math.factorial(servers) * (1 - utilization))
    p0 = 1.0 / (p0_terms + tail)
    erlang_c = tail * p0
    wait_queue = erlang_c / (servers * service_rate - arrival_rate)
    return wait_queue + (1.0 / service_rate)


def required_replicas_for_qps(
    qps: float,
    service_rate_per_pod: float,
    slo_seconds: float,
    min_replicas: int,
    max_replicas: int,
) -> int:
    """Find the smallest replica count satisfying the latency SLO."""
    lower = max(1, min_replicas)
    upper = max(lower, max_replicas)
    for replicas in range(lower, upper + 1):
        if mmc_response_time_seconds(qps, service_rate_per_pod, replicas) <= slo_seconds:
            return replicas
    return upper


def pahpa_select_replicas(
    predicted_qps: float,
    current_qps: float,
    current_replicas: int,
    service_rate_per_pod: float,
    slo_seconds: float,
    min_replicas: int,
    max_replicas: int,
) -> tuple[int, dict[str, float | int]]:
    """PAHPA-style decision using predicted and real-time QPS candidates."""
    predicted_replicas = required_replicas_for_qps(
        predicted_qps, service_rate_per_pod, slo_seconds, min_replicas, max_replicas
    )
    current_replicas_needed = required_replicas_for_qps(
        current_qps, service_rate_per_pod, slo_seconds, min_replicas, max_replicas
    )

    predicted_latency = mmc_response_time_seconds(predicted_qps, service_rate_per_pod, predicted_replicas)
    current_latency = mmc_response_time_seconds(current_qps, service_rate_per_pod, current_replicas_needed)
    predicted_ok = predicted_latency <= slo_seconds
    current_ok = current_latency <= slo_seconds

    if predicted_ok and current_ok:
        desired = min(predicted_replicas, current_replicas_needed)
    elif predicted_ok:
        desired = predicted_replicas
    elif current_ok:
        desired = current_replicas_needed
    else:
        desired = min(max(predicted_replicas, current_replicas_needed) + 1, max_replicas)

    desired = max(min_replicas, min(max_replicas, desired))
    return desired, {
        "predicted_replicas": predicted_replicas,
        "current_replicas_needed": current_replicas_needed,
        "predicted_latency_seconds": predicted_latency,
        "current_latency_seconds": current_latency,
        "current_replicas": current_replicas,
    }

