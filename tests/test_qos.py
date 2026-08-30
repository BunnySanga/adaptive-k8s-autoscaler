from controller.qos import mmc_response_time_seconds, required_replicas_for_qps


def test_response_time_decreases_with_more_replicas() -> None:
    one = mmc_response_time_seconds(arrival_rate=20, service_rate=30, servers=1)
    two = mmc_response_time_seconds(arrival_rate=20, service_rate=30, servers=2)
    assert two < one


def test_required_replicas_respects_bounds() -> None:
    replicas = required_replicas_for_qps(
        qps=500,
        service_rate_per_pod=30,
        slo_seconds=1.0,
        min_replicas=1,
        max_replicas=5,
    )
    assert replicas == 5

