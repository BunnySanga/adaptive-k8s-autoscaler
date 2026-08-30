from controller.autoscaler import AutoscalerConfig, decide


def test_confidence_aware_uses_median_when_confident() -> None:
    config = AutoscalerConfig(mode="confidence_aware")
    # Narrow interval: P10=45, P50=50, P90=52 → high confidence → use P50
    _, details = decide(config, current_replicas=1, current_qps=30, forecast=[45, 50, 52])
    assert details["confidence"] > 0.7
    assert details["decision_qps"] == 50


def test_confidence_aware_uses_upper_when_uncertain() -> None:
    config = AutoscalerConfig(mode="confidence_aware")
    # Wide interval: P10=10, P50=50, P90=120 → low confidence → use P90
    _, details = decide(config, current_replicas=1, current_qps=30, forecast=[10, 50, 120])
    assert details["confidence"] <= 0.3
    assert details["decision_qps"] == 120


def test_confidence_aware_floor_is_current_qps() -> None:
    config = AutoscalerConfig(mode="confidence_aware")
    # Even with low prediction, decision_qps should not go below current_qps
    _, details = decide(config, current_replicas=1, current_qps=80, forecast=[5, 10, 15])
    assert details["decision_qps"] >= 80


def test_pahpa_still_works() -> None:
    config = AutoscalerConfig(mode="pahpa_style")
    desired, details = decide(config, current_replicas=1, current_qps=42, forecast=[50])
    assert desired >= 1
