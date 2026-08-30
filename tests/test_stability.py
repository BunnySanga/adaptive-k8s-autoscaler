from controller.stability import StabilityFilter


def test_scale_up_requires_consecutive() -> None:
    sf = StabilityFilter(scale_up_consecutive=3, scale_down_n=7)
    # First two proposals: suppressed (only 2 of 3 consecutive)
    r1, d1 = sf.filter(proposed_replicas=4, current_replicas=2)
    assert d1["stability_action"] == "allowed"  # cold start (history < K)

    r2, d2 = sf.filter(proposed_replicas=4, current_replicas=2)
    assert d2["stability_action"] == "allowed"  # cold start (history < K)

    r3, d3 = sf.filter(proposed_replicas=4, current_replicas=2)
    assert r3 == 4
    assert d3["stability_action"] == "allowed"
    assert d3["consecutive_up"] == 3


def test_scale_up_suppressed_when_interrupted() -> None:
    sf = StabilityFilter(scale_up_consecutive=3, scale_down_n=7)
    # Fill history past cold-start
    for _ in range(3):
        sf.filter(proposed_replicas=2, current_replicas=2)

    # Two scale-ups, then a no-change interrupts the streak
    sf.filter(proposed_replicas=4, current_replicas=2)
    sf.filter(proposed_replicas=4, current_replicas=2)
    sf.filter(proposed_replicas=2, current_replicas=2)  # interrupts

    r, d = sf.filter(proposed_replicas=4, current_replicas=2)
    assert r == 2
    assert d["stability_action"] == "suppressed"


def test_scale_down_requires_m_of_n() -> None:
    sf = StabilityFilter(scale_up_consecutive=3, scale_down_m=3, scale_down_n=5)
    # Fill history to get past cold-start
    for _ in range(5):
        sf.filter(proposed_replicas=4, current_replicas=4)

    # Now propose scale-down: need 3 of 5 recent to be < current
    sf.filter(proposed_replicas=2, current_replicas=4)  # 1 down
    sf.filter(proposed_replicas=2, current_replicas=4)  # 2 down
    r, d = sf.filter(proposed_replicas=2, current_replicas=4)  # 3 down → allowed
    assert r == 2
    assert d["stability_action"] == "allowed"
    assert d["down_count"] >= 3


def test_scale_down_suppressed_insufficient() -> None:
    sf = StabilityFilter(scale_up_consecutive=3, scale_down_m=4, scale_down_n=5)
    # Fill with no-change to get past cold-start
    for _ in range(5):
        sf.filter(proposed_replicas=4, current_replicas=4)

    # Only 2 of 5 propose down — not enough (need 4)
    sf.filter(proposed_replicas=2, current_replicas=4)
    sf.filter(proposed_replicas=2, current_replicas=4)
    sf.filter(proposed_replicas=4, current_replicas=4)
    sf.filter(proposed_replicas=4, current_replicas=4)
    r, d = sf.filter(proposed_replicas=2, current_replicas=4)
    assert r == 4
    assert d["stability_action"] == "suppressed"


def test_burst_bypass() -> None:
    sf = StabilityFilter(
        scale_up_consecutive=3,
        burst_multiplier=2.0,
        service_rate_per_pod=30.0,
    )
    # Fill history with no-change to get past cold-start
    for _ in range(3):
        sf.filter(proposed_replicas=2, current_replicas=2)

    # Current capacity = 2 * 30 = 60 QPS. Decision QPS = 130 > 2*60 = 120
    r, d = sf.filter(proposed_replicas=5, current_replicas=2, decision_qps=130.0)
    assert r == 5
    assert d["stability_action"] == "burst_bypass"


def test_cold_start_allows_changes() -> None:
    sf = StabilityFilter(scale_up_consecutive=3)
    r, d = sf.filter(proposed_replicas=4, current_replicas=2)
    assert r == 4
    assert d["reason"] == "cold_start"


def test_no_change_passes_through() -> None:
    sf = StabilityFilter()
    r, d = sf.filter(proposed_replicas=3, current_replicas=3)
    assert r == 3
    assert d["stability_action"] == "no_change"


def test_integration_with_decide() -> None:
    from controller.autoscaler import AutoscalerConfig, decide

    config = AutoscalerConfig(mode="confidence_aware")
    sf = StabilityFilter(scale_up_consecutive=3)

    # Fill past cold-start with no-change
    for _ in range(3):
        sf.filter(proposed_replicas=1, current_replicas=1)

    # First call with high forecast — should be suppressed (only 1 of 3 consecutive)
    desired1, details1 = decide(
        config, current_replicas=1, current_qps=10,
        forecast=[45, 50, 52], stability_filter=sf,
    )
    assert "stability" in details1
    assert details1["pre_stability_replicas"] >= 1

    # Second call
    desired2, details2 = decide(
        config, current_replicas=1, current_qps=10,
        forecast=[45, 50, 52], stability_filter=sf,
    )

    # Third call — should now be allowed after 3 consecutive
    desired3, details3 = decide(
        config, current_replicas=1, current_qps=10,
        forecast=[45, 50, 52], stability_filter=sf,
    )
    assert details3["stability"]["stability_action"] == "allowed"
    assert desired3 == details3["pre_stability_replicas"]
