# Novelty 2: Prediction-Stability-Aware Scaling

## Problem Statement

Prediction-assisted autoscalers act on every forecast, even when predictions are noisy. A single noisy prediction can trigger a scale-up, the next minute's prediction corrects, causing a scale-down, and the cycle repeats. This is **scaling flap** — rapid up-down-up oscillations that waste resources on pod startup/teardown without improving service quality.

In our Novelty 1 results, even with confidence-aware scaling, the system produced **355 scaling events** and **71 flaps** across the 2,880-minute test period. Each scaling event has real cost:
- Pod startup latency (container pull, readiness probes)
- Connection draining on scale-down
- Kubernetes API load from frequent HPA updates
- Metrics instability during transitions

The core question: **should the autoscaler act on this prediction right now, or wait to see if the signal is consistent?**

## Our Approach

Add a **stability filter** between the scaling decision and the execution. The filter maintains a sliding window of recent proposed replica counts and only allows changes when the proposals are temporally consistent.

### Three Rules

| Rule | Condition | Default | Rationale |
|------|-----------|---------|-----------|
| **Scale-up** | K consecutive intervals must all propose scaling up | K=3 | Require sustained demand signal before adding pods |
| **Scale-down** | M out of the last N intervals must propose scaling down | M=5, N=7 | Be more conservative removing pods (asymmetric risk) |
| **Burst bypass** | Skip stability check when predicted load > B x current capacity | B=2.0 | Never delay response to genuine traffic bursts |

### Why Asymmetric Windows?

Scale-up and scale-down have different risk profiles:
- **Under-provisioning** (missed scale-up) causes SLO violations — users experience high latency
- **Over-provisioning** (delayed scale-down) wastes resources but doesn't affect users

The stability filter reflects this asymmetry:
- Scale-up uses a strict consecutive check (3 out of 3) — must be consistently needed
- Scale-down uses a relaxed majority check (5 out of 7) — allows some fluctuation but requires a clear trend

### Filter Logic

```
On each decision interval:
  1. Compute proposed_replicas using confidence-aware scaling (Novelty 1)
  2. Append proposed_replicas to history window

  3. If proposed == current:
       → No change needed, pass through

  4. If proposed > current AND decision_qps > 2x current capacity:
       → Burst bypass: scale up immediately (genuine spike)

  5. If proposed > current (scale-up):
       → Check last K entries: all propose scaling up?
       → Yes: allow scale-up
       → No: suppress, keep current replicas

  6. If proposed < current (scale-down):
       → Check last N entries: at least M propose scaling down?
       → Yes: allow scale-down
       → No: suppress, keep current replicas

  7. Cold start (history < required window):
       → Allow the change (fast convergence on startup)
```

### Burst Bypass

The stability filter must not delay response to genuine traffic bursts. When the predicted QPS exceeds 2x the current pod capacity (current_replicas x service_rate_per_pod), the filter is bypassed entirely. This ensures:
- A sudden 3x traffic spike triggers immediate scale-up (no 3-interval delay)
- Gradual increases still go through the stability check
- The 2x threshold is configurable via `--burst-multiplier`

## Architecture

```
Prometheus (metrics)
        |
        v
Prediction Service
(3x LightGBM quantile: P10, P50, P90)
        |
        v
Confidence Score = f(interval width)          ← Novelty 1
        |
        v
Risk-Adjusted QPS Target
(blend P50/P90 based on confidence)           ← Novelty 1
        |
        v
M/M/c Queueing Theory -> Proposed Pod Count
        |
        v
Stability Filter                              ← Novelty 2
(K-consecutive up, M/N down, burst bypass)
        |
        v
KEDA/HPA -> Scale Pods
```

## Implementation

### Core Module: `controller/stability.py`

```python
class StabilityFilter:
    def __init__(self,
        scale_up_consecutive=3,    # K: need K consecutive scale-up proposals
        scale_down_m=5,            # M: need M scale-down proposals...
        scale_down_n=7,            # N: ...out of the last N intervals
        burst_multiplier=2.0,      # bypass when load > 2x current capacity
        service_rate_per_pod=30.0
    )

    def filter(self, proposed_replicas, current_replicas, decision_qps=None)
        -> tuple[int, dict]        # (filtered_replicas, details)
```

The filter is **stateful** — it maintains a sliding window (`collections.deque`) of recent proposed replica counts. The deque has `maxlen=scale_down_n` so old entries are automatically discarded.

### Integration: `controller/autoscaler.py`

Stability is an **orthogonal flag**, not a separate mode. This means:
- It works with any prediction-based mode (`pahpa_style` or `confidence_aware`)
- It does NOT apply to `hpa_baseline` (reactive scaling doesn't benefit from stability filtering)
- It can be enabled/disabled independently via `--stability`

The filter is applied at the end of the `decide()` function, after the mode-specific logic computes proposed replicas. The filter wraps the result and adds `stability` and `pre_stability_replicas` to the details dict.

### Key Files

| File | Purpose |
|------|---------|
| `controller/stability.py` | StabilityFilter class |
| `controller/autoscaler.py` | Integration — stability flag on config, filter in `decide()` |
| `tests/test_stability.py` | 8 unit tests covering all filter behaviors |
| `training/train_final_kaggle.ipynb` | Simulation with 4-system comparison |

### CLI Usage

```bash
# Full system: confidence-aware + stability filter
python3 controller/autoscaler.py --mode confidence_aware --stability

# Custom stability parameters
python3 controller/autoscaler.py --mode confidence_aware --stability \
  --scale-up-consecutive 3 \
  --scale-down-m 5 \
  --scale-down-n 7 \
  --burst-multiplier 2.0

# Dry-run to inspect stability details
python3 controller/autoscaler.py --mode confidence_aware --stability --dry-run \
  --history "10,20,30,40,50,60" --forecast "45,50,52" --current-replicas 2
```

## Results

### Scaling Simulation (M/M/c Queueing, SLO=1s, 2,880-minute test)

| Metric | HPA Baseline | PAHPA (point) | Novelty 1 (conf-aware) | Novelty 1+2 (full system) |
|--------|-------------|---------------|------------------------|--------------------------|
| **SLO violation rate** | 19.55% | 20.38% | 12.92% | **13.12%** |
| P99 latency (ms) | 1000 | 1008 | 1000 | **1000** |
| Avg pods | 2.00 | 1.92 | 2.12 | **2.13** |
| **Scaling events** | 1128 | 440 | 355 | **47** |
| **Flap count** | 228 | 79 | 71 | **0** |
| Over-provisioning | 10.62% | 6.75% | 12.93% | **13.80%** |

### Novelty 2 Improvements Over Novelty 1 Alone

| Metric | Novelty 1 | Novelty 1+2 | Change |
|--------|-----------|-------------|--------|
| Scaling events | 355 | 47 | **87% fewer** |
| Flap count | 71 | 0 | **100% eliminated** |
| SLO violations | 12.92% | 13.12% | +0.2pp (negligible) |
| Avg pods | 2.12 | 2.13 | +0.01 (negligible) |

### Full Pipeline Improvements Over PAHPA Baseline

| Metric | PAHPA | Our Full System | Change |
|--------|-------|-----------------|--------|
| SLO violations | 20.38% | 13.12% | **36% fewer** |
| Scaling events | 440 | 47 | **89% fewer** |
| Flap count | 79 | 0 | **100% eliminated** |
| Avg pods | 1.92 | 2.13 | +0.21 (11% more) |

### Full Pipeline Improvements Over HPA Baseline

| Metric | HPA | Our Full System | Change |
|--------|-----|-----------------|--------|
| SLO violations | 19.55% | 13.12% | **33% fewer** |
| Scaling events | 1128 | 47 | **96% fewer** |
| Flap count | 228 | 0 | **100% eliminated** |
| Avg pods | 2.00 | 2.13 | +0.13 (7% more) |

## Why This Works

### The Flap Problem

Without stability filtering, the autoscaler responds to every prediction. Consider a period where QPS oscillates between 55 and 65 around a threshold that requires 2 vs 3 pods:

```
Minute 1: predict 65 QPS → scale to 3 pods
Minute 2: predict 55 QPS → scale to 2 pods
Minute 3: predict 63 QPS → scale to 3 pods
Minute 4: predict 57 QPS → scale to 2 pods
```

Four scaling events, two flaps, zero benefit — the actual load was stable around 60 QPS the entire time.

### The Stability Solution

With K=3 consecutive scale-up check:

```
Minute 1: predict 65 → propose 3 (1 of 3 consecutive) → keep 2
Minute 2: predict 55 → propose 2 (same as current) → keep 2
Minute 3: predict 63 → propose 3 (1 of 3 consecutive, reset) → keep 2
Minute 4: predict 57 → propose 2 (same as current) → keep 2
```

Zero scaling events, zero flaps — the noisy signal is correctly filtered out.

### The Burst Safety Net

If a genuine traffic spike arrives (e.g., 180 QPS with 2 pods running at capacity 60):

```
decision_qps = 180 > 2.0 x 2 x 30 = 120 → BURST BYPASS
→ Scale to 7 pods immediately, no stability delay
```

The burst bypass ensures the filter never causes an SLO violation during a real traffic surge.

## Relationship to Novelty 1

Novelty 1 and Novelty 2 are complementary layers in the scaling decision pipeline:

| Aspect | Novelty 1 (Confidence-Aware) | Novelty 2 (Stability-Aware) |
|--------|------------------------------|----------------------------|
| **Question answered** | How much should I trust this prediction? | Should I act on this prediction yet? |
| **Mechanism** | Prediction interval width → confidence score | Sliding window of proposals → consistency check |
| **Effect on scaling** | Adjusts target QPS (P50 vs P90) | Suppresses or allows the scaling action |
| **Prevents** | Wrong-magnitude scaling from overconfident predictions | Scaling flap from noisy prediction sequences |
| **Operates on** | Single prediction (current interval) | History of proposals (multiple intervals) |

Together they form a complete decision framework:
1. **Novelty 1** decides *what* the right pod count is (risk-adjusted by confidence)
2. **Novelty 2** decides *when* to actually change the pod count (stability-filtered)

The combined system achieves 36% fewer SLO violations than PAHPA, 89% fewer scaling events, and **zero flaps** — with only 11% more pods on average.

## Test Coverage

8 tests in `tests/test_stability.py`:

| Test | Verifies |
|------|----------|
| `test_scale_up_requires_consecutive` | Scale-up allowed after K consecutive proposals |
| `test_scale_up_suppressed_when_interrupted` | Interrupted streak resets the count |
| `test_scale_down_requires_m_of_n` | Scale-down allowed when M of N proposals agree |
| `test_scale_down_suppressed_insufficient` | Scale-down blocked when fewer than M proposals |
| `test_burst_bypass` | Immediate scale-up when QPS > 2x capacity |
| `test_cold_start_allows_changes` | First calls pass through before window fills |
| `test_no_change_passes_through` | No-change proposals always pass through |
| `test_integration_with_decide` | Filter works end-to-end through `decide()` |

## Configuration Guide

| Parameter | Default | Description | Tuning guidance |
|-----------|---------|-------------|-----------------|
| `scale_up_consecutive` | 3 | Consecutive intervals proposing scale-up before allowing | Lower (2) for faster response, higher (4-5) for more stability |
| `scale_down_m` | 5 | Minimum scale-down proposals needed | Higher = more conservative about removing pods |
| `scale_down_n` | 7 | Window size for scale-down check | Larger window smooths over longer oscillations |
| `burst_multiplier` | 2.0 | Bypass threshold as multiple of current capacity | Lower (1.5) for faster burst response, higher (3.0) for stricter bypass |
