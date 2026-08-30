# Novelty 1: Confidence-Aware Scaling via Quantile Regression

## Problem Statement

The PAHPA base paper uses **point prediction** (single QPS forecast) to drive autoscaling decisions. The autoscaler blindly trusts this prediction — when it's wrong:
- Over-prediction leads to over-provisioning (wasted resources)
- Under-prediction leads to SLO violations (high latency)

There is no mechanism to express prediction uncertainty. A forecast of "50 QPS" could mean "I'm very sure it's 50" or "it could be anywhere from 10 to 120." The scaling decision should differ in these cases.

## Our Approach

Replace the single point-prediction LightGBM with **three quantile regression models** that produce prediction intervals:

| Model | Quantile | Role |
|-------|----------|------|
| P10 | alpha=0.1 | Optimistic lower bound |
| P50 | alpha=0.5 | Median (best estimate) |
| P90 | alpha=0.9 | Conservative upper bound |

### Confidence Score

The prediction interval width measures confidence:

```
confidence = 1 - clip((P90 - P10) / max(P50, 1.0), 0, 2) / 2.0
```

- Narrow interval (P90 ~ P10) -> high confidence (close to 1.0)
- Wide interval (P90 >> P10) -> low confidence (close to 0.0)

### Risk-Adjusted Scaling Decision

The autoscaler uses confidence to blend between efficiency and safety:

```
if confidence > 0.7:
    target_qps = P50              # confident -> scale to median (efficient)
elif confidence > 0.3:
    target_qps = 0.5*P50 + 0.5*P90  # moderate -> blend median + upper
else:
    target_qps = P90              # uncertain -> scale to upper bound (safe)
```

The target QPS is then fed into M/M/c queueing theory to determine the pod count, exactly as in PAHPA.

## Architecture

```
Prometheus (metrics)
        |
        v
Prediction Service
(3x LightGBM quantile: P10, P50, P90)
        |
        v
Confidence Score = f(interval width)
        |
        v
Risk-Adjusted QPS Target
(blend P50/P90 based on confidence)
        |
        v
M/M/c Queueing Theory -> Pod Count
        |
        v
KEDA/HPA -> Scale Pods
```

## Implementation

### Training (Kaggle Notebook)

- **Dataset:** NASA-HTTP July 1995 access logs, parsed to minute-level QPS (36,520 data points)
- **Features:** 151 enhanced features per window of 60 minutes (raw lags, first differences, multi-scale rolling statistics, global stats like slope and range)
- **Target transform:** log1p (stabilizes training on skewed QPS distribution)
- **Models:** 4 LightGBM models total (1 point + 3 quantile), same hyperparameters:
  - n_estimators=500, learning_rate=0.035, num_leaves=63
- **Train/test split:** Last 2,880 samples for testing

### Key Files

| File | Purpose |
|------|---------|
| `training/train_final_kaggle.ipynb` | Training notebook (run on Kaggle) |
| `prediction/models/qps_point_model.joblib` | PAHPA baseline model |
| `prediction/models/qps_quantile_models.joblib` | P10/P50/P90 quantile models |
| `prediction/serve.py` | Prediction API (returns forecast + quantiles + confidence) |
| `controller/autoscaler.py` | Autoscaler with `confidence_aware` mode |
| `controller/qos.py` | M/M/c queueing theory calculations |

## Results

### Prediction Accuracy

| System | MAPE | MAE | RMSE | R-squared |
|--------|------|-----|------|-----------|
| Paper SLMD-LightGBM | 24.51% | 7.59 | 10.53 | 0.81 |
| Our PAHPA (point) | 32.02% | 11.17 | 14.95 | 0.73 |
| Novelty1 P50 (median) | 32.95% | 11.04 | 14.75 | 0.74 |

Note: The prediction accuracy gap with the paper is due to our use of enhanced features instead of LMD (Local Mean Decomposition), which had numerical overflow issues in implementation. This is a documented deviation.

**Novelty 1 does not aim to improve prediction accuracy.** It improves how predictions are used in scaling decisions.

### Prediction Interval Analysis

| Metric | Value |
|--------|-------|
| Coverage (P10-P90) | 70.7% |
| Average interval width | 29.61 QPS |
| Median interval width | 28.65 QPS |

### Confidence Distribution

| Level | Threshold | Percentage | Action |
|-------|-----------|------------|--------|
| High | > 0.7 | 24.5% | Scale to median (efficient) |
| Medium | 0.3 - 0.7 | 72.8% | Blend median + upper |
| Low | <= 0.3 | 2.6% | Scale to upper bound (safe) |

Mean confidence score: 0.588

### Scaling Simulation (M/M/c Queueing, SLO=1s)

| Metric | HPA Baseline | PAHPA (point) | Ours (confidence-aware) |
|--------|-------------|---------------|------------------------|
| **SLO violation rate** | 19.55% | 20.38% | **12.92%** |
| P99 latency (ms) | 1000 | 1008 | 1000 |
| Avg pods | 2.00 | 1.92 | 2.12 |
| Scaling events | 1128 | 440 | **355** |
| Flap count | 228 | 79 | **71** |
| Over-provisioning | 10.62% | 6.75% | 12.93% |

### Key Improvements Over PAHPA

| Metric | Improvement |
|--------|-------------|
| SLO violations | **37% fewer** (20.38% -> 12.92%) |
| Scaling events | **19% fewer** (440 -> 355) |
| Flap count | **10% fewer** (79 -> 71) |
| Cost | +0.2 avg pods (1.92 -> 2.12) |

## Why This Works

PAHPA blindly trusts its point prediction. When the prediction is wrong (which happens ~32% of the time by MAPE), the autoscaler either over-provisions or under-provisions with no middle ground.

Our confidence-aware approach adds a **risk dimension** to the scaling decision:
- When the model is confident (narrow prediction interval), it scales efficiently to the median — saving resources
- When the model is uncertain (wide prediction interval), it scales conservatively to the upper bound — preventing SLO violations
- The result: 37% fewer SLO violations with only 10% more pods

This is analogous to how a human operator would behave: "If I'm not sure how much traffic is coming, I'd rather have a few extra pods than risk downtime."

## Relationship to Novelty 2

Novelty 1 (confidence-aware) and Novelty 2 (prediction-stability-aware) are complementary:
- **Novelty 1** answers: "How much should I trust this prediction?" (confidence level)
- **Novelty 2** answers: "Should I act on this prediction yet?" (temporal stability)

Together they form a complete decision framework for prediction-assisted autoscaling.
