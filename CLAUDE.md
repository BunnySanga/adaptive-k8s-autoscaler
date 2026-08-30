# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Machine Learning-Based Adaptive Resource Scaling** — a Kubernetes-focused cloud computing course project that extends the PAHPA (Prediction-Assisted HPA) paper with two novelties for smarter autoscaling.

**Base paper:** Xiao et al., "PAHPA: Revolutionizing Kubernetes Autoscaling With Integrated Predictive Analytics and Real-Time Monitoring," IEEE TSC, vol. 19, no. 3, May/June 2026.

PAHPA predicts T+1 workload (QPS) using SLMD-LightGBM, combines predicted + current QPS via M/M/c queueing theory, and scales pods proactively through KEDA/HPA. Our system extends this baseline.

## Two Novelties

### Novelty 1: Confidence-Aware Scaling via Quantile Regression
- Train LightGBM with quantile regression (P10, P50, P90) instead of point prediction
- Prediction interval width measures confidence: narrow = certain, wide = uncertain
- Scaling policy: high confidence → scale to median (efficient); low confidence → scale to upper bound (safe)
- Addresses the key autoscaling problem: when to trust predictions vs when to be conservative

### Novelty 2: Prediction-Stability-Aware Scaling
- Prevents scaling flap caused by noisy single-point predictions
- Rule: only execute scale-up if predicted workload exceeds threshold for K consecutive decision intervals (e.g., 3 out of 3)
- Scale-down uses a longer window (e.g., 5 out of 7) to be conservative
- Bypass: skip stability check when predicted load exceeds a hard threshold (>2x current) to avoid missing genuine bursts
- Complements Novelty 1: confidence says "how much to trust this prediction", stability says "should I act on it yet"

## Architecture

```
Locust (load gen) → K8s Service → App Pods → Prometheus (metrics)
                                                    ↓
                                          Prediction Service
                                          (LightGBM quantile: P10, P50, P90)
                                                    ↓
                                          Confidence Score
                                          (interval width → confidence)
                                                    ↓
                                          Risk-Adjusted QoS Decision
                                          (M/M/c queueing theory)
                                                    ↓
                                          Stability Check
                                          (consecutive-interval filter)
                                                    ↓
                                          KEDA/HPA → Scale Pods
```

## Three-System Comparison

All experiments run the same app and workload traces across three configurations:
1. **Baseline HPA** — reactive, stock Kubernetes HPA
2. **PAHPA-style** — single-step T+1 prediction + QoS-aware scaling
3. **Our system** — confidence-aware quantile prediction + stability-aware scaling

## Technology Stack

- **Orchestration:** Kubernetes (Minikube or Kind for local, GKE/EKS optional)
- **App:** Python Flask/FastAPI with simulated CPU-bound processing
- **Monitoring:** Prometheus + Grafana
- **Load testing:** Locust
- **ML:** Python + LightGBM (scikit-learn for preprocessing)
- **Scaling:** KEDA or custom controller + native HPA

## Workload Patterns for Testing

Six patterns exercised against all three systems:
1. Stable — validates no unnecessary scaling
2. Gradual increase — confidence-aware should scale earlier
3. Sudden burst — stability bypass must not delay
4. Repeated short spikes — stability should prevent flapping
5. Sustained growth — both novelties work together
6. Spike-then-drop — stability prevents unnecessary scale-up

## Evaluation Metrics

- P99 latency (ms)
- QoS/SLO violation rate (%)
- Average pod count (resource efficiency)
- Number of scaling events (fewer = more stable)
- Scaling flap count (up→down→up within short window)
- Over-provisioning ratio
- Prediction accuracy (MAPE for point vs quantile)
- Prediction interval coverage and sharpness
- Confidence score distribution

## Implementation Phases

1. **Baseline setup** (Weeks 1–3): K8s cluster + app + Prometheus + Locust + vanilla HPA
2. **PAHPA-style baseline** (Weeks 4–6): LightGBM T+1 prediction + queueing theory + KEDA integration
3. **Our novelties** (Weeks 7–10): Quantile regression + confidence-aware scaling + stability filter
4. **Evaluation & report** (Weeks 11–13): Three-way comparison across all workload patterns

## Key Design Decisions

- ML is a supporting component for forecasting, not the project's focus. The core contribution is in the autoscaling decision logic.
- LMD (Local Mean Decomposition) from PAHPA can be simplified or skipped if too complex; document any deviation.
- Quantile regression (P10/P50/P90) replaces multi-step T+1..T+5 which showed near-random trend accuracy (29%).
- Pod-level load redistribution was explicitly excluded — it requires service-mesh complexity that belongs at a different layer.
- No public dataset required; Locust generates controlled synthetic workloads. Real traces (Google Cluster, Alibaba) can supplement if time permits.

## Build & Run Commands

```bash
# Start local Kubernetes cluster
minikube start --cpus=4 --memory=8192

# Deploy the application
kubectl apply -f k8s/

# Port-forward Prometheus
kubectl port-forward svc/prometheus 9090:9090

# Port-forward Grafana
kubectl port-forward svc/grafana 3000:3000

# Run Locust load tests
locust -f locust/locustfile.py --host=http://$(minikube ip):30080

# Train prediction model (run on Kaggle — see training/train_final_kaggle.ipynb)

# Run prediction service
python3 prediction/serve.py --port 5000

# Run autoscaler controller
python3 controller/autoscaler.py --mode confidence_aware --stability
# Modes: hpa_baseline | pahpa_style | confidence_aware
# Add --stability to enable prediction-stability filter (Novelty 2)
# Stability options: --scale-up-consecutive 3 --scale-down-m 5 --scale-down-n 7 --burst-multiplier 2.0
```

## Project File Structure (planned)

```
k8s/                  # Kubernetes manifests (deployment, service, HPA, KEDA)
prediction/           # LightGBM training and serving
  train.py            # Model training (point + quantile)
  serve.py            # Prediction API server
controller/           # Autoscaler decision logic
  autoscaler.py       # Main controller with three modes + stability flag
  qos.py              # M/M/c queueing theory calculations
  stability.py        # Stability-aware scaling filter
locust/               # Load test scripts
  locustfile.py       # Six workload patterns
monitoring/           # Prometheus/Grafana configs
data/                 # Collected QPS data and experiment results
evaluation/           # Analysis scripts, comparison tables, plots
```
