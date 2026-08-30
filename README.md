# Machine Learning-Based Adaptive Resource Scaling

This project implements a Kubernetes autoscaling experiment inspired by PAHPA:

- baseline Kubernetes HPA
- PAHPA-style single-step predictive scaling with real-time correction
- proposed Novelty 1 adaptive multi-step predictive scaling

The current implementation is intentionally modular. You can run the app, train a predictor, test the autoscaling logic in dry-run mode, and later connect it to a live Kubernetes cluster.

## Project Layout

```text
app/          FastAPI workload service with Prometheus metrics
controller/   QoS queueing model and autoscaler controller
prediction/   LightGBM-style multi-step training and prediction API
locust/       Six workload patterns for experiments
k8s/          Kubernetes manifests for app, HPA, Prometheus, and prediction service
monitoring/   Prometheus scrape configuration
evaluation/   Metric aggregation and plotting helpers
data/         Input traces and experiment outputs
tests/        Unit tests for core decision logic
```

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The App Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `GET /healthz`
- `GET /work?duration_ms=80`
- `GET /metrics`

## Local Training

The current training path uses the official NASA-HTTP July gzip file stored at:

```text
training/raw/NASA_access_log_Jul95.gz
```

Run:

```bash
source .venv/bin/activate
PYTHONPATH=. python training/train_novelty1_local.py
```

This trains:

- PAHPA-style `T+1` baseline
- Novelty 1 adaptive multi-step predictor, `T+1...T+5`

Because the NASA-HTTP data is aggregated into one-minute buckets, `T+1` means the next minute. It does not mean 24 hours.

## Run Prediction Service

```bash
uvicorn prediction.serve:app --host 0.0.0.0 --port 5000
```

Prediction request example:

```bash
curl -X POST http://localhost:5000/predict \
  -H 'content-type: application/json' \
  -d '{"history":[21,24,26,29,33,35]}'
```

## Run Autoscaler In Dry-Run Mode

Dry-run mode computes desired replicas but does not patch Kubernetes:

```bash
python3 controller/autoscaler.py --mode novelty1 --dry-run --history 20,24,27,31,36,42
```

Modes:

- `hpa_baseline`
- `pahpa_style`
- `novelty1`

`novelty1` applies adaptive multi-step forecasting. It first respects `T+1` like PAHPA, then uses farther horizons only when `T+1` confirms a rising workload.

## Kubernetes Flow

Start a local cluster:

```bash
minikube start --cpus=4 --memory=8192
```

Build and load the shared project image:

```bash
docker build -t adaptive-scaler:latest .
minikube image load adaptive-scaler:latest
```

Apply manifests:

```bash
kubectl apply -f monitoring/
kubectl apply -f k8s/
```

Port-forward services:

```bash
kubectl port-forward svc/adaptive-scaler-app 8000:80
kubectl port-forward svc/prediction-service 5000:5000
kubectl port-forward svc/prometheus 9090:9090
```

## Experiments

Run Locust against the app:

```bash
locust -f locust/locustfile.py --host=http://localhost:8000
```

Then aggregate experiment logs:

```bash
python3 evaluation/analyze.py --input data/experiment_results.csv
```

## Implementation Notes

The base paper uses SLMD-LightGBM. The current implementation uses a stronger local LightGBM feature pipeline on the same NASA-HTTP preprocessing path. This keeps the PAHPA baseline and Novelty 1 comparison fair: same data, same split, same model family, different forecast-decision module.
