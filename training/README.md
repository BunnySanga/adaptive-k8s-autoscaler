# Training Guide — Kaggle Notebook

## Overview

All model training runs on **Kaggle** using the notebook `train_final_kaggle.ipynb`. The notebook trains four LightGBM models (1 point prediction + 3 quantile), runs a 4-system scaling simulation, and produces all outputs needed by the controller and evaluation pipeline.

## Dataset

- **Source:** NASA-HTTP July 1995 access log (raw server log file)
- **Kaggle dataset:** Upload the raw `access_log_Jul95` file as a Kaggle dataset
- **Parsing:** The notebook's first code cell parses the raw log into minute-level QPS (queries per second) and saves it as `nasa_qps_history.csv`
- **Size:** 36,520 data points (minutes), trimmed from ~39,693 raw minutes
- **QPS range:** 0 to �200+, mean ~33

## How to Run on Kaggle

### Step 1: Setup

1. Go to [kaggle.com/code](https://kaggle.com/code) and create a new notebook
2. Upload the NASA HTTP raw log as a Kaggle dataset (or attach an existing one)
3. Upload `train_final_kaggle.ipynb` as the notebook source
4. **Accelerator:** None needed — CPU is sufficient (~2 minutes total training time for all 4 models on 36K rows)

### Step 2: Run All Cells

The notebook runs sequentially through these sections:

| Section | Cell(s) | What it does |
|---------|---------|--------------|
| 0. Setup | 0-1 | Install packages, parse raw NASA log to QPS CSV |
| 1. Config | 2-3 | Set hyperparameters, window size, quantiles, SLO |
| 2. Load Data | 4-5 | Load QPS CSV, trim to 36,520 rows |
| 3. Features | 6-7 | `make_window_features()` — 151 enhanced features per 60-minute window |
| 4. Helpers | 8-9 | LightGBM builder, metrics, M/M/c queueing functions |
| 5. Dataset | 10-11 | Build supervised dataset, train/test split (last 2,880 for test) |
| 6. PAHPA Baseline | 12-13 | Train point prediction model, evaluate MAPE/MAE/RMSE/R² |
| 7. Quantile Models | 14-15 | Train P10, P50, P90 quantile LightGBM models |
| 8. Interval Analysis | 16-17 | Coverage, sharpness, confidence distribution |
| 9a. Confidence-Aware | 18-19 | Risk-adjusted QPS target using confidence score |
| 9b. Stability Filter | 20-21 | StabilityFilter class for Novelty 2 |
| 10. Scaling Simulation | 22-23 | 4-system comparison: HPA vs PAHPA vs Conf-Aware vs Stability-Aware |
| 11. Comparison | 24-25 | Print comparison tables |
| 12. Visualizations | 26-29 | Actual vs predicted, confidence/scaling analysis, QPS overview |
| 13. Save Outputs | 30-33 | Save all artifacts to `/kaggle/working/output/` and zip |

### Step 3: Download Outputs

After the notebook finishes, download the zip file from Kaggle's **Output** tab:
- File: `final_output.zip` (in `/kaggle/working/`)

### Step 4: Place Files in Project

Extract the zip and copy files to their project locations:

| Output file | Copy to |
|-------------|---------|
| `qps_point_model.joblib` | `prediction/models/` |
| `qps_quantile_models.joblib` | `prediction/models/` |
| `predictions.csv` | `data/` |
| `scaling_simulation.csv` | `data/` |
| `prediction_metrics.csv` | `data/` |
| `training_summary.json` | `training/reports/` |
| `actual_vs_predicted.png` | `training/reports/` |
| `scaling_analysis.png` | `training/reports/` |
| `qps_overview.png` | `training/reports/` |

## Feature Engineering

Each 60-minute window produces **151 features**:

| Feature group | Count | Description |
|---------------|-------|-------------|
| Raw lags | 60 | All 60 QPS values in the window |
| First differences | 59 | Minute-to-minute changes |
| Multi-scale rolling stats | 20 | Mean, std, min, max, median at windows of 3, 6, 12, 24 minutes |
| Global stats | 12 | Last value, mean, std, min, max, range, slope, last diff, mean diff, std diff, deviation from mean, ratio to mean |

This replaces the LMD (Local Mean Decomposition) signal decomposition from the PAHPA paper. LMD was attempted 3 times but failed due to numerical overflow and accuracy degradation.

## Models Trained

### Point Prediction Model (PAHPA Baseline)

- **Objective:** `regression` (MSE)
- **Target transform:** `log1p` on QPS (stabilizes skewed distribution), `expm1` to invert
- **Hyperparameters:** n_estimators=500, learning_rate=0.035, num_leaves=63, min_child_samples=12, reg_alpha=0.05, reg_lambda=0.15, subsample=0.9, colsample_bytree=0.9
- **Output:** Single QPS forecast for T+1
- **Saved as:** `qps_point_model.joblib` — dict with keys: `model`, `window`, `steps`, `features`, `target_transform`, `metrics`

### Quantile Models (Novelty 1)

Three separate LightGBM models with `objective='quantile'`:

| Model | Alpha | Role |
|-------|-------|------|
| P10 | 0.1 | Optimistic lower bound |
| P50 | 0.5 | Median prediction |
| P90 | 0.9 | Conservative upper bound |

Same hyperparameters and target transform as the point model. All three are saved together.

- **Saved as:** `qps_quantile_models.joblib` — dict with keys: `models` (dict mapping `"0.1"`, `"0.5"`, `"0.9"` to model objects), `quantiles`, `window`, `steps`, `features`, `target_transform`, `confidence_high`, `confidence_low`, `metrics_p50`

## Model Artifact Format

The prediction service (`prediction/serve.py`) loads the quantile models joblib and expects this structure:

```python
artifact = {
    "models": {
        "0.1": <LGBMRegressor>,   # P10 model
        "0.5": <LGBMRegressor>,   # P50 model
        "0.9": <LGBMRegressor>,   # P90 model
    },
    "quantiles": [0.1, 0.5, 0.9],
    "window": 60,
    "steps": 1,
    "features": "enhanced",
    "target_transform": "log1p",
    "confidence_high": 0.7,
    "confidence_low": 0.3,
    "metrics_p50": { ... },
}
```

The service iterates over `artifact["models"]`, calls `model.predict(features)`, applies `expm1` if `target_transform == "log1p"`, computes the confidence score, and returns `[p10, p50, p90]` as the forecast.

## MAPE Calculation Note

The `regression_metrics()` function uses a **floor threshold** of 10.0 for MAPE:

```python
denom = np.maximum(np.abs(y_true), 10.0)
mape = mean(|y_true - y_pred| / denom) * 100
```

This prevents MAPE inflation from near-zero QPS values. Without the floor, periods with 1-5 QPS would produce 200-500% individual errors, inflating the aggregate MAPE from ~32% to ~43% despite identical MAE and R².

## Scaling Simulation

The notebook simulates four autoscaling strategies on the test set (2,880 minutes):

1. **HPA Baseline** — reactive, uses previous minute's actual QPS
2. **PAHPA** — uses point prediction QPS
3. **Novelty 1 (Confidence-Aware)** — uses risk-adjusted QPS based on confidence from quantile intervals
4. **Novelty 1+2 (Full System)** — confidence-aware + stability filter (K=3 consecutive scale-up, 5/7 scale-down, 2x burst bypass)

Each strategy computes `required_replicas()` using M/M/c queueing theory, then the simulation records the resulting latency given the actual QPS and chosen replica count.

## Output Files Reference

### `training_summary.json`

Complete results in one file — prediction metrics, interval analysis, confidence distribution, stability filter config, and all four scaling simulation results.

### `scaling_simulation.csv`

Per-sample (2,880 rows) with columns: `sample_index`, `actual_qps`, `hpa_replicas`, `pahpa_replicas`, `conf_replicas`, `stab_replicas`, `hpa_latency_ms`, `pahpa_latency_ms`, `conf_latency_ms`, `stab_latency_ms`.

### `predictions.csv`

Per-sample with columns: `sample_index`, `actual`, `pahpa_point`, `quantile_p10`, `quantile_p50`, `quantile_p90`, `confidence`, `conf_aware_target`.

## Running Locally / On Other Datasets (`run_pipeline.py`)

`training/run_pipeline.py` is a line-for-line port of the notebook as a CLI script. Use it to re-run the pipeline on any Common Log Format trace without Kaggle:

```bash
python3 -m venv .venv && .venv/bin/pip install -r training/requirements-pipeline.txt
.venv/bin/python training/run_pipeline.py --log <log> [<log> ...] --name <dataset> \
    --service-rate 30 --max-replicas 15 --out-dir .
```

It writes `prediction/models/<name>/`, `data/<name>/` and `training/reports/<name>/` under `--out-dir`. On the NASA log it reproduces the committed outputs exactly, provided numpy < 2.4. The script also forces OpenBLAS Haswell kernels; see `results_clarknet.md` §8 for why. The ClarkNet-HTTP generalisation results are in `results_clarknet.md`.
