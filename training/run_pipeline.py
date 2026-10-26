"""Reusable port of training/train_final_kaggle.ipynb.

Parses raw Common Log Format access logs into per-minute request counts ("QPS"),
trains the PAHPA point model and the P10/P50/P90 quantile models, and replays the
test split through the four-system M/M/c scaling simulation (HPA, PAHPA,
Novelty 1, Novelty 1+2).

The modelling, scaling and metric code below is copied from the notebook cells
unchanged; only the I/O (paths, CLI args, multiple input logs, gzip support) and
the burst-bypass counters were added. Thresholds are deliberately not
configurable here so that every dataset runs with the NASA-tuned settings.

Example:
    .venv/bin/python training/run_pipeline.py \
        --log training/raw/clarknet_access_log_Aug28 training/raw/clarknet_access_log_Sep4 \
        --name clarknet --service-rate 150 --max-replicas 15 --out-dir .

Outputs (relative to --out-dir):
    prediction/models/<name>/qps_point_model.joblib, qps_quantile_models.joblib
    data/<name>/predictions.csv, scaling_simulation.csv, prediction_metrics.csv,
                <name>_qps_history.csv
    training/reports/<name>/training_summary.json + 3 PNG plots
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import os
import re
import time
import warnings
from collections import Counter
from datetime import datetime

# Reproducibility: the slope feature (np.polyfit -> LAPACK) differs in the last
# bit between OpenBLAS CPU kernels, and LightGBM bin edges amplify that into
# visibly different models. Kaggle's CPUs use the Haswell (AVX2) kernels; forcing
# them here (with numpy < 2.4) reproduces the committed NASA models exactly.
# Must be set before numpy is imported.
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ============================================================
# Configuration (notebook cell 3) — not exposed on the CLI on purpose
# ============================================================
WINDOW_SIZE = 60
TEST_SIZE = 2880
SEED = 42
TARGET_TRANSFORM = "log1p"
TRIM_TO_MAX = 36520

QUANTILES = [0.1, 0.5, 0.9]
CONFIDENCE_HIGH = 0.7
CONFIDENCE_LOW = 0.3

SLO = 1.0  # seconds

SCALE_UP_K = 3
SCALE_DOWN_M = 5
SCALE_DOWN_N = 7
BURST_MULTIPLIER = 2.0

# Set from the CLI in main(); notebook values are 30.0 and 15.
SERVICE_RATE = 30.0
MAX_REPLICAS = 15


# ============================================================
# Parse raw HTTP log -> minute-level QPS (notebook cell 1)
# ============================================================
def _open_log(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="latin-1")
    return open(path, "r", encoding="latin-1")


def parse_logs(paths):
    timestamp_pattern = re.compile(r'\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})')
    minute_counts = Counter()
    parse_errors = 0
    total_lines = 0

    for raw_log in paths:
        with _open_log(raw_log) as f:
            for line in f:
                total_lines += 1
                match = timestamp_pattern.search(line)
                if match:
                    try:
                        dt = datetime.strptime(match.group(1), '%d/%b/%Y:%H:%M:%S')
                        minute_key = dt.replace(second=0)
                        minute_counts[minute_key] += 1
                    except ValueError:
                        parse_errors += 1
                else:
                    parse_errors += 1

    print(f"Parsed {total_lines} lines, {parse_errors} errors")
    print(f"Unique minutes: {len(minute_counts)}")

    min_time = min(minute_counts.keys())
    max_time = max(minute_counts.keys())
    all_minutes = pd.date_range(min_time, max_time, freq='min')
    qps_values = [minute_counts.get(t.to_pydatetime(), 0) for t in all_minutes]
    df_qps = pd.DataFrame({'timestamp': all_minutes, 'qps': qps_values})
    print(f"QPS range: [{min(qps_values)}, {max(qps_values)}], mean: {sum(qps_values)/len(qps_values):.1f}")
    stats = {
        "total_lines": total_lines,
        "parse_errors": parse_errors,
        "unique_minutes": len(minute_counts),
        "first_minute": str(min_time),
        "last_minute": str(max_time),
        "raw_minutes": len(df_qps),
        "zero_minutes": int(sum(1 for v in qps_values if v == 0)),
    }
    return df_qps, stats


# ============================================================
# Feature engineering (notebook cell 7)
# ============================================================
def make_window_features(window_values):
    v = np.asarray(window_values, dtype=float).ravel()
    diffs = np.diff(v)
    mean_val = float(np.mean(v))
    std_val = float(np.std(v)) if len(v) > 1 else 0.0
    last = float(v[-1])

    features = list(v)
    features.extend(diffs.tolist())

    for size in (3, 6, 12, 24):
        seg = v[-min(size, len(v)):]
        features.extend([
            float(np.mean(seg)),
            float(np.std(seg)) if len(seg) > 1 else 0.0,
            float(np.min(seg)),
            float(np.max(seg)),
            float(np.median(seg)),
        ])

    slope = float(np.polyfit(np.arange(len(v), dtype=float), v, 1)[0]) if len(v) > 1 else 0.0
    features.extend([
        last, mean_val, std_val,
        float(np.min(v)), float(np.max(v)), float(np.ptp(v)),
        slope,
        float(diffs[-1]) if len(diffs) else 0.0,
        float(np.mean(diffs)) if len(diffs) else 0.0,
        float(np.std(diffs)) if len(diffs) > 1 else 0.0,
        last - mean_val,
        last / max(mean_val, 1e-9),
    ])
    return np.array(features, dtype=float)


def make_supervised(series_values, window):
    n = len(series_values)
    rows = n - window
    if rows <= 0:
        raise ValueError(f"Series too short: {n} <= {window}")
    sample = make_window_features(series_values[:window])
    feat_dim = len(sample)
    X = np.zeros((rows, feat_dim))
    Y = np.zeros(rows)
    for i in range(rows):
        X[i] = make_window_features(series_values[i:i + window])
        Y[i] = series_values[i + window]
    return X, Y


# ============================================================
# Model helpers + M/M/c queueing (notebook cell 9)
# ============================================================
def build_lgbm(seed, objective="regression", alpha=None):
    params = dict(
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=12,
        reg_alpha=0.05,
        reg_lambda=0.15,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        verbosity=-1,
        objective=objective,
    )
    if alpha is not None:
        params["alpha"] = alpha
    return LGBMRegressor(**params)


def transform_y(y):
    if TARGET_TRANSFORM == "log1p":
        return np.log1p(y)
    return y


def inverse_y(y):
    if TARGET_TRANSFORM == "log1p":
        return np.expm1(y)
    return y


def regression_metrics(y_true, y_pred, mape_floor=10.0):
    """Compute metrics. MAPE uses max(y_true, mape_floor) as denominator
    to avoid inflation from near-zero QPS values."""
    mask = y_true > 0
    denom = np.maximum(np.abs(y_true[mask]), mape_floor)
    mape = float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom) * 100)
    return {
        "mape": mape,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def mmc_response_time(arrival_rate, service_rate, servers):
    if servers < 1 or service_rate <= 0:
        return math.inf
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


def required_replicas(qps, service_rate=30.0, slo=1.0, min_r=1, max_r=15):
    for r in range(min_r, max_r + 1):
        if mmc_response_time(qps, service_rate, r) <= slo:
            return r
    return max_r


# ============================================================
# Confidence (notebook cells 17, 19)
# ============================================================
def compute_confidence(p10, p50, p90):
    width = p90 - p10
    return 1.0 - np.clip(width / np.maximum(p50, 1.0), 0, 2) / 2.0


def confidence_aware_qps(p10, p50, p90, high_thresh, low_thresh):
    confidence = compute_confidence(p10, p50, p90)
    target = np.where(
        confidence > high_thresh,
        p50,
        np.where(
            confidence > low_thresh,
            0.5 * p50 + 0.5 * p90,
            p90,
        )
    )
    return target, confidence


# ============================================================
# Novelty 2 stability filter (notebook cell 21)
# ============================================================
class StabilityFilter:
    def __init__(self, scale_up_consecutive=3, scale_down_m=5, scale_down_n=7,
                 burst_multiplier=2.0, service_rate_per_pod=30.0):
        self.scale_up_consecutive = scale_up_consecutive
        self.scale_down_m = scale_down_m
        self.scale_down_n = scale_down_n
        self.burst_multiplier = burst_multiplier
        self.service_rate_per_pod = service_rate_per_pod
        self._history = collections.deque(maxlen=scale_down_n)

    def filter(self, proposed_replicas, current_replicas, decision_qps=None):
        self._history.append(proposed_replicas)

        if proposed_replicas == current_replicas:
            return current_replicas

        direction = "up" if proposed_replicas > current_replicas else "down"

        # Burst bypass
        if (direction == "up" and decision_qps is not None
                and decision_qps > self.burst_multiplier * current_replicas * self.service_rate_per_pod):
            return proposed_replicas

        if direction == "up":
            if len(self._history) < self.scale_up_consecutive:
                return proposed_replicas
            recent = list(self._history)[-self.scale_up_consecutive:]
            if all(r > current_replicas for r in recent):
                return proposed_replicas
            return current_replicas

        # direction == "down"
        if len(self._history) < self.scale_down_n:
            return proposed_replicas
        recent_n = list(self._history)[-self.scale_down_n:]
        down_count = sum(1 for r in recent_n if r < current_replicas)
        if down_count >= self.scale_down_m:
            return proposed_replicas
        return current_replicas


# ============================================================
# Scaling metrics (notebook cell 23)
# ============================================================
def scaling_metrics(replicas, latency, actual_qps, slo=1.0):
    finite_lat = latency[np.isfinite(latency)]
    slo_violations = np.sum(latency > slo) / len(latency) * 100
    p99_latency = np.percentile(finite_lat, 99) if len(finite_lat) > 0 else float('inf')
    avg_pods = np.mean(replicas)
    changes = int(np.sum(np.diff(replicas) != 0))
    flaps = 0
    for j in range(2, len(replicas)):
        if replicas[j] > replicas[j-1] < replicas[j-2]:
            flaps += 1
    min_needed = np.array([required_replicas(q, SERVICE_RATE, slo, max_r=MAX_REPLICAS) for q in actual_qps])
    over_prov = np.mean(np.maximum(replicas - min_needed, 0)) / max(np.mean(min_needed), 1) * 100
    return {
        "slo_violation_pct": round(slo_violations, 2),
        "p99_latency_ms": round(p99_latency * 1000, 1),
        "avg_pods": round(avg_pods, 2),
        "scaling_events": changes,
        "flap_count": flaps,
        "over_prov_pct": round(over_prov, 2),
    }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def main():
    global SERVICE_RATE, MAX_REPLICAS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", nargs="+", required=True,
                    help="raw access log path(s), plain or .gz, given in time order")
    ap.add_argument("--name", required=True, help="dataset name, used for output sub-directories")
    ap.add_argument("--service-rate", type=float, default=30.0, help="requests/minute bucket per pod (default 30)")
    ap.add_argument("--max-replicas", type=int, default=15, help="replica ceiling (default 15)")
    ap.add_argument("--out-dir", default=".", help="root under which prediction/, data/, training/reports/ are written")
    args = ap.parse_args()

    SERVICE_RATE = float(args.service_rate)
    MAX_REPLICAS = int(args.max_replicas)

    model_dir = os.path.join(args.out_dir, "prediction", "models", args.name)
    data_dir = os.path.join(args.out_dir, "data", args.name)
    report_dir = os.path.join(args.out_dir, "training", "reports", args.name)
    for d in (model_dir, data_dir, report_dir):
        os.makedirs(d, exist_ok=True)

    print(f"Window: {WINDOW_SIZE}, Test size: {TEST_SIZE}, Transform: {TARGET_TRANSFORM}")
    print(f"Quantiles: {QUANTILES}, Confidence thresholds: [{CONFIDENCE_LOW}, {CONFIDENCE_HIGH}]")
    print(f"Service rate: {SERVICE_RATE}/pod, SLO: {SLO}s, replicas 1-{MAX_REPLICAS}")

    # ---- Parse ----
    df_qps, parse_stats = parse_logs(args.log)
    history_path = os.path.join(data_dir, f"{args.name}_qps_history.csv")
    df_qps.to_csv(history_path, index=False)
    print(f"Saved {len(df_qps)} rows to {history_path}")

    # ---- Load data (cell 5) ----
    df = pd.read_csv(history_path)
    series_raw = df["qps"].astype(float).to_numpy()
    print(f"Raw dataset: {len(series_raw)} data points")

    TRIM_TO = min(TRIM_TO_MAX, len(series_raw))
    series = series_raw[:TRIM_TO]

    print(f"Using first {len(series)} data points")
    print(f"QPS range: [{series.min():.0f}, {series.max():.0f}], mean: {series.mean():.1f}")
    print(f"Zero-QPS minutes: {np.sum(series == 0)} ({np.sum(series == 0)/len(series)*100:.1f}%)")

    # ---- Dataset (cell 11) ----
    t_start = time.time()
    X, Y = make_supervised(series, WINDOW_SIZE)
    split = len(X) - TEST_SIZE
    split = min(max(split, 1), len(X) - 1)

    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    print(f"Features built in {time.time()-t_start:.1f}s — train {len(X_train)}, test {len(X_test)}, features {X.shape[1]}")

    # ---- PAHPA point model (cell 13) ----
    t_start = time.time()
    model_pahpa = build_lgbm(SEED)
    model_pahpa.fit(X_train, transform_y(Y_train))
    preds_pahpa = np.clip(inverse_y(model_pahpa.predict(X_test)), 0, None)
    metrics_pahpa = regression_metrics(Y_test, preds_pahpa)
    print(f"PAHPA trained in {time.time()-t_start:.1f}s — MAPE {metrics_pahpa['mape']:.2f}%, "
          f"MAE {metrics_pahpa['mae']:.4f}, RMSE {metrics_pahpa['rmse']:.4f}, R² {metrics_pahpa['r2']:.4f}")

    # ---- Quantile models (cell 15) ----
    quantile_models = {}
    quantile_preds = {}
    for q in QUANTILES:
        t0 = time.time()
        model = build_lgbm(SEED, objective="quantile", alpha=q)
        model.fit(X_train, transform_y(Y_train))
        preds = np.clip(inverse_y(model.predict(X_test)), 0, None)
        quantile_models[q] = model
        quantile_preds[q] = preds
        print(f"  P{int(q*100)} trained in {time.time()-t0:.1f}s")

    p10 = quantile_preds[0.1]
    p50 = quantile_preds[0.5]
    p90 = quantile_preds[0.9]

    # ---- Interval analysis (cell 17) ----
    metrics_p50 = regression_metrics(Y_test, p50)
    coverage = np.mean((Y_test >= p10) & (Y_test <= p90)) * 100
    avg_width = np.mean(p90 - p10)
    median_width = np.median(p90 - p10)
    print(f"P50 MAPE {metrics_p50['mape']:.2f}%, R² {metrics_p50['r2']:.4f}; "
          f"coverage {coverage:.1f}%, avg width {avg_width:.2f}, median width {median_width:.2f}")

    # ---- Confidence-aware target (cell 19) ----
    target_qps, confidence = confidence_aware_qps(p10, p50, p90, CONFIDENCE_HIGH, CONFIDENCE_LOW)
    metrics_conf = regression_metrics(Y_test, target_qps)

    # ---- Scaling simulation (cell 23) ----
    n_test = len(Y_test)

    hpa_replicas = np.zeros(n_test, dtype=int)
    pahpa_replicas = np.zeros(n_test, dtype=int)
    conf_replicas = np.zeros(n_test, dtype=int)
    stab_replicas = np.zeros(n_test, dtype=int)

    hpa_latency = np.zeros(n_test)
    pahpa_latency = np.zeros(n_test)
    conf_latency = np.zeros(n_test)
    stab_latency = np.zeros(n_test)

    stability_filter = StabilityFilter(
        scale_up_consecutive=SCALE_UP_K,
        scale_down_m=SCALE_DOWN_M,
        scale_down_n=SCALE_DOWN_N,
        burst_multiplier=BURST_MULTIPLIER,
        service_rate_per_pod=SERVICE_RATE,
    )

    # Burst-bypass instrumentation (observational only, does not change decisions):
    # fired = the bypass branch returned; decisive = the K-consecutive rule would
    # have suppressed that scale-up.
    burst_fired = 0
    burst_decisive = 0

    for i in range(n_test):
        actual = Y_test[i]
        prev_actual = Y_test[i-1] if i > 0 else actual

        # HPA: reactive, uses previous minute's QPS
        hpa_r = required_replicas(prev_actual, SERVICE_RATE, SLO, max_r=MAX_REPLICAS)
        hpa_replicas[i] = hpa_r
        hpa_latency[i] = mmc_response_time(actual, SERVICE_RATE, hpa_r)

        # PAHPA: uses point prediction
        pahpa_r = required_replicas(preds_pahpa[i], SERVICE_RATE, SLO, max_r=MAX_REPLICAS)
        pahpa_replicas[i] = pahpa_r
        pahpa_latency[i] = mmc_response_time(actual, SERVICE_RATE, pahpa_r)

        # Confidence-aware: uses risk-adjusted target
        conf_r = required_replicas(target_qps[i], SERVICE_RATE, SLO, max_r=MAX_REPLICAS)
        conf_replicas[i] = conf_r
        conf_latency[i] = mmc_response_time(actual, SERVICE_RATE, conf_r)

        # Stability-aware: confidence-aware + stability filter (Novelty 2)
        current_stab = stab_replicas[i-1] if i > 0 else 1
        stab_r = stability_filter.filter(
            proposed_replicas=conf_r,
            current_replicas=current_stab,
            decision_qps=target_qps[i],
        )
        stab_replicas[i] = stab_r
        stab_latency[i] = mmc_response_time(actual, SERVICE_RATE, stab_r)

        if conf_r > current_stab and target_qps[i] > BURST_MULTIPLIER * current_stab * SERVICE_RATE:
            burst_fired += 1
            hist = list(stability_filter._history)
            would_allow = (len(hist) < SCALE_UP_K
                           or all(r > current_stab for r in hist[-SCALE_UP_K:]))
            if not would_allow:
                burst_decisive += 1

    metrics_hpa_scale = scaling_metrics(hpa_replicas, hpa_latency, Y_test)
    metrics_pahpa_scale = scaling_metrics(pahpa_replicas, pahpa_latency, Y_test)
    metrics_conf_scale = scaling_metrics(conf_replicas, conf_latency, Y_test)
    metrics_stab_scale = scaling_metrics(stab_replicas, stab_latency, Y_test)

    ceiling_hits = {
        "hpa_baseline": int(np.sum(hpa_replicas == MAX_REPLICAS)),
        "pahpa_point": int(np.sum(pahpa_replicas == MAX_REPLICAS)),
        "confidence_aware": int(np.sum(conf_replicas == MAX_REPLICAS)),
        "stability_aware": int(np.sum(stab_replicas == MAX_REPLICAS)),
    }
    needed_actual = np.array([required_replicas(q, SERVICE_RATE, SLO, max_r=10_000) for q in Y_test])

    scale_df = pd.DataFrame([
        {"System": "HPA Baseline (reactive)", **metrics_hpa_scale},
        {"System": "PAHPA (point prediction)", **metrics_pahpa_scale},
        {"System": "Novelty 1 (confidence-aware)", **metrics_conf_scale},
        {"System": "Novelty 1+2 (conf + stability)", **metrics_stab_scale},
    ])
    print("=" * 110)
    print(f"Scaling Simulation Results (M/M/c queueing, SLO={SLO}s, service rate {SERVICE_RATE}, max {MAX_REPLICAS})")
    print(scale_df.to_string(index=False))
    print(f"Burst bypass fired {burst_fired}x ({burst_decisive}x decisive). Ceiling hits: {ceiling_hits}. "
          f"Max replicas needed for actual load: {int(needed_actual.max())}")
    print("=" * 110)

    # ---- Plots (cells 27-29) ----
    plot_range = slice(0, min(500, n_test))
    x_axis = np.arange(len(Y_test[plot_range]))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    axes[0].plot(x_axis, Y_test[plot_range], label="Actual", linewidth=0.8, color='black')
    axes[0].plot(x_axis, preds_pahpa[plot_range], label="PAHPA Predicted", linewidth=0.8, alpha=0.8, color='#2196F3')
    axes[0].set_title(f"PAHPA Point Prediction (MAPE: {metrics_pahpa['mape']:.2f}%)")
    axes[0].legend()
    axes[0].set_ylabel("QPS")
    axes[1].plot(x_axis, Y_test[plot_range], label="Actual", linewidth=0.8, color='black')
    axes[1].plot(x_axis, p50[plot_range], label="P50 (median)", linewidth=0.8, color='#4CAF50')
    axes[1].fill_between(x_axis, p10[plot_range], p90[plot_range], alpha=0.25, color='#4CAF50', label="P10-P90 interval")
    axes[1].set_title(f"Quantile Prediction with 80% Interval (Coverage: {coverage:.1f}%)")
    axes[1].legend()
    axes[1].set_ylabel("QPS")
    axes[2].plot(x_axis, Y_test[plot_range], label="Actual", linewidth=0.8, color='black')
    axes[2].plot(x_axis, target_qps[plot_range], label="Conf-Aware Target", linewidth=0.8, color='#FF5722')
    axes[2].plot(x_axis, p50[plot_range], label="P50", linewidth=0.5, alpha=0.5, color='#4CAF50', linestyle='--')
    axes[2].plot(x_axis, p90[plot_range], label="P90", linewidth=0.5, alpha=0.5, color='#FFC107', linestyle='--')
    axes[2].set_title(f"Confidence-Aware QPS Target (MAPE: {metrics_conf['mape']:.2f}%)")
    axes[2].legend()
    axes[2].set_ylabel("QPS")
    axes[2].set_xlabel("Test sample index")
    plt.tight_layout()
    plt.savefig(f"{report_dir}/actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(confidence, bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
    axes[0, 0].axvline(CONFIDENCE_HIGH, color='green', linestyle='--', label=f'High thresh ({CONFIDENCE_HIGH})')
    axes[0, 0].axvline(CONFIDENCE_LOW, color='red', linestyle='--', label=f'Low thresh ({CONFIDENCE_LOW})')
    axes[0, 0].set_title('Confidence Score Distribution')
    axes[0, 0].set_xlabel('Confidence')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    interval_width = p90 - p10
    axes[0, 1].scatter(Y_test, interval_width, s=1, alpha=0.3, color='#FF5722')
    axes[0, 1].set_title('Prediction Interval Width vs Actual QPS')
    axes[0, 1].set_xlabel('Actual QPS')
    axes[0, 1].set_ylabel('P90 - P10 (interval width)')
    scale_range = slice(0, min(500, n_test))
    axes[1, 0].plot(hpa_replicas[scale_range], label='HPA', linewidth=0.8, alpha=0.6)
    axes[1, 0].plot(pahpa_replicas[scale_range], label='PAHPA', linewidth=0.8, alpha=0.6)
    axes[1, 0].plot(conf_replicas[scale_range], label='Conf-Aware', linewidth=0.8, alpha=0.7, color='#FF5722')
    axes[1, 0].plot(stab_replicas[scale_range], label='Conf+Stability', linewidth=1.2, alpha=0.9, color='#9C27B0')
    axes[1, 0].set_title('Replica Count Over Time')
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].set_ylabel('Replicas')
    axes[1, 0].legend()
    for label, lat, color in [
        ('HPA', hpa_latency, '#2196F3'),
        ('PAHPA', pahpa_latency, '#4CAF50'),
        ('Conf-Aware', conf_latency, '#FF5722'),
        ('Conf+Stability', stab_latency, '#9C27B0'),
    ]:
        finite = np.sort(lat[np.isfinite(lat)])
        cdf = np.arange(1, len(finite)+1) / len(finite)
        axes[1, 1].plot(finite * 1000, cdf, label=label, linewidth=1.5, color=color)
    axes[1, 1].axvline(SLO * 1000, color='red', linestyle=':', label=f'SLO ({SLO*1000:.0f}ms)')
    axes[1, 1].set_title('Latency CDF')
    axes[1, 1].set_xlabel('Response Time (ms)')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_xlim(0, min(SLO * 3000, 5000))
    axes[1, 1].legend()
    plt.tight_layout()
    plt.savefig(f"{report_dir}/scaling_analysis.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(series, linewidth=0.3, color='#2196F3')
    ax.axvline(len(series) - TEST_SIZE, color='red', linestyle='--', label='Train/Test split')
    ax.set_title(f'{args.name} HTTP QPS — Full Series ({len(series)} minutes)')
    ax.set_xlabel('Minute')
    ax.set_ylabel('QPS')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{report_dir}/qps_overview.png", dpi=150)
    plt.close(fig)

    # ---- Save outputs (cells 31-32) ----
    pd.DataFrame({
        "sample_index": np.arange(n_test),
        "actual": Y_test,
        "pahpa_point": preds_pahpa,
        "quantile_p10": p10,
        "quantile_p50": p50,
        "quantile_p90": p90,
        "confidence": confidence,
        "conf_aware_target": target_qps,
    }).to_csv(f"{data_dir}/predictions.csv", index=False)

    pd.DataFrame({
        "sample_index": np.arange(n_test),
        "actual_qps": Y_test,
        "hpa_replicas": hpa_replicas,
        "pahpa_replicas": pahpa_replicas,
        "conf_replicas": conf_replicas,
        "stab_replicas": stab_replicas,
        "hpa_latency_ms": hpa_latency * 1000,
        "pahpa_latency_ms": pahpa_latency * 1000,
        "conf_latency_ms": conf_latency * 1000,
        "stab_latency_ms": stab_latency * 1000,
    }).to_csv(f"{data_dir}/scaling_simulation.csv", index=False)

    pd.DataFrame([
        {"system": "pahpa_point", **metrics_pahpa},
        {"system": "quantile_p50", **metrics_p50},
        {"system": "conf_aware", **metrics_conf},
    ]).to_csv(f"{data_dir}/prediction_metrics.csv", index=False)

    joblib.dump({
        "model": model_pahpa,
        "window": WINDOW_SIZE,
        "steps": 1,
        "features": "enhanced",
        "target_transform": TARGET_TRANSFORM,
        "metrics": metrics_pahpa,
    }, f"{model_dir}/qps_point_model.joblib")

    joblib.dump({
        "models": {str(q): quantile_models[q] for q in QUANTILES},
        "quantiles": QUANTILES,
        "window": WINDOW_SIZE,
        "steps": 1,
        "features": "enhanced",
        "target_transform": TARGET_TRANSFORM,
        "confidence_high": CONFIDENCE_HIGH,
        "confidence_low": CONFIDENCE_LOW,
        "metrics_p50": metrics_p50,
    }, f"{model_dir}/qps_quantile_models.joblib")

    summary = {
        "architecture": "LightGBM enhanced features + quantile regression + stability filter",
        "dataset": args.name,
        "source_logs": [os.path.basename(p) for p in args.log],
        "parse": parse_stats,
        "dataset_rows": int(len(series)),
        "series_mean_qps": round(float(series.mean()), 3),
        "series_max_qps": float(series.max()),
        "test_mean_qps": round(float(Y_test.mean()), 3),
        "test_max_qps": float(Y_test.max()),
        "window_size": WINDOW_SIZE,
        "target_transform": TARGET_TRANSFORM,
        "test_size": TEST_SIZE,
        "seed": SEED,
        "service_rate": SERVICE_RATE,
        "slo_s": SLO,
        "max_replicas": MAX_REPLICAS,
        "paper_reference": {
            "mape": 24.51, "mae": 7.59, "rmse": 10.53, "r2": 0.81
        },
        "pahpa_metrics": metrics_pahpa,
        "novelty1_p50_metrics": metrics_p50,
        "novelty1_conf_aware_metrics": metrics_conf,
        "prediction_interval": {
            "coverage_pct": round(coverage, 2),
            "avg_width": round(avg_width, 2),
            "median_width": round(median_width, 2),
        },
        "confidence_distribution": {
            "mean": round(float(np.mean(confidence)), 3),
            "high_pct": round(float(np.mean(confidence > CONFIDENCE_HIGH) * 100), 1),
            "medium_pct": round(float(np.mean((confidence > CONFIDENCE_LOW) & (confidence <= CONFIDENCE_HIGH)) * 100), 1),
            "low_pct": round(float(np.mean(confidence <= CONFIDENCE_LOW) * 100), 1),
        },
        "stability_filter": {
            "scale_up_consecutive": SCALE_UP_K,
            "scale_down_m_of_n": f"{SCALE_DOWN_M}/{SCALE_DOWN_N}",
            "burst_multiplier": BURST_MULTIPLIER,
            "burst_bypass_fired": burst_fired,
            "burst_bypass_decisive": burst_decisive,
        },
        "replica_ceiling_hits": ceiling_hits,
        "max_replicas_needed_for_actual": int(needed_actual.max()),
        "scaling_simulation": {
            "hpa_baseline": metrics_hpa_scale,
            "pahpa_point": metrics_pahpa_scale,
            "confidence_aware": metrics_conf_scale,
            "stability_aware": metrics_stab_scale,
        },
    }
    with open(f"{report_dir}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"Wrote outputs to {model_dir}, {data_dir}, {report_dir}")


if __name__ == "__main__":
    main()
