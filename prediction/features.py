from __future__ import annotations

import numpy as np


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if len(values) > 1 else 0.0


def make_window_features(window_values: np.ndarray) -> np.ndarray:
    """Build robust lag/statistical features from one recent QPS window."""
    values = np.asarray(window_values, dtype=float).reshape(-1)
    if len(values) < 1:
        raise ValueError("window_values must not be empty")

    diffs = np.diff(values)
    x_axis = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x_axis, values, 1)[0]) if len(values) > 1 else 0.0
    mean = float(np.mean(values))
    last = float(values[-1])

    features: list[float] = []
    features.extend(values.tolist())
    features.extend(diffs.tolist())

    for size in (3, 6, 12, 24):
        segment = values[-min(size, len(values)) :]
        features.extend(
            [
                float(np.mean(segment)),
                _safe_std(segment),
                float(np.min(segment)),
                float(np.max(segment)),
                float(np.median(segment)),
            ]
        )

    features.extend(
        [
            last,
            mean,
            _safe_std(values),
            float(np.min(values)),
            float(np.max(values)),
            float(np.ptp(values)),
            slope,
            float(diffs[-1]) if len(diffs) else 0.0,
            float(np.mean(diffs)) if len(diffs) else 0.0,
            _safe_std(diffs) if len(diffs) else 0.0,
            last - mean,
            last / max(mean, 1e-9),
        ]
    )

    return np.asarray(features, dtype=float)


def make_prediction_features(history: np.ndarray, window: int, feature_mode: str) -> np.ndarray:
    values = np.asarray(history, dtype=float).reshape(-1)
    if len(values) < window:
        raise ValueError(f"history must contain at least {window} values")
    window_values = values[-window:]
    if feature_mode == "raw":
        return window_values.reshape(1, -1)
    if feature_mode == "enhanced":
        return make_window_features(window_values).reshape(1, -1)
    raise ValueError(f"unsupported feature mode: {feature_mode}")


def make_supervised(series: np.ndarray, window: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 1D time series into multi-output sliding-window samples."""
    values = np.asarray(series, dtype=float).reshape(-1)
    if window < 1:
        raise ValueError("window must be >= 1")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if len(values) < window + steps:
        raise ValueError("series is too short for the requested window and steps")

    rows = len(values) - window - steps + 1
    x = np.zeros((rows, window), dtype=float)
    y = np.zeros((rows, steps), dtype=float)
    for idx in range(rows):
        x[idx] = values[idx : idx + window]
        y[idx] = values[idx + window : idx + window + steps]
    return x, y


def make_supervised_enhanced(series: np.ndarray, window: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 1D series into supervised samples using engineered features."""
    values = np.asarray(series, dtype=float).reshape(-1)
    if window < 1:
        raise ValueError("window must be >= 1")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if len(values) < window + steps:
        raise ValueError("series is too short for the requested window and steps")

    rows = len(values) - window - steps + 1
    x_rows = [make_window_features(values[idx : idx + window]) for idx in range(rows)]
    y = np.zeros((rows, steps), dtype=float)
    for idx in range(rows):
        y[idx] = values[idx + window : idx + window + steps]
    return np.vstack(x_rows), y


def make_supervised_dataset(
    series: np.ndarray,
    window: int,
    steps: int,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if feature_mode == "raw":
        return make_supervised(series, window, steps)
    if feature_mode == "enhanced":
        return make_supervised_enhanced(series, window, steps)
    raise ValueError(f"unsupported feature mode: {feature_mode}")


def generate_sample_qps(points: int = 1440, seed: int = 7) -> np.ndarray:
    """Generate a repeatable synthetic QPS trace with trend, bursts, and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(points)
    daily = 18 + 8 * np.sin(2 * np.pi * t / max(points // 2, 1))
    trend = np.linspace(0, 10, points)
    noise = rng.normal(0, 2.5, points)
    bursts = np.zeros(points)

    for center, height, width in (
        (int(points * 0.18), 35, 18),
        (int(points * 0.42), 22, 35),
        (int(points * 0.73), 50, 14),
    ):
        bursts += height * np.exp(-((t - center) ** 2) / (2 * width**2))

    return np.clip(daily + trend + bursts + noise, 1, None)
