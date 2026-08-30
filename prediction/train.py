from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from prediction.features import generate_sample_qps, make_supervised_dataset


DEFAULT_MODEL_PATH = Path("prediction/models/qps_multi_step.joblib")


def build_model(random_state: int) -> Any:
    try:
        from lightgbm import LGBMRegressor

        base = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.035,
            num_leaves=63,
            min_child_samples=12,
            reg_alpha=0.05,
            reg_lambda=0.15,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            verbosity=-1,
        )
    except Exception:
        base = HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            random_state=random_state,
        )
    return MultiOutputRegressor(base)


def load_series(args: argparse.Namespace) -> np.ndarray:
    if args.generate_sample:
        return generate_sample_qps(points=args.sample_points, seed=args.seed)

    if not args.data:
        raise ValueError("provide --data or use --generate-sample")

    frame = pd.read_csv(args.data)
    if args.target_col not in frame.columns:
        raise ValueError(f"target column {args.target_col!r} not found in {args.data}")
    return frame[args.target_col].astype(float).to_numpy()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def transform_target(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return values
    if transform == "log1p":
        return np.log1p(values)
    raise ValueError(f"unsupported target transform: {transform}")


def inverse_transform_target(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return values
    if transform == "log1p":
        return np.expm1(values)
    raise ValueError(f"unsupported target transform: {transform}")


def train(args: argparse.Namespace) -> dict[str, Any]:
    series = load_series(args)
    x, y = make_supervised_dataset(series, window=args.window, steps=args.steps, feature_mode=args.feature_mode)

    split = len(x) - args.test_size if args.test_size else int(len(x) * args.train_ratio)
    split = min(max(split, 1), len(x) - 1)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model(random_state=args.seed)
    model.fit(x_train, transform_target(y_train, args.target_transform))
    predictions = np.clip(inverse_transform_target(model.predict(x_test), args.target_transform), 0, None)
    metrics = regression_metrics(y_test, predictions)

    artifact = {
        "model": model,
        "window": args.window,
        "steps": args.steps,
        "feature_mode": args.feature_mode,
        "target_transform": args.target_transform,
        "target_col": args.target_col,
        "metrics": metrics,
        "split_index": split,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    if args.predictions_output:
        frame = pd.DataFrame({"sample_index": np.arange(split, len(x))})
        for step in range(args.steps):
            frame[f"actual_t{step + 1}"] = y_test[:, step]
            frame[f"predicted_t{step + 1}"] = predictions[:, step]
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.predictions_output, index=False)
    if args.metrics_output:
        rows = []
        for step in range(args.steps):
            step_metrics = regression_metrics(y_test[:, [step]], predictions[:, [step]])
            rows.append({"horizon": f"T+{step + 1}", **step_metrics})
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.metrics_output, index=False)
    return {"output": str(args.output), "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-step QPS predictor.")
    parser.add_argument("--data", type=Path, help="CSV file containing QPS history.")
    parser.add_argument("--target-col", default="qps", help="Target column in the CSV.")
    parser.add_argument("--window", type=int, default=6, help="Sliding input window size.")
    parser.add_argument("--steps", type=int, default=5, help="Forecast horizon.")
    parser.add_argument("--feature-mode", choices=["raw", "enhanced"], default="enhanced")
    parser.add_argument("--target-transform", choices=["none", "log1p"], default="log1p")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/test split ratio.")
    parser.add_argument("--test-size", type=int, help="Use the last N supervised samples as test data.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="Model artifact path.")
    parser.add_argument("--predictions-output", type=Path, help="Optional prediction CSV output.")
    parser.add_argument("--metrics-output", type=Path, help="Optional per-horizon metrics CSV output.")
    parser.add_argument("--generate-sample", action="store_true", help="Train on generated sample data.")
    parser.add_argument("--sample-points", type=int, default=1440, help="Generated sample length.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    result = train(parse_args())
    print(result)
