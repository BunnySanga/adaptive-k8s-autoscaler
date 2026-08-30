from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"system", "p99_latency_ms", "pod_count", "scaling_event", "slo_violation"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    grouped = frame.groupby("system", dropna=False)
    summary = grouped.agg(
        p99_latency_ms=("p99_latency_ms", "mean"),
        max_pods=("pod_count", "max"),
        avg_pods=("pod_count", "mean"),
        scaling_events=("scaling_event", "sum"),
        violation_rate_pct=("slo_violation", lambda s: float(s.mean() * 100)),
    )

    if "over_provisioned_pods" in frame.columns:
        summary["over_provisioning_ratio"] = grouped["over_provisioned_pods"].sum() / grouped["pod_count"].sum()

    return summary.reset_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize autoscaling experiment results.")
    parser.add_argument("--input", type=Path, required=True, help="Experiment CSV.")
    parser.add_argument("--output", type=Path, default=Path("evaluation/output/summary.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.input)
    result = summarize(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))

