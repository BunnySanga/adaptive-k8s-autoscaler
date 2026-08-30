from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_timeseries(input_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(input_path)
    required = {"timestamp", "system", "qps", "pod_count", "p99_latency_ms"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for system, part in frame.groupby("system"):
        axes[0].plot(part["timestamp"], part["qps"], label=system)
        axes[1].plot(part["timestamp"], part["pod_count"], label=system)
        axes[2].plot(part["timestamp"], part["p99_latency_ms"], label=system)

    axes[0].set_ylabel("QPS")
    axes[1].set_ylabel("Pods")
    axes[2].set_ylabel("P99 latency ms")
    axes[2].set_xlabel("Time")

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot autoscaling experiment results.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation/output/timeseries.png"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_timeseries(args.input, args.output)

