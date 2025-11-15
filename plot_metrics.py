#!/usr/bin/env python3
"""
Utility to visualize PPO training metrics logged by train.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def _maybe_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _filter_series(xs: List[int], ys: List[float | None]) -> Tuple[List[int], List[float]]:
    filtered_x, filtered_y = [], []
    for x, y in zip(xs, ys):
        if y is None:
            continue
        filtered_x.append(x)
        filtered_y.append(y)
    return filtered_x, filtered_y


def load_metrics(path: Path):
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        steps: List[int] = []
        reward_mean: List[float] = []
        reward_std: List[float] = []
        policy_loss: List[float] = []
        value_loss: List[float] = []
        entropy: List[float] = []
        approx_kl: List[float] = []
        episode_return_mean: List[float | None] = []
        episode_length_mean: List[float | None] = []
        for row in reader:
            if not row:
                continue
            steps.append(int(row["total_steps"]))
            reward_mean.append(float(row["reward_mean"]))
            reward_std.append(float(row["reward_std"]))
            policy_loss.append(float(row["policy_loss"]))
            value_loss.append(float(row["value_loss"]))
            entropy.append(float(row["entropy"]))
            approx_kl.append(float(row["approx_kl"]))
            episode_return_mean.append(_maybe_float(row.get("episode_return_mean", "")))
            episode_length_mean.append(_maybe_float(row.get("episode_length_mean", "")))
    if not steps:
        raise ValueError(f"No metric rows found in {path}")
    return {
        "steps": steps,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "episode_return_mean": episode_return_mean,
        "episode_length_mean": episode_length_mean,
    }


def plot(metrics: dict, output: Path, show: bool) -> None:
    steps = metrics["steps"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(steps, metrics["reward_mean"], label="Rollout reward mean", color="#1f77b4")
    upper = [m + s for m, s in zip(metrics["reward_mean"], metrics["reward_std"])]
    lower = [m - s for m, s in zip(metrics["reward_mean"], metrics["reward_std"])]
    axes[0].fill_between(steps, lower, upper, color="#1f77b4", alpha=0.15, label="±1 std")
    er_x, er_y = _filter_series(steps, metrics["episode_return_mean"])
    if er_x:
        axes[0].plot(er_x, er_y, label="Episode return mean", color="#ff7f0e")
    axes[0].set_ylabel("Reward")
    axes[0].legend(loc="best")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(steps, metrics["policy_loss"], label="Policy loss", color="#d62728")
    axes[1].plot(steps, metrics["value_loss"], label="Value loss", color="#2ca02c")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="best")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    axes[2].plot(steps, metrics["entropy"], label="Entropy", color="#9467bd")
    axes[2].plot(steps, metrics["approx_kl"], label="Approx KL", color="#8c564b")
    el_x, el_y = _filter_series(steps, metrics["episode_length_mean"])
    if el_x:
        axes[2].plot(el_x, el_y, label="Episode length mean", color="#17becf")
    axes[2].set_ylabel("Aux metrics")
    axes[2].set_xlabel("Environment steps")
    axes[2].legend(loc="best")
    axes[2].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Chess PPO Training Metrics")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"[plot] saved figure to {output.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot PPO training metrics.")
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("metrics.csv"),
        help="CSV file produced by train.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics.png"),
        help="Where to store the resulting plot (PNG).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the file.",
    )
    args = parser.parse_args()

    if not args.metrics_path.exists():
        raise FileNotFoundError(
            f"No metrics file found at {args.metrics_path}. "
            "Run train.py first or point to an existing CSV."
        )
    metrics = load_metrics(args.metrics_path)
    plot(metrics, args.output, args.show)


if __name__ == "__main__":
    main()

