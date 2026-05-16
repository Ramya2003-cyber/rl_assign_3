#!/usr/bin/env python3
"""Generate the LunarLander report plots from saved CSV logs.

The script uses only logged experiment data. Continuous baseline and hover plots
use online training returns because complete offline eval logs are missing for
those runs; discrete SAC and DQN plots use the available 20-episode eval logs.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import pandas as pd
import matplotlib.pyplot as plt


IMAGES = ROOT / "images"

plt.rcParams.update({
    "figure.figsize": (8.2, 4.8),
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#d0d0d0",
    "grid.linewidth": 0.9,
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
})

COLORS = {
    "Continuous SAC": "#4C72B0",
    "Auto alpha": "#4C72B0",
    "Fixed alpha = 0.01": "#DD8452",
    "Discrete SAC": "#4C72B0",
    "DQN": "#DD8452",
}


def load_binned_returns(
    pattern: str,
    label: str,
    value_column: str = "episode_reward",
    step_bin: int = 10_000,
    smooth_window: int | None = None,
) -> pd.DataFrame:
    """Load CSV files and return one value per seed per timestep bin."""
    rows: list[pd.DataFrame] = []
    for seed_idx, path in enumerate(sorted(glob.glob(str(ROOT / pattern)))):
        df = pd.read_csv(path)
        if value_column not in df.columns and "reward" in df.columns:
            df = df.rename(columns={"reward": value_column})
        if value_column not in df.columns and "value" in df.columns:
            df = df.rename(columns={"value": value_column})

        df = df[["step", value_column]].dropna().copy()
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
        df = df.dropna()
        if df.empty:
            continue

        df = df.sort_values("step")
        if smooth_window:
            df[value_column] = df[value_column].rolling(
                smooth_window, min_periods=1
            ).mean()

        df["step_bin"] = (df["step"] // step_bin) * step_bin
        binned = (
            df.groupby("step_bin", as_index=False)[value_column]
            .mean()
            .rename(columns={"step_bin": "step", value_column: "return"})
        )
        binned["seed"] = seed_idx + 1
        binned["variant"] = label
        rows.append(binned)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and 95% CI across seeds for each variant/timestep."""
    stats = (
        data.groupby(["variant", "step"])["return"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["ci"] = 1.96 * stats["std"].fillna(0.0) / stats["count"].pow(0.5)
    stats["low"] = stats["mean"] - stats["ci"]
    stats["high"] = stats["mean"] + stats["ci"]
    return stats


def plot_with_ci(ax: plt.Axes, data: pd.DataFrame) -> pd.DataFrame:
    """Draw one Matplotlib mean line with CI band for each variant."""
    stats = summarize(data)
    for variant, group in stats.groupby("variant", sort=False):
        group = group.sort_values("step")
        color = COLORS.get(variant)
        x = group["step"].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        low = group["low"].to_numpy(dtype=float)
        high = group["high"].to_numpy(dtype=float)
        ax.plot(x, mean, label=variant, linewidth=2.2, color=color)
        ax.fill_between(x, low, high, alpha=0.18, color=color)
    return stats


def set_data_ylim(ax: plt.Axes, stats: pd.DataFrame, include: list[float] | None = None) -> None:
    """Set y-limits from plotted mean/95% CI bands, including reference lines."""
    vals = stats["low"].dropna().tolist()
    vals.extend(stats["high"].dropna().tolist())
    vals.extend(include or [])
    low, high = min(vals), max(vals)
    margin = max(20.0, 0.08 * (high - low))
    ax.set_ylim(low - margin, high + margin)


def save_current(name: str) -> None:
    plt.tight_layout()
    plt.savefig(IMAGES / name, bbox_inches="tight")
    plt.close()
    print(f"saved {IMAGES / name}")


def plot_continuous_baseline() -> None:
    data = load_binned_returns(
        "q2_lunar_lander/q2_2_1_continuous_sac/pytorch_sac/exp_local/"
        "LunarLanderContinuous-v3/baseline_seed_*/train.csv",
        "Continuous SAC",
        step_bin=5_000,
        smooth_window=50,
    )
    _, ax = plt.subplots()
    stats = plot_with_ci(ax, data)
    ax.axhline(200, color="crimson", linestyle="--", linewidth=1.6, label="Solved threshold")
    ax.set_title("Continuous SAC Baseline")
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Smoothed online episode return")
    ax.set_xlim(0, 300_000)
    set_data_ylim(ax, stats, include=[200])
    ax.legend(loc="lower right", frameon=True)
    save_current("q2_2_1_continuous_sac_baseline.png")


def plot_hover_reward_flip() -> None:
    auto = load_binned_returns(
        "q2_lunar_lander/q2_2_3_hover_reward_flip/pytorch_sac/exp_local/"
        "LunarLanderContinuous-v3/hover_auto_seed_*/train.csv",
        "Auto alpha",
        step_bin=10_000,
        smooth_window=50,
    )
    fixed = load_binned_returns(
        "q2_lunar_lander/q2_2_3_hover_reward_flip/pytorch_sac/exp_local/"
        "LunarLanderContinuous-v3/hover_fixed_seed_*/train.csv",
        "Fixed alpha = 0.01",
        step_bin=10_000,
        smooth_window=50,
    )
    data = pd.concat([auto, fixed], ignore_index=True)
    _, ax = plt.subplots()
    stats = plot_with_ci(ax, data)
    ax.axvline(300_000, color="black", linestyle="--", linewidth=1.6, alpha=0.8)
    ax.set_title("Hover Reward Flip")
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Smoothed online episode return")
    ax.set_xlim(0, 600_000)
    set_data_ylim(ax, stats)
    y_top = ax.get_ylim()[1]
    ax.text(310_000, y_top - 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            "reward flip: +200 to -100", fontsize=10.5)
    ax.legend(loc="lower right", frameon=True)
    save_current("q2_2_3_hover_reward_flip.png")


def plot_discrete_sac_eval() -> pd.DataFrame:
    data = load_binned_returns(
        "q2_lunar_lander/q2_2_4_discrete_sac_vs_dqn/discrete_sac/exp_local/"
        "LunarLander-v3/discrete_sac_seed_*/eval.csv",
        "Discrete SAC",
        step_bin=10_000,
    )
    _, ax = plt.subplots()
    stats = plot_with_ci(ax, data)
    ax.axhline(200, color="crimson", linestyle="--", linewidth=1.6, label="Solved threshold")
    ax.set_title("Discrete SAC Evaluation")
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Average return over 20 eval episodes")
    ax.set_xlim(0, 300_000)
    set_data_ylim(ax, stats, include=[200])
    ax.legend(loc="lower right", frameon=True)
    save_current("q2_2_4_discrete_sac_eval.png")
    return data


def plot_discrete_vs_dqn_eval(sac: pd.DataFrame) -> None:
    dqn = load_binned_returns(
        "q2_lunar_lander/q2_2_4_discrete_sac_vs_dqn/dqn/exp_local/"
        "LunarLander-v3/dqn_seed_*/eval.csv",
        "DQN",
        step_bin=10_000,
    )
    data = pd.concat([sac, dqn], ignore_index=True)
    _, ax = plt.subplots()
    stats = plot_with_ci(ax, data)
    ax.axhline(200, color="crimson", linestyle="--", linewidth=1.6, label="Solved threshold")
    ax.set_title("Discrete SAC vs DQN")
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Average return over 20 eval episodes")
    ax.set_xlim(0, 300_000)
    set_data_ylim(ax, stats, include=[200])
    ax.legend(loc="lower right", frameon=True)
    save_current("q2_2_4_discrete_sac_vs_dqn_eval.png")


def main() -> None:
    IMAGES.mkdir(exist_ok=True)
    plot_continuous_baseline()
    plot_hover_reward_flip()
    sac = plot_discrete_sac_eval()
    plot_discrete_vs_dqn_eval(sac)


if __name__ == "__main__":
    main()
