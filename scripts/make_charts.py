"""Create the two figures embedded in the project README."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "outputs" / "free_factor_pilot"
IMAGES = ROOT / "docs" / "images"


def style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.22, linewidth=0.8)


def plot_rolling_ic() -> None:
    data = pd.read_csv(RESULTS / "IC序列.csv", parse_dates=["date"])
    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.axhline(0, color="#6b7280", linewidth=1)
    axis.plot(
        data["date"],
        data["alpha_neutral_z_ic_5d"].rolling(60, min_periods=30).mean(),
        label="Neutralized 5-day IC (60-day mean)",
        color="#2563eb",
        linewidth=1.8,
    )
    axis.plot(
        data["date"],
        data["alpha_neutral_z_ic_20d"].rolling(60, min_periods=30).mean(),
        label="Neutralized 20-day IC (60-day mean)",
        color="#f97316",
        linewidth=1.8,
    )
    axis.set_title("Rolling Rank IC after risk neutralization")
    axis.set_ylabel("Rank IC")
    axis.legend(frameon=False, ncol=2, fontsize=9)
    style_axis(axis)
    fig.tight_layout()
    fig.savefig(IMAGES / "rolling_ic.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_quantile_equity() -> None:
    data = pd.read_csv(RESULTS / "五组回测.csv", parse_dates=["date"])
    fig, axis = plt.subplots(figsize=(10, 5.2))
    colors = ["#9ca3af", "#60a5fa", "#34d399", "#fbbf24", "#ef4444"]
    for group, color in zip(range(1, 6), colors):
        axis.plot(
            data["date"], data[f"q{group}_equity"],
            label=f"Q{group}", color=color, linewidth=1.35,
        )
    axis.plot(
        data["date"], data["long_short_equity"],
        label="Q5 - Q1", color="#111827", linewidth=2.2,
    )
    axis.axhline(1, color="#6b7280", linewidth=0.8, linestyle="--")
    axis.set_title("Five-quantile backtest after transaction costs")
    axis.set_ylabel("Net value")
    axis.legend(frameon=False, ncol=6, fontsize=9, loc="upper center")
    style_axis(axis)
    fig.tight_layout()
    fig.savefig(IMAGES / "quantile_equity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    plot_rolling_ic()
    plot_quantile_equity()
    print(f"charts written to {IMAGES}")


if __name__ == "__main__":
    main()
