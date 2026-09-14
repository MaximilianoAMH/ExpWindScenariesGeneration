"""Plot the first five and last five years of daily generated energy."""

from __future__ import annotations

import matplotlib.pyplot as plt

from wind_scenarios.data import (
    DEFAULT_DATA_PATH,
    DEFAULT_TARGET_COLUMN,
    PROJECT_ROOT,
    load_daily_generation_dataframe,
)


OUTPUT_PATH = PROJECT_ROOT / "outputs" / "diagnostics" / "daily_generation_first_last_5_years.png"


def main() -> None:
    df = load_daily_generation_dataframe(DEFAULT_DATA_PATH).copy()
    df["YEAR"] = df["DATE"].dt.year
    df["DAY_OF_YEAR"] = df["DATE"].dt.dayofyear

    years = sorted(df["YEAR"].unique())
    selected_years = years[:5] + years[-5:]
    y_min = 0.0
    y_max = float(df[DEFAULT_TARGET_COLUMN].max()) * 1.05

    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(16, 16),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()

    for ax, year in zip(axes, selected_years):
        part = df[df["YEAR"] == year]
        ax.plot(part["DAY_OF_YEAR"], part[DEFAULT_TARGET_COLUMN], linewidth=1.0)
        ax.set_title(str(year))
        ax.set_xlim(1, 366)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Daily generated energy [MWh]")

    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    for ax in axes:
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels)
    for ax in axes[-2:]:
        ax.set_xlabel("Day of year")

    fig.suptitle("Daily generated energy — first 5 and last 5 years", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
