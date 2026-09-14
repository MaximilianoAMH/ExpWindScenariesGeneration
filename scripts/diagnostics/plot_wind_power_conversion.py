"""Visual diagnostics for the hourly wind-to-power preprocessing step."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from wind_scenarios.data import PROJECT_ROOT
from wind_scenarios.power import DEFAULT_ALTO_BAGUALES_CURVE, load_power_curve


DEFAULT_HOURLY_POWER = PROJECT_ROOT / "data" / "power_ab_hourly.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hourly_data", type=Path, default=DEFAULT_HOURLY_POWER)
    parser.add_argument("--power_curve", type=Path, default=DEFAULT_ALTO_BAGUALES_CURVE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--days", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.hourly_data, parse_dates=["TIMESTAMP"])
    required = {"TIMESTAMP", "WIND_MS", "POWER_MW"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in hourly audit file: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = frame["TIMESTAMP"].min()
    end = start + pd.Timedelta(days=args.days)
    sample = frame.loc[(frame["TIMESTAMP"] >= start) & (frame["TIMESTAMP"] < end)]

    fig, ax_wind = plt.subplots(figsize=(15, 7))
    ax_power = ax_wind.twinx()
    wind_line = ax_wind.plot(
        sample["TIMESTAMP"],
        sample["WIND_MS"],
        linewidth=1.8,
        label="Hourly wind speed",
    )[0]
    power_line = ax_power.plot(
        sample["TIMESTAMP"],
        sample["POWER_MW"],
        linewidth=1.8,
        linestyle="--",
        label="Hourly calculated power",
    )[0]
    ax_wind.set_xlabel("Time")
    ax_wind.set_ylabel("Wind speed [m/s]")
    ax_power.set_ylabel("Power [MW]")
    ax_wind.grid(alpha=0.25)
    ax_wind.legend(handles=[wind_line, power_line], loc="lower left")
    fig.autofmt_xdate()
    timeseries_path = args.output_dir / "hourly_wind_vs_power.png"
    fig.savefig(timeseries_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    curve = load_power_curve(args.power_curve)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(curve.wind_speed_ms, curve.power_kw / 1000.0, linewidth=2.5, label="Power curve")
    ax.scatter(
        frame["WIND_MS"],
        frame["POWER_MW"],
        s=8,
        alpha=0.12,
        label="Hourly transformed observations",
    )
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Power [MW]")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    curve_path = args.output_dir / "hourly_points_on_power_curve.png"
    fig.savefig(curve_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", timeseries_path)
    print("Saved:", curve_path)


if __name__ == "__main__":
    main()
