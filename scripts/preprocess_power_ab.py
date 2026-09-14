from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wind_scenarios.power import (
    DEFAULT_ALTO_BAGUALES_CURVE,
    DEFAULT_HUB_HEIGHT_M,
    DEFAULT_N_TURBINES,
    DEFAULT_RATED_POWER_MW,
    DEFAULT_TURBINE_MODEL,
    effective_capacity_mw,
    load_power_curve,
    plant_power_from_wind_speed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "wind_hourly.csv"
DEFAULT_HOURLY_OUTPUT_PATH = PROJECT_ROOT / "data" / "power_ab_hourly.csv"
DEFAULT_DAILY_OUTPUT_PATH = PROJECT_ROOT / "data" / "power_ab_daily.csv"
EXPECTED_HOURS_PER_DAY = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert hourly wind speed to hourly Alto Baguales power and "
            "then aggregate complete 24-hour days to daily mean power."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--hourly_output", type=Path, default=DEFAULT_HOURLY_OUTPUT_PATH)
    parser.add_argument("--daily_output", type=Path, default=DEFAULT_DAILY_OUTPUT_PATH)
    parser.add_argument("--timestamp_column", type=str, default="source_timestamp")
    parser.add_argument("--wind_column", type=str, default="wind_ab")
    parser.add_argument("--power_curve", type=Path, default=DEFAULT_ALTO_BAGUALES_CURVE)
    parser.add_argument("--rated_power_mw", type=float, default=DEFAULT_RATED_POWER_MW)
    parser.add_argument("--hub_height_m", type=float, default=DEFAULT_HUB_HEIGHT_M)
    parser.add_argument("--turbine_model", type=str, default=DEFAULT_TURBINE_MODEL)
    parser.add_argument("--cut_out_speed_ms", type=float, default=None)
    parser.add_argument("--n_turbines", type=float, default=DEFAULT_N_TURBINES)
    parser.add_argument("--availability", type=float, default=1.0)
    parser.add_argument("--electrical_efficiency", type=float, default=1.0)
    parser.add_argument(
        "--incomplete_day_policy",
        choices=["error", "drop"],
        default="error",
        help=(
            "error: fail if an observed calendar day does not contain exactly "
            "24 hourly observations. drop: remove incomplete days."
        ),
    )
    parser.add_argument(
        "--no_hourly_output",
        action="store_true",
        help="Do not save the intermediate hourly wind/power audit file.",
    )
    return parser.parse_args()


def load_hourly_wind(
    csv_path: str | Path,
    *,
    timestamp_column: str = "source_timestamp",
    wind_column: str = "wind_ab",
) -> pd.DataFrame:
    """Load and validate the hourly wind dataset."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    required = {timestamp_column, wind_column}
    missing = required.difference(columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {columns}"
        )

    frame = pd.read_csv(csv_path, usecols=[timestamp_column, wind_column])
    frame = frame.rename(
        columns={timestamp_column: "TIMESTAMP", wind_column: "WIND_MS"}
    )

    frame["TIMESTAMP"] = pd.to_datetime(frame["TIMESTAMP"], errors="raise")
    frame["WIND_MS"] = pd.to_numeric(frame["WIND_MS"], errors="raise").astype(float)
    frame = frame.sort_values("TIMESTAMP").reset_index(drop=True)

    if frame["TIMESTAMP"].duplicated().any():
        duplicated = frame.loc[
            frame["TIMESTAMP"].duplicated(keep=False), "TIMESTAMP"
        ]
        raise ValueError(
            "Duplicated hourly timestamps were found. Examples:\n"
            f"{duplicated.head(10).to_string(index=False)}"
        )
    if frame["WIND_MS"].isna().any() or not np.isfinite(frame["WIND_MS"]).all():
        raise ValueError("Hourly wind speed contains missing or non-finite values.")
    if (frame["WIND_MS"] < 0).any():
        raise ValueError("Hourly wind speed contains negative values.")

    return frame


def hourly_gap_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return discontinuities where consecutive observations are not one hour apart."""
    delta = frame["TIMESTAMP"].diff()
    irregular = delta.index[delta.notna() & (delta != pd.Timedelta(hours=1))]
    rows: list[dict[str, object]] = []

    for idx in irregular:
        left = frame.loc[idx - 1, "TIMESTAMP"]
        right = frame.loc[idx, "TIMESTAMP"]
        difference = right - left
        rows.append(
            {
                "previous_timestamp": left,
                "next_timestamp": right,
                "time_difference_hours": int(difference / pd.Timedelta(hours=1)),
                "missing_hours": int(difference / pd.Timedelta(hours=1)) - 1,
            }
        )

    return pd.DataFrame(rows)


def day_completeness_table(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["DATE"] = working["TIMESTAMP"].dt.floor("D")
    working["HOUR"] = working["TIMESTAMP"].dt.hour

    stats = (
        working.groupby("DATE", as_index=False)
        .agg(
            N_ROWS=("WIND_MS", "size"),
            N_UNIQUE_HOURS=("HOUR", "nunique"),
            MIN_HOUR=("HOUR", "min"),
            MAX_HOUR=("HOUR", "max"),
        )
        .sort_values("DATE")
        .reset_index(drop=True)
    )
    stats["IS_COMPLETE"] = (
        (stats["N_ROWS"] == EXPECTED_HOURS_PER_DAY)
        & (stats["N_UNIQUE_HOURS"] == EXPECTED_HOURS_PER_DAY)
        & (stats["MIN_HOUR"] == 0)
        & (stats["MAX_HOUR"] == 23)
    )
    return stats


def keep_complete_days(
    frame: pd.DataFrame,
    *,
    policy: str,
) -> pd.DataFrame:
    """Enforce or filter complete 24-hour observed days."""
    stats = day_completeness_table(frame)
    bad = stats.loc[~stats["IS_COMPLETE"]]

    if bad.empty:
        return frame.copy()

    if policy == "error":
        raise ValueError(
            "At least one observed day is incomplete. Examples:\n"
            f"{bad.head(20).to_string(index=False)}"
        )
    if policy != "drop":
        raise ValueError("policy must be 'error' or 'drop'.")

    valid_dates = set(stats.loc[stats["IS_COMPLETE"], "DATE"])
    dates = frame["TIMESTAMP"].dt.floor("D")
    return frame.loc[dates.isin(valid_dates)].reset_index(drop=True)


def convert_hourly_wind_to_power(
    frame: pd.DataFrame,
    *,
    power_curve: str | Path = DEFAULT_ALTO_BAGUALES_CURVE,
    rated_power_mw: float = DEFAULT_RATED_POWER_MW,
    hub_height_m: float = DEFAULT_HUB_HEIGHT_M,
    turbine_model: str = DEFAULT_TURBINE_MODEL,
    cut_out_speed_ms: float | None = None,
    n_turbines: float = DEFAULT_N_TURBINES,
    availability: float = 1.0,
    electrical_efficiency: float = 1.0,
) -> tuple[pd.DataFrame, object, float]:
    """Apply the power curve independently to every hourly wind observation."""
    curve = load_power_curve(
        power_curve,
        rated_power_mw=rated_power_mw,
        hub_height_m=hub_height_m,
        turbine_model=turbine_model,
        cut_out_speed_ms=cut_out_speed_ms,
        rescale_to_rated_power=True,
    )

    output = frame.copy()
    output["POWER_MW"] = plant_power_from_wind_speed(
        output["WIND_MS"].to_numpy(dtype=float),
        curve,
        n_turbines=n_turbines,
        availability=availability,
        electrical_efficiency=electrical_efficiency,
        output_unit="MW",
    )

    capacity_mw = effective_capacity_mw(
        curve,
        n_turbines=n_turbines,
        availability=availability,
        electrical_efficiency=electrical_efficiency,
    )

    return output, curve, capacity_mw


def hourly_power_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly power to daily mean power and daily energy.

    Because only complete 24-hour days are admitted:
        ENERGY_MWH = sum_h POWER_MW_h * 1 h
        POWER_MW   = mean_h POWER_MW_h
        ENERGY_MWH = 24 * POWER_MW
    """
    working = hourly.copy()
    working["DATE"] = working["TIMESTAMP"].dt.floor("D")

    daily = (
        working.groupby("DATE", as_index=False)
        .agg(
            POWER_MW=("POWER_MW", "mean"),
            ENERGY_MWH=("POWER_MW", "sum"),
            N_HOURS=("POWER_MW", "size"),
        )
        .sort_values("DATE")
        .reset_index(drop=True)
    )

    if not (daily["N_HOURS"] == EXPECTED_HOURS_PER_DAY).all():
        raise RuntimeError("Daily aggregation encountered a non-24-hour day.")

    daily = daily.drop(columns="N_HOURS")
    daily["POWER_MW"] = daily["POWER_MW"].astype(np.float32)
    daily["ENERGY_MWH"] = daily["ENERGY_MWH"].astype(np.float32)
    return daily


def daily_gap_table(daily: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(daily["DATE"])
    rows: list[dict[str, object]] = []
    for left, right in zip(dates[:-1], dates[1:]):
        difference = (right - left).days
        if difference != 1:
            rows.append(
                {
                    "previous_date": left,
                    "next_date": right,
                    "missing_days": difference - 1,
                    "first_missing_date": left + pd.Timedelta(days=1),
                    "last_missing_date": right - pd.Timedelta(days=1),
                }
            )
    return pd.DataFrame(rows)


def save_outputs(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    hourly_output: Path,
    daily_output: Path,
    save_hourly: bool,
) -> None:
    daily_output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_output, index=False, date_format="%Y-%m-%d")

    if save_hourly:
        hourly_output.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(hourly_output, index=False, date_format="%Y-%m-%d %H:%M:%S")


def save_metadata(
    daily_output: Path,
    *,
    input_path: Path,
    power_curve_path: Path,
    curve,
    effective_capacity: float,
    n_turbines: float,
    availability: float,
    electrical_efficiency: float,
    hourly_rows: int,
    daily_rows: int,
) -> Path:
    metadata_path = daily_output.with_suffix(".metadata.json")
    metadata = {
        "source_hourly_file": str(input_path),
        "power_curve_file": str(power_curve_path),
        "target_column": "POWER_MW",
        "energy_column": "ENERGY_MWH",
        "turbine_model": curve.turbine_model,
        "hub_height_m": curve.hub_height_m,
        "rated_power_mw_per_turbine": curve.rated_power_mw,
        "cut_out_speed_ms": curve.cut_out_speed_ms,
        "n_turbines": n_turbines,
        "availability": availability,
        "electrical_efficiency": electrical_efficiency,
        "effective_capacity_mw": effective_capacity,
        "hourly_rows": hourly_rows,
        "daily_rows": daily_rows,
        "daily_aggregation": "mean hourly power over complete 24-hour days",
        "energy_definition": "sum of 24 hourly POWER_MW values times 1 hour",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def main() -> None:
    args = parse_args()

    hourly_wind = load_hourly_wind(
        args.input,
        timestamp_column=args.timestamp_column,
        wind_column=args.wind_column,
    )
    gaps = hourly_gap_table(hourly_wind)
    complete_hourly = keep_complete_days(
        hourly_wind,
        policy=args.incomplete_day_policy,
    )

    hourly_power, curve, capacity_mw = convert_hourly_wind_to_power(
        complete_hourly,
        power_curve=args.power_curve,
        rated_power_mw=args.rated_power_mw,
        hub_height_m=args.hub_height_m,
        turbine_model=args.turbine_model,
        cut_out_speed_ms=args.cut_out_speed_ms,
        n_turbines=args.n_turbines,
        availability=args.availability,
        electrical_efficiency=args.electrical_efficiency,
    )

    daily = hourly_power_to_daily(hourly_power)
    daily_gaps = daily_gap_table(daily)

    save_outputs(
        hourly_power,
        daily,
        hourly_output=args.hourly_output,
        daily_output=args.daily_output,
        save_hourly=not args.no_hourly_output,
    )
    metadata_path = save_metadata(
        args.daily_output,
        input_path=args.input,
        power_curve_path=args.power_curve,
        curve=curve,
        effective_capacity=capacity_mw,
        n_turbines=args.n_turbines,
        availability=args.availability,
        electrical_efficiency=args.electrical_efficiency,
        hourly_rows=len(hourly_power),
        daily_rows=len(daily),
    )

    print("\n=== WIND -> POWER PREPROCESSING ===")
    print(f"Hourly source:       {args.input}")
    print(f"Hourly observations: {len(hourly_wind):,}")
    print(f"Hourly gap events:   {len(gaps):,}")
    print(f"Complete hours used: {len(hourly_power):,}")
    print(f"Daily observations:  {len(daily):,}")
    print(f"Daily gaps:          {len(daily_gaps):,}")
    print(f"Rated power/turbine: {curve.rated_power_mw:.3f} MW")
    print(f"Effective capacity:  {capacity_mw:.3f} MW")
    print(f"Daily dataset:       {args.daily_output}")
    if not args.no_hourly_output:
        print(f"Hourly audit file:   {args.hourly_output}")
    print(f"Metadata:            {metadata_path}")


if __name__ == "__main__":
    main()
