from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "wind_hourly.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "wind_ab_daily.csv"

EXPECTED_HOURS_PER_DAY = 24


def load_hourly_wind_ab(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the hourly CSV and retain the fields needed for preprocessing.

    wind_ab is NOT rescaled here; its original units are preserved.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    # trajectory_year is kept only for diagnostics.
    available_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()

    required = {"source_timestamp", "wind_ab"}
    missing = required.difference(available_columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {available_columns}"
        )

    usecols = ["source_timestamp", "wind_ab"]
    if "trajectory_year" in available_columns:
        usecols.insert(0, "trajectory_year")

    df = pd.read_csv(csv_path, usecols=usecols)

    df["source_timestamp"] = pd.to_datetime(
        df["source_timestamp"],
        format="%Y-%m-%d %H:%M:%S",
        errors="raise",
    )

    df["wind_ab"] = pd.to_numeric(
        df["wind_ab"],
        errors="raise",
    ).astype(np.float32)

    df = df.sort_values("source_timestamp").reset_index(drop=True)

    if df["source_timestamp"].duplicated().any():
        duplicates = df.loc[
            df["source_timestamp"].duplicated(keep=False),
            "source_timestamp",
        ]

        raise ValueError(
            "Duplicated timestamps were found. Examples:\n"
            f"{duplicates.head(10).to_string(index=False)}"
        )

    if df["wind_ab"].isna().any():
        raise ValueError("wind_ab contains missing values.")

    if not np.isfinite(df["wind_ab"].to_numpy()).all():
        raise ValueError("wind_ab contains non-finite values.")

    return df


def find_hourly_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find discontinuities in the hourly calendar.

    A normal step is exactly one hour. A gap is reported but is not
    automatically interpolated.
    """
    timestamps = df["source_timestamp"]

    delta = timestamps.diff()
    irregular_idx = delta.index[
        delta.notna() & (delta != pd.Timedelta(hours=1))
    ]

    rows = []

    for idx in irregular_idx:
        previous = timestamps.iloc[idx - 1]
        current = timestamps.iloc[idx]
        difference = current - previous

        # Number of absent hourly timestamps between the two observed rows.
        missing_hours = int(difference / pd.Timedelta(hours=1)) - 1

        row = {
            "previous_timestamp": previous,
            "next_timestamp": current,
            "time_difference_hours": int(
                difference / pd.Timedelta(hours=1)
            ),
            "missing_hours": missing_hours,
        }

        if "trajectory_year" in df.columns:
            row["previous_trajectory_year"] = df["trajectory_year"].iloc[idx - 1]
            row["next_trajectory_year"] = df["trajectory_year"].iloc[idx]

        rows.append(row)

    return pd.DataFrame(rows)


def validate_complete_observed_days(df: pd.DataFrame) -> None:
    """
    Require every PRESENT calendar day to contain the complete 24 hourly
    observations 00:00,...,23:00.

    Entire missing days are allowed. Partial days are not.
    """
    working = df.copy()
    working["DATE"] = working["source_timestamp"].dt.floor("D")
    working["HOUR"] = working["source_timestamp"].dt.hour

    stats = (
        working.groupby("DATE", as_index=False)
        .agg(
            N_ROWS=("wind_ab", "size"),
            N_UNIQUE_HOURS=("HOUR", "nunique"),
            MIN_HOUR=("HOUR", "min"),
            MAX_HOUR=("HOUR", "max"),
        )
    )

    bad_days = stats.loc[
        (stats["N_ROWS"] != EXPECTED_HOURS_PER_DAY)
        | (stats["N_UNIQUE_HOURS"] != EXPECTED_HOURS_PER_DAY)
        | (stats["MIN_HOUR"] != 0)
        | (stats["MAX_HOUR"] != 23)
    ]

    if not bad_days.empty:
        raise ValueError(
            "At least one observed day is incomplete. "
            "The script will not average partial days. Examples:\n"
            f"{bad_days.head(20).to_string(index=False)}"
        )


def hourly_to_daily_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute one arithmetic mean of wind_ab for each complete observed day.
    """
    working = df.copy()
    working["DATE"] = working["source_timestamp"].dt.floor("D")

    daily = (
        working.groupby("DATE", as_index=False)
        .agg(WIND=("wind_ab", "mean"))
        .sort_values("DATE")
        .reset_index(drop=True)
    )

    daily["WIND"] = daily["WIND"].astype(np.float32)

    return daily


def find_missing_daily_intervals(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize gaps between consecutive observed daily means.
    """
    dates = pd.DatetimeIndex(daily["DATE"])
    delta_days = daily["DATE"].diff().dt.days

    irregular_idx = delta_days.index[
        delta_days.notna() & (delta_days != 1)
    ]

    rows = []

    for idx in irregular_idx:
        previous = dates[idx - 1]
        current = dates[idx]

        missing_days = (current - previous).days - 1

        rows.append(
            {
                "previous_date": previous,
                "next_date": current,
                "missing_days": missing_days,
                "first_missing_date": previous + pd.Timedelta(days=1),
                "last_missing_date": current - pd.Timedelta(days=1),
            }
        )

    return pd.DataFrame(rows)


def save_daily_dataset(
    daily: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    daily.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
    )


def print_summary(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    hourly_gaps: pd.DataFrame,
    daily_gaps: pd.DataFrame,
    output_path: Path,
) -> None:
    print("\n=== HOURLY DATA ===")
    print(f"Rows:              {len(hourly):,}")
    print(
        f"Period:            "
        f"{hourly['source_timestamp'].iloc[0]} -> "
        f"{hourly['source_timestamp'].iloc[-1]}"
    )
    print(f"wind_ab min:       {hourly['wind_ab'].min():.4f}")
    print(f"wind_ab max:       {hourly['wind_ab'].max():.4f}")

    print("\n=== HOURLY GAPS ===")
    print(f"Irregular steps:   {len(hourly_gaps)}")

    if not hourly_gaps.empty:
        print(hourly_gaps.head(20).to_string(index=False))

    print("\n=== DAILY DATA ===")
    print(f"Observed days:     {len(daily):,}")
    print(
        f"Period:            "
        f"{daily['DATE'].iloc[0].date()} -> "
        f"{daily['DATE'].iloc[-1].date()}"
    )
    print(f"Daily WIND mean:   {daily['WIND'].mean():.4f}")
    print(f"Daily WIND std:    {daily['WIND'].std():.4f}")

    print("\n=== MISSING DAILY INTERVALS ===")
    print(f"Daily gaps:        {len(daily_gaps)}")

    if not daily_gaps.empty:
        print(daily_gaps.head(20).to_string(index=False))
        print(
            f"\nTotal missing calendar days: "
            f"{int(daily_gaps['missing_days'].sum())}"
        )

    print("\nFirst daily rows:")
    print(daily.head().to_string(index=False))

    print(f"\nSaved daily dataset to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess hourly wind_ab data into complete-day means. "
            "Calendar gaps are reported but not interpolated."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--strict-continuity",
        action="store_true",
        help=(
            "Fail if any hourly calendar gap exists. By default, complete "
            "missing intervals are reported and preserved as gaps."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hourly = load_hourly_wind_ab(args.input)

    hourly_gaps = find_hourly_gaps(hourly)

    if args.strict_continuity and not hourly_gaps.empty:
        first = hourly_gaps.iloc[0]

        raise ValueError(
            "The hourly series contains calendar gaps. First gap: "
            f"{first['previous_timestamp']} -> "
            f"{first['next_timestamp']} "
            f"({int(first['missing_hours'])} missing hours)."
        )

    # We allow whole missing days, but never average an incomplete day.
    validate_complete_observed_days(hourly)

    daily = hourly_to_daily_mean(hourly)
    daily_gaps = find_missing_daily_intervals(daily)

    save_daily_dataset(
        daily,
        args.output,
    )

    print_summary(
        hourly=hourly,
        daily=daily,
        hourly_gaps=hourly_gaps,
        daily_gaps=daily_gaps,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
