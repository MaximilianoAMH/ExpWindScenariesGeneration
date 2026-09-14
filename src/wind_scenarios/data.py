from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "power_ab_daily.csv"
DEFAULT_TARGET_COLUMN = "POWER_MW"
DEFAULT_ENERGY_COLUMN = "ENERGY_MWH"

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


@dataclass(frozen=True)
class SplitConfig:
    development_start_year: int = 1980
    development_end_year: int = 2013
    test_start_year: int = 2014
    test_end_year: int = 2017
    validation_start_month: int = 11


@dataclass(frozen=True)
class ForecastConfig:
    context_len: int = 32
    horizon_len: int = 28
    model_context_len: int = 32


@dataclass(frozen=True)
class SamplingConfig:
    num_samples: int = 900
    min_start_distance_days: int = 7
    seed: int = 42


def load_timeseries_dataframe(
    data_path: str | Path = DEFAULT_DATA_PATH,
    *,
    value_column: str = DEFAULT_TARGET_COLUMN,
    expected_start_year: int | None = 1980,
    expected_end_year: int | None = 2017,
) -> pd.DataFrame:
    """
    Load a daily numerical time series.

    The zero-shot power pipeline uses ``value_column='POWER_MW'``. Missing
    calendar days are allowed globally; individual context, target and
    historical windows must still be internally daily-continuous.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Processed daily dataset not found: {data_path}")

    columns = pd.read_csv(data_path, nrows=0).columns.tolist()
    required = {"DATE", value_column}
    missing = required.difference(columns)
    if missing:
        raise ValueError(
            f"{data_path} is missing required columns: {sorted(missing)}. "
            f"Available columns: {columns}"
        )

    usecols = ["DATE", value_column]
    if DEFAULT_ENERGY_COLUMN in columns and DEFAULT_ENERGY_COLUMN != value_column:
        usecols.append(DEFAULT_ENERGY_COLUMN)

    df = pd.read_csv(data_path, usecols=usecols)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d", errors="raise")
    df[value_column] = pd.to_numeric(df[value_column], errors="raise").astype(np.float32)

    if DEFAULT_ENERGY_COLUMN in df.columns:
        df[DEFAULT_ENERGY_COLUMN] = pd.to_numeric(
            df[DEFAULT_ENERGY_COLUMN], errors="raise"
        ).astype(np.float32)

    df = df.sort_values("DATE").reset_index(drop=True)

    if df["DATE"].duplicated().any():
        duplicated = df.loc[df["DATE"].duplicated(keep=False), "DATE"]
        raise ValueError(
            "Duplicated daily DATE values were found. Examples:\n"
            f"{duplicated.head(10).to_string(index=False)}"
        )

    values = df[value_column].to_numpy(dtype=float)
    if np.isnan(values).any() or not np.isfinite(values).all():
        raise ValueError(f"{value_column} contains missing or non-finite values.")

    if value_column == DEFAULT_TARGET_COLUMN and np.any(values < 0):
        raise ValueError("POWER_MW contains negative values.")

    if df.empty:
        raise ValueError("The daily dataset is empty.")

    if expected_start_year is not None and df["DATE"].iloc[0].year != expected_start_year:
        raise ValueError(
            f"Expected dataset to start in {expected_start_year}, but first "
            f"observed date is {df['DATE'].iloc[0].date()}."
        )

    if expected_end_year is not None and df["DATE"].iloc[-1].year != expected_end_year:
        raise ValueError(
            f"Expected dataset to end in {expected_end_year}, but last "
            f"observed date is {df['DATE'].iloc[-1].date()}."
        )

    return df


def load_power_dataframe(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    return load_timeseries_dataframe(
        data_path,
        value_column=DEFAULT_TARGET_COLUMN,
    )


def validation_month_for_year(
    year: int,
    split: SplitConfig = SplitConfig(),
) -> int:
    if not split.development_start_year <= year <= split.development_end_year:
        raise ValueError(f"Year {year} is outside the Development period.")

    offset = year - split.development_start_year
    return ((split.validation_start_month - 1 + offset) % 12) + 1


def validation_schedule(
    split: SplitConfig = SplitConfig(),
) -> dict[int, int]:
    return {
        year: validation_month_for_year(year, split)
        for year in range(split.development_start_year, split.development_end_year + 1)
    }


def split_summary(
    split: SplitConfig = SplitConfig(),
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "year": year,
                "validation_month": month,
                "validation_month_name": MONTH_NAMES[month],
            }
            for year, month in validation_schedule(split).items()
        ]
    )


def is_daily_continuous(dates: pd.DatetimeIndex) -> bool:
    if len(dates) <= 1:
        return True
    differences = np.diff(dates.values.astype("datetime64[D]"))
    return bool(np.all(differences == np.timedelta64(1, "D")))


def find_daily_gaps(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(df["DATE"])
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


class ContinuousRandomTrainDataset(Dataset):
    """Random continuous development windows for optional fine-tuning."""

    def __init__(
        self,
        df: pd.DataFrame,
        split: SplitConfig = SplitConfig(),
        forecast: ForecastConfig = ForecastConfig(),
        sampling: SamplingConfig = SamplingConfig(),
        *,
        value_column: str = DEFAULT_TARGET_COLUMN,
    ) -> None:
        if value_column not in df.columns:
            raise ValueError(f"Missing value column: {value_column}")

        self.value_column = value_column
        self.context_len = forecast.context_len
        self.horizon_len = forecast.horizon_len
        self.window_len = self.context_len + self.horizon_len
        self.min_distance = sampling.min_start_distance_days

        self.values_by_year: dict[int, np.ndarray] = {}
        self.dates_by_year: dict[int, pd.DatetimeIndex] = {}
        candidates: dict[int, np.ndarray] = {}

        for year in range(split.development_start_year, split.development_end_year + 1):
            part = df.loc[
                df["DATE"].dt.year == year,
                ["DATE", value_column],
            ].reset_index(drop=True)

            if part.empty:
                raise ValueError(f"No observations found for Development year {year}.")

            values = part[value_column].to_numpy(dtype=np.float32)
            dates = pd.DatetimeIndex(part["DATE"])
            self.values_by_year[year] = values
            self.dates_by_year[year] = dates

            val_month = validation_month_for_year(year, split)
            valid_starts: list[int] = []
            max_start = len(values) - self.window_len

            for start in range(max_start + 1):
                window_dates = dates[start : start + self.window_len]
                if len(window_dates) != self.window_len:
                    continue
                if not is_daily_continuous(window_dates):
                    continue
                if np.any(window_dates.month == val_month):
                    continue
                valid_starts.append(start)

            candidates[year] = np.asarray(valid_starts, dtype=np.int64)

        rng = np.random.default_rng(sampling.seed)
        remaining = {year: starts.copy() for year, starts in candidates.items()}
        self.samples: list[tuple[int, int]] = []

        while len(self.samples) < sampling.num_samples:
            available_years = [year for year, starts in remaining.items() if len(starts) > 0]
            if not available_years:
                raise ValueError(
                    "Not enough valid Train windows for the requested sampling "
                    f"configuration. Generated {len(self.samples)}, requested "
                    f"{sampling.num_samples}."
                )

            year = int(rng.choice(available_years))
            starts = remaining[year]
            start = int(rng.choice(starts))
            self.samples.append((year, start))
            remaining[year] = starts[np.abs(starts - start) >= self.min_distance]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        year, start = self.samples[idx]
        values = self.values_by_year[year]
        context_end = start + self.context_len
        target_end = context_end + self.horizon_len
        context = torch.from_numpy(values[start:context_end].copy()).to(torch.float32)
        target = torch.from_numpy(values[context_end:target_end].copy()).to(torch.float32)
        return context, target

    def sample_metadata(self, idx: int) -> dict[str, object]:
        year, start = self.samples[idx]
        dates = self.dates_by_year[year]
        context_end = start + self.context_len
        target_end = context_end + self.horizon_len
        month = validation_month_for_year(year)
        return {
            "year": year,
            "validation_month": month,
            "validation_month_name": MONTH_NAMES[month],
            "context_start": dates[start],
            "context_end": dates[context_end - 1],
            "target_start": dates[context_end],
            "target_end": dates[target_end - 1],
        }

    def samples_per_year(self) -> dict[int, int]:
        counts = {year: 0 for year in sorted(self.values_by_year)}
        for year, _ in self.samples:
            counts[year] += 1
        return counts


class RotatingMonthValidationDataset(Dataset):
    """One daily-continuous validation window per development year."""

    def __init__(
        self,
        df: pd.DataFrame,
        split: SplitConfig = SplitConfig(),
        forecast: ForecastConfig = ForecastConfig(),
        *,
        value_column: str = DEFAULT_TARGET_COLUMN,
    ) -> None:
        self.context_len = forecast.context_len
        self.horizon_len = forecast.horizon_len
        dates = pd.DatetimeIndex(df["DATE"])
        values = df[value_column].to_numpy(dtype=np.float32)
        date_to_position = {date: idx for idx, date in enumerate(dates)}
        self.items: list[dict[str, object]] = []

        for year in range(split.development_start_year, split.development_end_year + 1):
            month = validation_month_for_year(year, split)
            origin = pd.Timestamp(year=year, month=month, day=1)
            if origin not in date_to_position:
                raise ValueError(f"Validation origin is missing: {origin.date()}.")

            origin_idx = date_to_position[origin]
            context_start = origin_idx - self.context_len
            target_end = origin_idx + self.horizon_len
            if context_start < 0 or target_end > len(df):
                raise ValueError(f"Insufficient observations for {origin.date()}.")

            window_dates = dates[context_start:target_end]
            if not is_daily_continuous(window_dates):
                raise ValueError(f"Validation window crosses a gap: {origin.date()}.")

            target_dates = dates[origin_idx:target_end]
            if not np.all(target_dates.month == month):
                raise RuntimeError(
                    "Validation target leaves its assigned calendar month. "
                    "Use horizon_len <= the available days in that month for this "
                    "validation scheme."
                )

            self.items.append(
                {
                    "year": year,
                    "month": month,
                    "origin": origin,
                    "context": torch.from_numpy(values[context_start:origin_idx].copy()).to(torch.float32),
                    "target": torch.from_numpy(values[origin_idx:target_end].copy()).to(torch.float32),
                    "context_dates": dates[context_start:origin_idx],
                    "target_dates": target_dates,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        return item["context"], item["target"]

    def window_metadata(self, idx: int) -> dict[str, object]:
        item = self.items[idx]
        return {
            "year": item["year"],
            "validation_month": item["month"],
            "validation_month_name": MONTH_NAMES[int(item["month"])],
            "origin": item["origin"],
            "context_start": item["context_dates"][0],
            "context_end": item["context_dates"][-1],
            "target_start": item["target_dates"][0],
            "target_end": item["target_dates"][-1],
        }


class MonthlyTestDataset(Dataset):
    """One forecast origin on the first day of each test month."""

    def __init__(
        self,
        df: pd.DataFrame,
        split: SplitConfig = SplitConfig(),
        forecast: ForecastConfig = ForecastConfig(),
        *,
        value_column: str = DEFAULT_TARGET_COLUMN,
    ) -> None:
        self.context_len = forecast.context_len
        self.horizon_len = forecast.horizon_len
        dates = pd.DatetimeIndex(df["DATE"])
        values = df[value_column].to_numpy(dtype=np.float32)
        date_to_position = {date: idx for idx, date in enumerate(dates)}

        test_start = pd.Timestamp(year=split.test_start_year, month=1, day=1)
        test_end = pd.Timestamp(year=split.test_end_year, month=12, day=31)
        self.items: list[dict[str, object]] = []

        for origin in pd.date_range(test_start, test_end, freq="MS"):
            if origin not in date_to_position:
                raise ValueError(f"Test origin is missing: {origin.date()}.")

            origin_idx = date_to_position[origin]
            context_start = origin_idx - self.context_len
            target_end = origin_idx + self.horizon_len
            if context_start < 0 or target_end > len(df):
                raise ValueError(f"Insufficient observations for Test origin {origin.date()}.")

            window_dates = dates[context_start:target_end]
            if not is_daily_continuous(window_dates):
                raise ValueError(f"Test window crosses a gap: {origin.date()}.")

            target_dates = dates[origin_idx:target_end]
            if target_dates[0] < test_start or target_dates[-1] > test_end:
                raise ValueError(
                    f"Test target leaves the configured Test period: {origin.date()}."
                )

            self.items.append(
                {
                    "origin": origin,
                    "context": torch.from_numpy(values[context_start:origin_idx].copy()).to(torch.float32),
                    "target": torch.from_numpy(values[origin_idx:target_end].copy()).to(torch.float32),
                    "context_dates": dates[context_start:origin_idx],
                    "target_dates": target_dates,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        return item["context"], item["target"]

    def window_metadata(self, idx: int) -> dict[str, object]:
        item = self.items[idx]
        origin = pd.Timestamp(item["origin"])
        return {
            "origin": origin,
            "year": origin.year,
            "month": origin.month,
            "month_name": MONTH_NAMES[origin.month],
            "context_start": item["context_dates"][0],
            "context_end": item["context_dates"][-1],
            "target_start": item["target_dates"][0],
            "target_end": item["target_dates"][-1],
        }


# Explicit power-oriented names.
PowerContinuousRandomTrainDataset = ContinuousRandomTrainDataset
PowerRotatingMonthValidationDataset = RotatingMonthValidationDataset
PowerMonthlyTestDataset = MonthlyTestDataset

# Compatibility aliases for the existing fine-tuning experiment imports.
# They now operate on POWER_MW because DEFAULT_DATA_PATH is the power dataset.
WindContinuousRandomTrainDataset = ContinuousRandomTrainDataset
WindRotatingMonthValidationDataset = RotatingMonthValidationDataset
WindMonthlyTestDataset = MonthlyTestDataset
load_wind_dataframe = load_power_dataframe


def build_datasets(
    data_path: str | Path = DEFAULT_DATA_PATH,
    split: SplitConfig = SplitConfig(),
    forecast: ForecastConfig = ForecastConfig(),
    sampling: SamplingConfig = SamplingConfig(),
    *,
    value_column: str = DEFAULT_TARGET_COLUMN,
):
    df = load_timeseries_dataframe(data_path, value_column=value_column)
    train_ds = ContinuousRandomTrainDataset(
        df=df,
        split=split,
        forecast=forecast,
        sampling=sampling,
        value_column=value_column,
    )
    val_ds = RotatingMonthValidationDataset(
        df=df,
        split=split,
        forecast=forecast,
        value_column=value_column,
    )
    test_ds = MonthlyTestDataset(
        df=df,
        split=split,
        forecast=forecast,
        value_column=value_column,
    )
    return df, train_ds, val_ds, test_ds
