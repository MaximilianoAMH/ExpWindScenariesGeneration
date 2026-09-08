from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "wind_ab_daily.csv"

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
    """
    Chronological split for the new WIND_AB daily dataset.

    Development:
        1980-2013

    Test:
        2014-2017

    Validation:
        one complete target month per Development year,
        rotating one month forward every year.

    With validation_start_month=11:
        1980 -> November
        1981 -> December
        1982 -> January
        ...
        1991 -> October
        1992 -> November
        ...
        2013 -> August
    """

    development_start_year: int = 1980
    development_end_year: int = 2013

    test_start_year: int = 2014
    test_end_year: int = 2017

    validation_start_month: int = 11


@dataclass(frozen=True)
class ForecastConfig:
    """
    Continuous daily forecasting task.

    Train:
        32 consecutive observed days -> next 28 consecutive days

    Validation:
        previous 32 days -> first 28 days of the rotating validation month

    Test:
        previous 32 days -> next 28 days,
        with one origin on the first day of every month.
    """

    context_len: int = 32
    horizon_len: int = 28
    model_context_len: int = 32


@dataclass(frozen=True)
class SamplingConfig:
    """
    Random Train-window sampling.

    min_start_distance_days:
        minimum separation between the START dates of two accepted Train
        windows belonging to the same year.

    With the longer 1980-2013 Development period, 1000 samples is a more
    appropriate baseline than the 180 used with the older dataset.
    """

    num_samples: int = 900
    min_start_distance_days: int = 7
    seed: int = 42


def load_wind_dataframe(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """
    Load the already preprocessed daily CSV.

    Expected columns:
        DATE
        WIND

    Missing calendar days are ALLOWED globally. They are handled by rejecting
    any Train/Validation/Test window that is not internally daily-continuous.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed daily dataset not found: {data_path}"
        )

    df = pd.read_csv(
        data_path,
        usecols=["DATE", "WIND"],
    )

    df["DATE"] = pd.to_datetime(
        df["DATE"],
        format="%Y-%m-%d",
        errors="raise",
    )

    df["WIND"] = pd.to_numeric(
        df["WIND"],
        errors="raise",
    ).astype(np.float32)

    df = (
        df.sort_values("DATE")
        .reset_index(drop=True)
    )

    if df["DATE"].duplicated().any():
        duplicated = df.loc[
            df["DATE"].duplicated(keep=False),
            "DATE",
        ]

        raise ValueError(
            "Duplicated daily DATE values were found. Examples:\n"
            f"{duplicated.head(10).to_string(index=False)}"
        )

    if df["WIND"].isna().any():
        raise ValueError("WIND contains missing values.")

    if not np.isfinite(
        df["WIND"].to_numpy()
    ).all():
        raise ValueError(
            "WIND contains non-finite values."
        )

    first_year = df["DATE"].iloc[0].year
    last_year = df["DATE"].iloc[-1].year

    if first_year != 1980:
        raise ValueError(
            f"Expected the daily dataset to start in 1980, "
            f"but first observed date is {df['DATE'].iloc[0].date()}."
        )

    if last_year != 2017:
        raise ValueError(
            f"Expected the daily dataset to end in 2017, "
            f"but last observed date is {df['DATE'].iloc[-1].date()}."
        )

    return df


def validation_month_for_year(
    year: int,
    split: SplitConfig = SplitConfig(),
) -> int:
    """
    Rotate the validation month by one month every year.
    """
    if not (
        split.development_start_year
        <= year
        <= split.development_end_year
    ):
        raise ValueError(
            f"Year {year} is outside the Development period."
        )

    offset = (
        year
        - split.development_start_year
    )

    return (
        (
            split.validation_start_month
            - 1
            + offset
        )
        % 12
    ) + 1


def validation_schedule(
    split: SplitConfig = SplitConfig(),
) -> dict[int, int]:
    return {
        year: validation_month_for_year(
            year,
            split,
        )
        for year in range(
            split.development_start_year,
            split.development_end_year + 1,
        )
    }


def split_summary(
    split: SplitConfig = SplitConfig(),
) -> pd.DataFrame:
    schedule = validation_schedule(split)

    return pd.DataFrame(
        [
            {
                "year": year,
                "validation_month": month,
                "validation_month_name": MONTH_NAMES[month],
            }
            for year, month in schedule.items()
        ]
    )


def is_daily_continuous(
    dates: pd.DatetimeIndex,
) -> bool:
    """
    True only if every internal step is exactly one calendar day.
    """
    if len(dates) <= 1:
        return True

    differences = np.diff(
        dates.values.astype("datetime64[D]")
    )

    return bool(
        np.all(
            differences
            == np.timedelta64(1, "D")
        )
    )


def find_daily_gaps(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a diagnostic table with all missing calendar intervals.
    """
    dates = pd.DatetimeIndex(df["DATE"])
    rows = []

    for left, right in zip(
        dates[:-1],
        dates[1:],
    ):
        difference = (
            right - left
        ).days

        if difference != 1:
            rows.append(
                {
                    "previous_date": left,
                    "next_date": right,
                    "missing_days": difference - 1,
                    "first_missing_date": (
                        left
                        + pd.Timedelta(days=1)
                    ),
                    "last_missing_date": (
                        right
                        - pd.Timedelta(days=1)
                    ),
                }
            )

    return pd.DataFrame(rows)


class WindContinuousRandomTrainDataset(Dataset):
    """
    Random continuous Train windows from 1980-2013.

    A valid Train sample must satisfy ALL of the following:

      1. 32 context observations + 28 target observations.
      2. All 60 observations are consecutive calendar days.
      3. The full window stays inside one Development year.
      4. No observation in the full 60-day window belongs to the
         validation month reserved for that year.
      5. Accepted starts from the same year are separated by at least
         min_start_distance_days.

    Windows crossing one of the known 7-day gaps are therefore rejected
    automatically.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split: SplitConfig = SplitConfig(),
        forecast: ForecastConfig = ForecastConfig(),
        sampling: SamplingConfig = SamplingConfig(),
    ) -> None:
        self.context_len = (
            forecast.context_len
        )
        self.horizon_len = (
            forecast.horizon_len
        )
        self.window_len = (
            self.context_len
            + self.horizon_len
        )

        self.min_distance = (
            sampling.min_start_distance_days
        )

        self.values_by_year: dict[
            int,
            np.ndarray,
        ] = {}

        self.dates_by_year: dict[
            int,
            pd.DatetimeIndex,
        ] = {}

        candidates: dict[
            int,
            np.ndarray,
        ] = {}

        for year in range(
            split.development_start_year,
            split.development_end_year + 1,
        ):
            part = df.loc[
                df["DATE"].dt.year
                == year,
                ["DATE", "WIND"],
            ].reset_index(drop=True)

            if part.empty:
                raise ValueError(
                    f"No observations found for Development year {year}."
                )

            values = (
                part["WIND"]
                .to_numpy(
                    dtype=np.float32
                )
            )

            dates = pd.DatetimeIndex(
                part["DATE"]
            )

            self.values_by_year[
                year
            ] = values

            self.dates_by_year[
                year
            ] = dates

            validation_month = (
                validation_month_for_year(
                    year,
                    split,
                )
            )

            valid_starts: list[
                int
            ] = []

            max_start = (
                len(values)
                - self.window_len
            )

            for start in range(
                max_start + 1
            ):
                end = (
                    start
                    + self.window_len
                )

                window_dates = dates[
                    start:end
                ]

                if len(
                    window_dates
                ) != self.window_len:
                    continue

                # Reject any calendar gap.
                if not is_daily_continuous(
                    window_dates
                ):
                    continue

                # Strictly reserve the selected
                # validation month from Train.
                if np.any(
                    window_dates.month
                    == validation_month
                ):
                    continue

                valid_starts.append(
                    start
                )

            candidates[
                year
            ] = np.asarray(
                valid_starts,
                dtype=np.int64,
            )

        rng = np.random.default_rng(
            sampling.seed
        )

        remaining = {
            year: starts.copy()
            for year, starts
            in candidates.items()
        }

        self.samples: list[
            tuple[int, int]
        ] = []

        while (
            len(self.samples)
            < sampling.num_samples
        ):
            available_years = [
                year
                for year, starts
                in remaining.items()
                if len(starts) > 0
            ]

            if not available_years:
                maximum_obtained = len(
                    self.samples
                )

                raise ValueError(
                    "Not enough valid Train windows to satisfy "
                    "the requested sampling configuration. "
                    f"Generated {maximum_obtained} samples, "
                    f"requested {sampling.num_samples}, "
                    f"min_start_distance_days="
                    f"{self.min_distance}."
                )

            year = int(
                rng.choice(
                    available_years
                )
            )

            starts = remaining[
                year
            ]

            start = int(
                rng.choice(
                    starts
                )
            )

            self.samples.append(
                (year, start)
            )

            # Remove all candidate starts that
            # are too close to this accepted start.
            keep = (
                np.abs(
                    starts - start
                )
                >= self.min_distance
            )

            remaining[
                year
            ] = starts[
                keep
            ]

        self._validate_samples(
            split
        )

    def _validate_samples(
        self,
        split: SplitConfig,
    ) -> None:
        by_year: dict[
            int,
            list[int],
        ] = {}

        for year, start in self.samples:
            by_year.setdefault(
                year,
                [],
            ).append(
                start
            )

            dates = (
                self.dates_by_year[
                    year
                ][
                    start:
                    start + self.window_len
                ]
            )

            if not is_daily_continuous(
                dates
            ):
                raise RuntimeError(
                    "A sampled Train window "
                    f"contains a gap: year={year}, "
                    f"start={start}."
                )

            val_month = (
                validation_month_for_year(
                    year,
                    split,
                )
            )

            if np.any(
                dates.month
                == val_month
            ):
                raise RuntimeError(
                    "Train/Validation leakage "
                    f"detected in year {year}."
                )

        for year, starts in (
            by_year.items()
        ):
            starts = sorted(
                starts
            )

            for left, right in zip(
                starts[:-1],
                starts[1:],
            ):
                if (
                    right - left
                    < self.min_distance
                ):
                    raise RuntimeError(
                        "Minimum start-distance "
                        f"violation in {year}: "
                        f"{left}, {right}."
                    )

    def __len__(
        self,
    ) -> int:
        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx: int,
    ):
        year, start = (
            self.samples[
                idx
            ]
        )

        values = (
            self.values_by_year[
                year
            ]
        )

        context_end = (
            start
            + self.context_len
        )

        target_end = (
            context_end
            + self.horizon_len
        )

        context = torch.from_numpy(
            values[
                start:
                context_end
            ].copy()
        ).to(
            torch.float32
        )

        target = torch.from_numpy(
            values[
                context_end:
                target_end
            ].copy()
        ).to(
            torch.float32
        )

        return (
            context,
            target,
        )

    def sample_metadata(
        self,
        idx: int,
    ) -> dict[str, object]:
        year, start = (
            self.samples[
                idx
            ]
        )

        dates = (
            self.dates_by_year[
                year
            ]
        )

        context_end = (
            start
            + self.context_len
        )

        target_end = (
            context_end
            + self.horizon_len
        )

        val_month = (
            validation_month_for_year(
                year
            )
        )

        return {
            "year": year,
            "validation_month": val_month,
            "validation_month_name": (
                MONTH_NAMES[
                    val_month
                ]
            ),
            "context_start": dates[
                start
            ],
            "context_end": dates[
                context_end - 1
            ],
            "target_start": dates[
                context_end
            ],
            "target_end": dates[
                target_end - 1
            ],
        }

    def samples_per_year(
        self,
    ) -> dict[int, int]:
        counts = {
            year: 0
            for year
            in sorted(
                self.values_by_year
            )
        }

        for year, _ in (
            self.samples
        ):
            counts[
                year
            ] += 1

        return counts


class WindRotatingMonthValidationDataset(Dataset):
    """
    Exactly one continuous Validation forecast per Development year.

    For each rotating validation month:

        context = previous 32 consecutive calendar days
        target  = first 28 days of the validation month

    If a Validation window intersects a missing-data gap, the constructor
    raises an error rather than silently using a discontinuous sequence.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split: SplitConfig = SplitConfig(),
        forecast: ForecastConfig = ForecastConfig(),
    ) -> None:
        self.context_len = (
            forecast.context_len
        )
        self.horizon_len = (
            forecast.horizon_len
        )

        dates = pd.DatetimeIndex(
            df["DATE"]
        )

        values = (
            df["WIND"]
            .to_numpy(
                dtype=np.float32
            )
        )

        date_to_position = {
            date: idx
            for idx, date
            in enumerate(
                dates
            )
        }

        self.items = []

        for year in range(
            split.development_start_year,
            split.development_end_year + 1,
        ):
            month = (
                validation_month_for_year(
                    year,
                    split,
                )
            )

            origin = pd.Timestamp(
                year=year,
                month=month,
                day=1,
            )

            if (
                origin
                not in date_to_position
            ):
                raise ValueError(
                    "Validation origin is missing "
                    f"from the dataset: "
                    f"{origin.date()}."
                )

            origin_idx = (
                date_to_position[
                    origin
                ]
            )

            context_start = (
                origin_idx
                - self.context_len
            )

            target_end = (
                origin_idx
                + self.horizon_len
            )

            if (
                context_start < 0
                or target_end > len(df)
            ):
                raise ValueError(
                    "Insufficient observations "
                    "for Validation window "
                    f"{origin.date()}."
                )

            window_dates = dates[
                context_start:
                target_end
            ]

            if len(
                window_dates
            ) != (
                self.context_len
                + self.horizon_len
            ):
                raise RuntimeError(
                    "Unexpected Validation "
                    f"window length at "
                    f"{origin.date()}."
                )

            if not is_daily_continuous(
                window_dates
            ):
                raise ValueError(
                    "Validation window crosses "
                    "a missing-data gap: "
                    f"{origin.date()}."
                )

            context_dates = dates[
                context_start:
                origin_idx
            ]

            target_dates = dates[
                origin_idx:
                target_end
            ]

            if not np.all(
                target_dates.month
                == month
            ):
                raise RuntimeError(
                    "Validation target leaves "
                    "its assigned calendar month: "
                    f"{origin.date()}."
                )

            context = torch.from_numpy(
                values[
                    context_start:
                    origin_idx
                ].copy()
            ).to(
                torch.float32
            )

            target = torch.from_numpy(
                values[
                    origin_idx:
                    target_end
                ].copy()
            ).to(
                torch.float32
            )

            self.items.append(
                {
                    "year": year,
                    "month": month,
                    "origin": origin,
                    "context": context,
                    "target": target,
                    "context_dates": (
                        context_dates
                    ),
                    "target_dates": (
                        target_dates
                    ),
                }
            )

        expected = (
            split.development_end_year
            - split.development_start_year
            + 1
        )

        if len(
            self.items
        ) != expected:
            raise RuntimeError(
                f"Expected {expected} "
                "Validation windows, "
                f"got {len(self.items)}."
            )

    def __len__(
        self,
    ) -> int:
        return len(
            self.items
        )

    def __getitem__(
        self,
        idx: int,
    ):
        item = (
            self.items[
                idx
            ]
        )

        return (
            item["context"],
            item["target"],
        )

    def window_metadata(
        self,
        idx: int,
    ) -> dict[str, object]:
        item = (
            self.items[
                idx
            ]
        )

        return {
            "year": item[
                "year"
            ],
            "validation_month": item[
                "month"
            ],
            "validation_month_name": (
                MONTH_NAMES[
                    item[
                        "month"
                    ]
                ]
            ),
            "origin": item[
                "origin"
            ],
            "context_start": item[
                "context_dates"
            ][0],
            "context_end": item[
                "context_dates"
            ][-1],
            "target_start": item[
                "target_dates"
            ][0],
            "target_end": item[
                "target_dates"
            ][-1],
        }


class WindMonthlyTestDataset(Dataset):
    """
    Test period:
        2014-2017

    One forecast origin on the first day of every month:

        previous 32 consecutive days
            ->
        next 28 consecutive days

    This gives:
        4 years * 12 months = 48 Test origins.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split: SplitConfig = SplitConfig(),
        forecast: ForecastConfig = ForecastConfig(),
    ) -> None:
        self.context_len = (
            forecast.context_len
        )
        self.horizon_len = (
            forecast.horizon_len
        )

        dates = pd.DatetimeIndex(
            df["DATE"]
        )

        values = (
            df["WIND"]
            .to_numpy(
                dtype=np.float32
            )
        )

        date_to_position = {
            date: idx
            for idx, date
            in enumerate(
                dates
            )
        }

        test_start = pd.Timestamp(
            year=split.test_start_year,
            month=1,
            day=1,
        )

        test_end = pd.Timestamp(
            year=split.test_end_year,
            month=12,
            day=31,
        )

        self.items = []

        for origin in pd.date_range(
            test_start,
            test_end,
            freq="MS",
        ):
            if (
                origin
                not in date_to_position
            ):
                raise ValueError(
                    "Test origin is missing "
                    f"from the dataset: "
                    f"{origin.date()}."
                )

            origin_idx = (
                date_to_position[
                    origin
                ]
            )

            context_start = (
                origin_idx
                - self.context_len
            )

            target_end = (
                origin_idx
                + self.horizon_len
            )

            if (
                context_start < 0
                or target_end > len(df)
            ):
                raise ValueError(
                    "Insufficient observations "
                    "for Test origin "
                    f"{origin.date()}."
                )

            window_dates = dates[
                context_start:
                target_end
            ]

            if not is_daily_continuous(
                window_dates
            ):
                raise ValueError(
                    "Test window crosses "
                    "a missing-data gap: "
                    f"{origin.date()}."
                )

            context_dates = dates[
                context_start:
                origin_idx
            ]

            target_dates = dates[
                origin_idx:
                target_end
            ]

            if (
                target_dates[0]
                < test_start
                or target_dates[-1]
                > test_end
            ):
                raise ValueError(
                    "Test target leaves the "
                    "2014-2017 Test period: "
                    f"{origin.date()}."
                )

            context = torch.from_numpy(
                values[
                    context_start:
                    origin_idx
                ].copy()
            ).to(
                torch.float32
            )

            target = torch.from_numpy(
                values[
                    origin_idx:
                    target_end
                ].copy()
            ).to(
                torch.float32
            )

            self.items.append(
                {
                    "origin": origin,
                    "context": context,
                    "target": target,
                    "context_dates": (
                        context_dates
                    ),
                    "target_dates": (
                        target_dates
                    ),
                }
            )

        expected = (
            (
                split.test_end_year
                - split.test_start_year
                + 1
            )
            * 12
        )

        if len(
            self.items
        ) != expected:
            raise RuntimeError(
                f"Expected {expected} "
                f"Test windows, got "
                f"{len(self.items)}."
            )

    def __len__(
        self,
    ) -> int:
        return len(
            self.items
        )

    def __getitem__(
        self,
        idx: int,
    ):
        item = (
            self.items[
                idx
            ]
        )

        return (
            item["context"],
            item["target"],
        )

    def window_metadata(
        self,
        idx: int,
    ) -> dict[str, object]:
        item = (
            self.items[
                idx
            ]
        )

        origin = item[
            "origin"
        ]

        return {
            "origin": origin,
            "year": origin.year,
            "month": origin.month,
            "month_name": (
                MONTH_NAMES[
                    origin.month
                ]
            ),
            "context_start": item[
                "context_dates"
            ][0],
            "context_end": item[
                "context_dates"
            ][-1],
            "target_start": item[
                "target_dates"
            ][0],
            "target_end": item[
                "target_dates"
            ][-1],
        }


def build_datasets(
    data_path: str | Path = DEFAULT_DATA_PATH,
    split: SplitConfig = SplitConfig(),
    forecast: ForecastConfig = ForecastConfig(),
    sampling: SamplingConfig = SamplingConfig(),
):
    df = load_wind_dataframe(
        data_path
    )

    train_ds = (
        WindContinuousRandomTrainDataset(
            df=df,
            split=split,
            forecast=forecast,
            sampling=sampling,
        )
    )

    val_ds = (
        WindRotatingMonthValidationDataset(
            df=df,
            split=split,
            forecast=forecast,
        )
    )

    test_ds = (
        WindMonthlyTestDataset(
            df=df,
            split=split,
            forecast=forecast,
        )
    )

    return (
        df,
        train_ds,
        val_ds,
        test_ds,
    )


if __name__ == "__main__":
    split = SplitConfig()
    forecast = ForecastConfig()
    sampling = SamplingConfig()

    df = load_wind_dataframe()

    gaps = find_daily_gaps(
        df
    )

    print(
        "\n=== DAILY DATA ==="
    )
    print(
        f"Observed days: "
        f"{len(df):,}"
    )
    print(
        f"Period: "
        f"{df['DATE'].iloc[0].date()} "
        f"-> "
        f"{df['DATE'].iloc[-1].date()}"
    )
    print(
        f"Daily gaps: "
        f"{len(gaps)}"
    )

    if not gaps.empty:
        print(
            gaps.to_string(
                index=False
            )
        )

    print(
        "\n=== ROTATING "
        "VALIDATION SCHEDULE ==="
    )
    print(
        split_summary(
            split
        ).to_string(
            index=False
        )
    )

    (
        _,
        train_ds,
        val_ds,
        test_ds,
    ) = build_datasets(
        split=split,
        forecast=forecast,
        sampling=sampling,
    )

    print(
        "\n=== DATASETS ==="
    )
    print(
        f"Training samples:   "
        f"{len(train_ds)}"
    )
    print(
        f"Validation samples: "
        f"{len(val_ds)}"
    )
    print(
        f"Test samples:       "
        f"{len(test_ds)}"
    )

    print(
        "\n=== TRAIN SAMPLES "
        "PER YEAR ==="
    )
    for year, count in (
        train_ds
        .samples_per_year()
        .items()
    ):
        print(
            f"{year}: {count}"
        )

    print(
        "\n=== VALIDATION "
        "WINDOWS ==="
    )
    for idx in range(
        len(val_ds)
    ):
        print(
            val_ds
            .window_metadata(
                idx
            )
        )

    train_context, train_target = (
        train_ds[0]
    )

    val_context, val_target = (
        val_ds[0]
    )

    test_context, test_target = (
        test_ds[0]
    )

    assert train_context.shape == (
        forecast.context_len,
    )

    assert train_target.shape == (
        forecast.horizon_len,
    )

    assert val_context.shape == (
        forecast.context_len,
    )

    assert val_target.shape == (
        forecast.horizon_len,
    )

    assert test_context.shape == (
        forecast.context_len,
    )

    assert test_target.shape == (
        forecast.horizon_len,
    )

    assert len(
        train_ds
    ) == sampling.num_samples

    assert len(
        val_ds
    ) == (
        split.development_end_year
        - split.development_start_year
        + 1
    )

    assert len(
        test_ds
    ) == 48

    print(
        "\nAll dataset checks passed."
    )