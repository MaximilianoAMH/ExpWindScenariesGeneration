from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DEFAULT_TARGET_COLUMN, is_daily_continuous


@dataclass(frozen=True)
class HistoricalTrajectorySet:
    """Selected historical trajectories used as Schaake rank templates."""

    values: np.ndarray
    metadata: pd.DataFrame
    method: str
    forecast_origin: pd.Timestamp
    value_column: str = DEFAULT_TARGET_COLUMN

    @property
    def n_trajectories(self) -> int:
        return int(self.values.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.values.shape[1])


def _historical_anchor(
    year: int,
    forecast_origin: pd.Timestamp,
) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(
            year=year,
            month=forecast_origin.month,
            day=forecast_origin.day,
        )
    except ValueError:
        return None


def _extract_window(
    df_indexed: pd.DataFrame,
    start: pd.Timestamp,
    horizon_len: int,
    *,
    value_column: str,
) -> tuple[np.ndarray, pd.Timestamp] | None:
    end = start + pd.Timedelta(days=horizon_len - 1)
    expected_dates = pd.date_range(start, end, freq="D")

    if not expected_dates.isin(df_indexed.index).all():
        return None

    part = df_indexed.loc[expected_dates, [value_column]]
    dates = pd.DatetimeIndex(part.index)

    if len(part) != horizon_len or not is_daily_continuous(dates):
        return None

    values = part[value_column].to_numpy(dtype=np.float32, copy=True)
    return values, end


def build_historical_candidates(
    df: pd.DataFrame,
    forecast_origin: str | pd.Timestamp,
    horizon_len: int = 28,
    history_start_year: int = 1980,
    history_end_year: int = 2013,
    method: str = "same_calendar_date",
    window_radius_days: int = 15,
    *,
    value_column: str = DEFAULT_TARGET_COLUMN,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build all valid historical power trajectories before selecting L."""
    forecast_origin = pd.Timestamp(forecast_origin).normalize()

    if horizon_len <= 0:
        raise ValueError("horizon_len must be positive.")
    if history_start_year > history_end_year:
        raise ValueError("history_start_year must be <= history_end_year.")
    if method not in {"same_calendar_date", "moving_window"}:
        raise ValueError(
            "method must be 'same_calendar_date' or 'moving_window'."
        )
    if window_radius_days < 0:
        raise ValueError("window_radius_days must be >= 0.")

    required = {"DATE", value_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}.")

    working = df[["DATE", value_column]].copy()
    working["DATE"] = pd.to_datetime(working["DATE"], errors="raise").dt.normalize()
    working = working.sort_values("DATE").set_index("DATE")

    if working.index.duplicated().any():
        raise ValueError("Historical dataframe contains duplicated DATE values.")

    history_start = pd.Timestamp(year=history_start_year, month=1, day=1)
    history_end = pd.Timestamp(year=history_end_year, month=12, day=31)

    values_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []

    for year in range(history_start_year, history_end_year + 1):
        anchor = _historical_anchor(year, forecast_origin)
        if anchor is None:
            continue

        offsets = (
            [0]
            if method == "same_calendar_date"
            else range(-window_radius_days, window_radius_days + 1)
        )

        for offset in offsets:
            start = anchor + pd.Timedelta(days=int(offset))
            end = start + pd.Timedelta(days=horizon_len - 1)

            if start < history_start or end > history_end:
                continue

            extracted = _extract_window(
                working,
                start=start,
                horizon_len=horizon_len,
                value_column=value_column,
            )
            if extracted is None:
                continue

            values, end = extracted
            values_rows.append(values)
            metadata_rows.append(
                {
                    "trajectory_year": year,
                    "anchor_date": anchor,
                    "start_date": start,
                    "end_date": end,
                    "seasonal_offset_days": int((start - anchor).days),
                    "abs_seasonal_offset_days": abs(int((start - anchor).days)),
                }
            )

    if not values_rows:
        raise ValueError(
            "No valid historical trajectories were found for the requested configuration."
        )

    return (
        np.vstack(values_rows).astype(np.float32, copy=False),
        pd.DataFrame(metadata_rows),
    )


def _greedy_select_with_min_distance(
    metadata: pd.DataFrame,
    order: np.ndarray,
    n_trajectories: int | None,
    min_start_distance_days: int,
) -> np.ndarray:
    selected: list[int] = []
    selected_starts_by_year: dict[int, list[pd.Timestamp]] = {}

    for idx in order:
        row = metadata.iloc[int(idx)]
        year = int(row["trajectory_year"])
        start = pd.Timestamp(row["start_date"])
        previous = selected_starts_by_year.get(year, [])

        if min_start_distance_days > 0 and any(
            abs((start - other).days) < min_start_distance_days
            for other in previous
        ):
            continue

        selected.append(int(idx))
        selected_starts_by_year.setdefault(year, []).append(start)

        if n_trajectories is not None and len(selected) >= n_trajectories:
            break

    return np.asarray(selected, dtype=np.int64)


def select_historical_trajectories(
    df: pd.DataFrame,
    forecast_origin: str | pd.Timestamp,
    horizon_len: int = 28,
    n_trajectories: int | None = None,
    history_start_year: int = 1980,
    history_end_year: int = 2013,
    method: str = "same_calendar_date",
    window_radius_days: int = 15,
    selection: str = "closest",
    min_start_distance_days: int = 0,
    seed: int = 42,
    strict_n: bool = True,
    *,
    value_column: str = DEFAULT_TARGET_COLUMN,
) -> HistoricalTrajectorySet:
    """Select L complete historical POWER_MW trajectories for Schaake Shuffle."""
    if n_trajectories is not None and n_trajectories <= 0:
        raise ValueError("n_trajectories must be positive or None.")
    if min_start_distance_days < 0:
        raise ValueError("min_start_distance_days must be >= 0.")
    if selection not in {"closest", "random"}:
        raise ValueError("selection must be 'closest' or 'random'.")

    candidate_values, candidate_metadata = build_historical_candidates(
        df=df,
        forecast_origin=forecast_origin,
        horizon_len=horizon_len,
        history_start_year=history_start_year,
        history_end_year=history_end_year,
        method=method,
        window_radius_days=window_radius_days,
        value_column=value_column,
    )

    n_candidates = len(candidate_metadata)

    if selection == "closest":
        order = (
            candidate_metadata
            .assign(_candidate_index=np.arange(n_candidates))
            .sort_values(
                [
                    "abs_seasonal_offset_days",
                    "trajectory_year",
                    "start_date",
                ],
                kind="stable",
            )["_candidate_index"]
            .to_numpy(dtype=np.int64)
        )
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_candidates)

    selected_idx = _greedy_select_with_min_distance(
        metadata=candidate_metadata,
        order=order,
        n_trajectories=n_trajectories,
        min_start_distance_days=min_start_distance_days,
    )

    if (
        n_trajectories is not None
        and len(selected_idx) < n_trajectories
        and strict_n
    ):
        raise ValueError(
            "Could not select the requested number of historical trajectories. "
            f"Requested L={n_trajectories}, selected={len(selected_idx)}, "
            f"raw candidates={n_candidates}. Increase window_radius_days or "
            "reduce min_start_distance_days."
        )

    selected_values = candidate_values[selected_idx, :]
    selected_metadata = (
        candidate_metadata.iloc[selected_idx].copy().reset_index(drop=True)
    )
    selected_metadata.insert(
        0,
        "trajectory_id",
        [f"hist_{i + 1:03d}" for i in range(len(selected_metadata))],
    )
    selected_metadata["selection_rank"] = np.arange(
        1, len(selected_metadata) + 1
    )
    selected_metadata["value_column"] = value_column

    return HistoricalTrajectorySet(
        values=selected_values,
        metadata=selected_metadata,
        method=method,
        forecast_origin=pd.Timestamp(forecast_origin).normalize(),
        value_column=value_column,
    )
