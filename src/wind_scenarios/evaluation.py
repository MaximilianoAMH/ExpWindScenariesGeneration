from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    DEFAULT_DATA_PATH,
    DEFAULT_TARGET_COLUMN,
    is_daily_continuous,
    load_timeseries_dataframe,
)


def forecast_dates(
    origin: str | pd.Timestamp,
    horizon_len: int,
) -> pd.DatetimeIndex:
    if horizon_len <= 0:
        raise ValueError("horizon_len must be a positive integer.")

    origin = pd.Timestamp(origin).normalize()
    return pd.date_range(start=origin, periods=horizon_len, freq="D")


def load_ground_truth(
    origin: str | pd.Timestamp,
    horizon_len: int,
    data_path: str | Path = DEFAULT_DATA_PATH,
    *,
    value_column: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    """Load the ex-post observed power values for one forecast horizon."""
    dates = forecast_dates(origin=origin, horizon_len=horizon_len)
    df = load_timeseries_dataframe(data_path, value_column=value_column)

    indexed = (
        df[["DATE", value_column]]
        .copy()
        .assign(DATE=lambda x: pd.to_datetime(x["DATE"]).dt.normalize())
        .set_index("DATE")
    )

    missing = dates[~dates.isin(indexed.index)]
    if len(missing) > 0:
        preview = ", ".join(d.date().isoformat() for d in missing[:10])
        if len(missing) > 10:
            preview += ", ..."
        raise ValueError(
            "Ground truth is incomplete for the requested forecast horizon. "
            f"Missing {len(missing)} date(s): {preview}"
        )

    selected = indexed.loc[dates, value_column]
    selected_dates = pd.DatetimeIndex(selected.index)
    if not is_daily_continuous(selected_dates):
        raise ValueError("Ground-truth dates are not daily-continuous.")

    values = selected.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Ground truth contains non-finite {value_column} values.")

    return pd.DataFrame(
        {
            "DATE": dates,
            value_column: values,
            "LEAD_DAY": np.arange(1, horizon_len + 1, dtype=np.int64),
        }
    )


def save_ground_truth(
    origin: str | pd.Timestamp,
    horizon_len: int,
    output_dir: str | Path,
    data_path: str | Path = DEFAULT_DATA_PATH,
    filename: str = "ground_truth.csv",
    *,
    value_column: str = DEFAULT_TARGET_COLUMN,
) -> Path:
    output_dir = Path(output_dir)
    ground_truth = load_ground_truth(
        origin=origin,
        horizon_len=horizon_len,
        data_path=data_path,
        value_column=value_column,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    ground_truth.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return output_path
