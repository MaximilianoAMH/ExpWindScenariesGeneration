from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    DEFAULT_DATA_PATH,
    is_daily_continuous,
    load_wind_dataframe,
)


def forecast_dates(
    origin: str | pd.Timestamp,
    horizon_len: int,
) -> pd.DatetimeIndex:
    """
    Return the exact daily dates covered by a forecast horizon.

    Parameters
    ----------
    origin:
        First forecast date.

    horizon_len:
        Number of daily forecast steps.

    Returns
    -------
    pd.DatetimeIndex
        Daily dates from ``origin`` through
        ``origin + horizon_len - 1 days``.
    """
    if horizon_len <= 0:
        raise ValueError(
            "horizon_len must be a positive integer."
        )

    origin = pd.Timestamp(origin).normalize()

    return pd.date_range(
        start=origin,
        periods=horizon_len,
        freq="D",
    )


def load_ground_truth(
    origin: str | pd.Timestamp,
    horizon_len: int,
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """
    Load the observed WIND values corresponding exactly to a forecast horizon.

    This function is intended for ex-post evaluation only. It does not take
    part in TimesFM forecasting or Schaake Shuffle generation.

    The returned DataFrame contains:

        DATE
        WIND
        LEAD_DAY

    where LEAD_DAY starts at 1.

    The function fails explicitly if the complete target horizon is not
    available or is not daily-continuous.
    """
    dates = forecast_dates(
        origin=origin,
        horizon_len=horizon_len,
    )

    df = load_wind_dataframe(data_path)

    indexed = (
        df.loc[:, ["DATE", "WIND"]]
        .copy()
        .assign(
            DATE=lambda x: pd.to_datetime(
                x["DATE"]
            ).dt.normalize()
        )
        .set_index("DATE")
    )

    missing = dates[
        ~dates.isin(indexed.index)
    ]

    if len(missing) > 0:
        preview = ", ".join(
            d.date().isoformat()
            for d in missing[:10]
        )

        if len(missing) > 10:
            preview += ", ..."

        raise ValueError(
            "Ground truth is incomplete for the requested forecast horizon. "
            f"Missing {len(missing)} date(s): {preview}"
        )

    selected = indexed.loc[
        dates,
        "WIND",
    ]

    selected_dates = pd.DatetimeIndex(
        selected.index
    )

    if not is_daily_continuous(
        selected_dates
    ):
        raise ValueError(
            "Ground-truth dates are not daily-continuous."
        )

    values = selected.to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Ground truth contains non-finite WIND values."
        )

    return pd.DataFrame(
        {
            "DATE": dates,
            "WIND": values,
            "LEAD_DAY": np.arange(
                1,
                horizon_len + 1,
                dtype=np.int64,
            ),
        }
    )


def save_ground_truth(
    origin: str | pd.Timestamp,
    horizon_len: int,
    output_dir: str | Path,
    data_path: str | Path = DEFAULT_DATA_PATH,
    filename: str = "ground_truth.csv",
) -> Path:
    """
    Save the observed target horizon to CSV for ex-post evaluation.

    Important
    ---------
    Call this function only after scenario generation has completed if the
    objective is to keep the generation stage clearly separated from future
    observations.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    output_dir = Path(output_dir)

    ground_truth = load_ground_truth(
        origin=origin,
        horizon_len=horizon_len,
        data_path=data_path,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / filename
    )

    ground_truth.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    return output_path
