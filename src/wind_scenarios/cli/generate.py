from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import TimesFm2_5ModelForPrediction

from ..data import (
    DEFAULT_DATA_PATH,
    PROJECT_ROOT,
    load_wind_dataframe,
)
from ..forecasting import (
    forecast_quantiles,
    quantile_crossing_rate,
)
from ..historical import (
    select_historical_trajectories,
)
from ..schaake import (
    schaake_shuffle,
    save_schaake_result,
    validate_shuffle_result,
)


MODEL_ID = "google/timesfm-2.5-200m-transformers"

CONTEXT_LEN = 32
HORIZON_LEN = 28
MODEL_CONTEXT_LEN = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate WIND scenarios using zero-shot TimesFM 2.5 "
            "marginals and Schaake Shuffle dependence."
        )
    )

    parser.add_argument(
        "--data_path",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )

    parser.add_argument(
        "--origin",
        type=str,
        required=True,
        help="Forecast origin, e.g. 2014-04-01.",
    )

    parser.add_argument(
        "--historical_method",
        choices=[
            "same_calendar_date",
            "moving_window",
        ],
        default="same_calendar_date",
    )

    parser.add_argument(
        "--n_scenarios",
        type=int,
        default=None,
        help=(
            "Desired L. Omit to use every eligible candidate "
            "after selection rules."
        ),
    )

    parser.add_argument(
        "--window_radius_days",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--min_start_distance_days",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--selection",
        choices=["closest", "random"],
        default="closest",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--history_start_year",
        type=int,
        default=1980,
    )

    parser.add_argument(
        "--history_end_year",
        type=int,
        default=2013,
    )

    parser.add_argument(
        "--probability_grid_mode",
        choices=[
            "timesfm_support",
            "schaake",
        ],
        default="timesfm_support",
        help=(
            "timesfm_support avoids Q10-Q90 tail extrapolation. "
            "schaake uses k/(L+1) and fails if those levels "
            "leave the TimesFM support."
        ),
    )

    parser.add_argument(
        "--crossing_policy",
        choices=["error", "sort"],
        default="error",
    )

    parser.add_argument(
        "--tie_break",
        choices=["stable", "random"],
        default="stable",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "schaake",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Execution device. 'auto' uses CUDA when available "
            "and otherwise falls back to CPU."
        ),
    )

    return parser.parse_args()


def extract_context(
    df: pd.DataFrame,
    origin: pd.Timestamp,
    context_len: int,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Extract only information available before the forecast origin.

    No target values are accessed.
    """
    origin = pd.Timestamp(
        origin
    ).normalize()

    context_dates = pd.date_range(
        end=origin - pd.Timedelta(days=1),
        periods=context_len,
        freq="D",
    )

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

    if not context_dates.isin(
        indexed.index
    ).all():
        missing = context_dates[
            ~context_dates.isin(indexed.index)
        ]

        raise ValueError(
            "The forecast context is not complete. "
            f"Missing dates: {[d.date().isoformat() for d in missing]}."
        )

    values = (
        indexed.loc[
            context_dates,
            "WIND",
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    return values, context_dates


def save_context(
    context: np.ndarray,
    context_dates: pd.DatetimeIndex,
    output_dir: Path,
) -> None:
    """
    Save the exact observed context supplied to TimesFM.
    """
    if len(context) != len(context_dates):
        raise ValueError(
            "context and context_dates must have the same length."
        )

    context_df = pd.DataFrame(
        {
            "DATE": context_dates,
            "WIND": context,
        }
    )

    context_df.to_csv(
        output_dir / "context.csv",
        index=False,
        date_format="%Y-%m-%d",
    )


def save_timesfm_quantiles(
    forecast,
    forecast_dates: pd.DatetimeIndex,
    output_dir: Path,
) -> None:
    columns = {
        f"q{int(round(q * 100)):02d}":
        forecast.values[:, k]
        for k, q in enumerate(
            forecast.probabilities
        )
    }

    df = pd.DataFrame(
        {
            "DATE": forecast_dates,
            **columns,
        }
    )

    df.to_csv(
        output_dir / "timesfm_quantiles.csv",
        index=False,
        date_format="%Y-%m-%d",
    )


def save_historical_values(
    historical,
    output_dir: Path,
) -> None:
    L, H = historical.values.shape

    df = pd.DataFrame(
        historical.values,
        columns=[
            f"day_{t + 1:02d}"
            for t in range(H)
        ],
    )

    df.insert(
        0,
        "trajectory_id",
        historical.metadata[
            "trajectory_id"
        ].to_numpy(),
    )

    df.to_csv(
        output_dir / "historical_trajectories.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested with --device cuda, "
                "but CUDA is not available."
            )
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_dtype = (
        torch.bfloat16
        if device.type == "cuda"
        else torch.float32
    )
    origin = pd.Timestamp(
        args.origin
    ).normalize()

    df = load_wind_dataframe(
        args.data_path
    )

    context, context_dates = extract_context(
        df=df,
        origin=origin,
        context_len=CONTEXT_LEN,
    )

    print("\n=== FORECAST ORIGIN ===")
    print(f"Origin:       {origin.date()}")
    print(
        f"Context:      "
        f"{context_dates[0].date()} -> "
        f"{context_dates[-1].date()}"
    )
    print("Target used in generation: NO")

    print(f"\nLoading zero-shot model: {MODEL_ID}")

    print(
        f"Execution device: {device} "
        f"(dtype={model_dtype})"
    )

    model = (
        TimesFm2_5ModelForPrediction
        .from_pretrained(
            MODEL_ID,
            dtype=model_dtype,
        )
        .to(device)
    )
    model.eval()

    forecast = forecast_quantiles(
        model=model,
        context=context,
        horizon_len=HORIZON_LEN,
        model_context_len=MODEL_CONTEXT_LEN,
        device=device,
    )

    crossing_rate = quantile_crossing_rate(
        forecast
    )

    print("\n=== TIMESFM ZERO-SHOT ===")
    print(
        f"Quantiles:    "
        f"{forecast.probabilities}"
    )
    print(
        f"Shape:        "
        f"{forecast.values.shape}"
    )
    print(
        f"Crossing rate: "
        f"{crossing_rate:.6f}"
    )

    historical = select_historical_trajectories(
        df=df,
        forecast_origin=origin,
        horizon_len=HORIZON_LEN,
        n_trajectories=args.n_scenarios,
        history_start_year=args.history_start_year,
        history_end_year=args.history_end_year,
        method=args.historical_method,
        window_radius_days=args.window_radius_days,
        selection=args.selection,
        min_start_distance_days=(
            args.min_start_distance_days
        ),
        seed=args.seed,
        strict_n=True,
    )

    print("\n=== HISTORICAL LIBRARY ===")
    print(
        f"Method:       "
        f"{args.historical_method}"
    )
    print(
        f"Trajectories: "
        f"{historical.n_trajectories}"
    )
    print(
        f"Horizon:      "
        f"{historical.horizon}"
    )

    result = schaake_shuffle(
        forecast=forecast,
        historical=historical,
        probability_grid_mode=(
            args.probability_grid_mode
        ),
        tie_break=args.tie_break,
        seed=args.seed,
        crossing_policy=args.crossing_policy,
    )

    validate_shuffle_result(
        result
    )

    forecast_dates = pd.date_range(
        origin,
        periods=HORIZON_LEN,
        freq="D",
    )

    run_output_dir = (
        args.output_dir
        / origin.date().isoformat()
    )
    run_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_context(
        context=context,
        context_dates=context_dates,
        output_dir=run_output_dir,
    )

    save_timesfm_quantiles(
        forecast=forecast,
        forecast_dates=forecast_dates,
        output_dir=run_output_dir,
    )

    save_historical_values(
        historical=historical,
        output_dir=run_output_dir,
    )

    save_schaake_result(
        result=result,
        output_dir=run_output_dir,
        forecast_dates=forecast_dates,
    )

    print("\n=== SCHAAKE SHUFFLE ===")
    print(
        f"Scenarios shape: "
        f"{result.scenarios.shape}"
    )
    print(
        "Marginal permutation checks: PASSED"
    )
    print(
        f"Probability grid: "
        f"{result.marginal_probability_grid[0]:.4f} "
        f"-> "
        f"{result.marginal_probability_grid[-1]:.4f}"
    )
    print(
        f"\nSaved to: {run_output_dir}"
    )


if __name__ == "__main__":
    main()