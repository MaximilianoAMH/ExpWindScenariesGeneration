from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import TimesFm2_5ModelForPrediction

from ..data import (
    DEFAULT_DATA_PATH,
    DEFAULT_TARGET_COLUMN,
    PROJECT_ROOT,
    load_timeseries_dataframe,
)
from ..forecasting import (
    apply_value_bounds,
    forecast_quantiles,
    quantile_crossing_rate,
)
from ..historical import select_historical_trajectories
from ..schaake import (
    save_schaake_result,
    schaake_shuffle,
    validate_shuffle_result,
)


MODEL_ID = "google/timesfm-2.5-200m-transformers"
DEFAULT_CONTEXT_LEN = 32
DEFAULT_HORIZON_LEN = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate daily wind-energy scenarios using zero-shot TimesFM 2.5 "
            "marginals and Schaake Shuffle dependence. The target is total "
            "generated energy per complete day, in MWh."
        )
    )

    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--target_column", type=str, default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--origin", type=str, required=True)
    parser.add_argument("--context_len", type=int, default=DEFAULT_CONTEXT_LEN)
    parser.add_argument("--horizon_len", type=int, default=DEFAULT_HORIZON_LEN)

    parser.add_argument(
        "--historical_method",
        choices=["same_calendar_date", "moving_window"],
        default="same_calendar_date",
    )
    parser.add_argument("--n_scenarios", type=int, default=None)
    parser.add_argument("--window_radius_days", type=int, default=15)
    parser.add_argument("--min_start_distance_days", type=int, default=0)
    parser.add_argument(
        "--selection",
        choices=["closest", "random"],
        default="closest",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history_start_year", type=int, default=1980)
    parser.add_argument("--history_end_year", type=int, default=2013)

    parser.add_argument(
        "--probability_grid_mode",
        choices=["timesfm_support", "schaake"],
        default="timesfm_support",
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
        "--energy_bounds_policy",
        choices=["clip", "error", "none"],
        default="clip",
        help=(
            "Physical bounds applied to TimesFM daily-energy quantiles before "
            "Schaake Shuffle. 'clip' is recommended."
        ),
    )
    parser.add_argument(
        "--energy_min_mwh",
        type=float,
        default=0.0,
        help="Physical lower bound for daily generated energy.",
    )
    parser.add_argument(
        "--energy_max_mwh",
        type=float,
        default=None,
        help=(
            "Physical upper bound in MWh/day. If omitted, the program attempts "
            "to read max_daily_energy_mwh from <data_path>.metadata.json."
        ),
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
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def extract_context(
    df: pd.DataFrame,
    origin: pd.Timestamp,
    context_len: int,
    *,
    value_column: str,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Extract only observations strictly before the forecast origin."""
    origin = pd.Timestamp(origin).normalize()
    context_dates = pd.date_range(
        end=origin - pd.Timedelta(days=1),
        periods=context_len,
        freq="D",
    )

    indexed = (
        df[["DATE", value_column]]
        .copy()
        .assign(DATE=lambda x: pd.to_datetime(x["DATE"]).dt.normalize())
        .set_index("DATE")
    )

    missing = context_dates[~context_dates.isin(indexed.index)]
    if len(missing) > 0:
        raise ValueError(
            "The forecast context is incomplete. Missing dates: "
            f"{[d.date().isoformat() for d in missing]}."
        )

    values = indexed.loc[context_dates, value_column].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    return values, context_dates


def save_context(
    context: np.ndarray,
    context_dates: pd.DatetimeIndex,
    output_dir: Path,
    *,
    value_column: str,
) -> None:
    frame = pd.DataFrame(
        {
            "DATE": context_dates,
            value_column: context,
        }
    )
    frame.to_csv(
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
        f"q{int(round(q * 100)):02d}": forecast.values[:, k]
        for k, q in enumerate(forecast.probabilities)
    }
    pd.DataFrame(
        {
            "DATE": forecast_dates,
            **columns,
        }
    ).to_csv(
        output_dir / "timesfm_quantiles.csv",
        index=False,
        date_format="%Y-%m-%d",
    )


def save_historical_values(historical, output_dir: Path) -> None:
    _, H = historical.values.shape
    frame = pd.DataFrame(
        historical.values,
        columns=[f"day_{t + 1:02d}" for t in range(H)],
    )
    frame.insert(
        0,
        "trajectory_id",
        historical.metadata["trajectory_id"].to_numpy(),
    )
    frame.to_csv(
        output_dir / "historical_trajectories.csv",
        index=False,
    )


def preprocessing_metadata_path(data_path: Path) -> Path:
    return data_path.with_suffix(".metadata.json")


def load_preprocessing_metadata(
    data_path: Path,
) -> dict[str, object] | None:
    path = preprocessing_metadata_path(data_path)
    if not path.exists():
        return None
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def resolve_energy_max_mwh(
    requested_max: float | None,
    data_path: Path,
) -> tuple[float | None, str]:
    if requested_max is not None:
        if requested_max <= 0:
            raise ValueError(
                "--energy_max_mwh must be positive."
            )
        return float(requested_max), "CLI"

    metadata = load_preprocessing_metadata(
        data_path
    )
    if (
        metadata is not None
        and metadata.get("max_daily_energy_mwh")
        is not None
    ):
        value = float(
            metadata["max_daily_energy_mwh"]
        )
        if value <= 0:
            raise ValueError(
                "max_daily_energy_mwh in preprocessing metadata "
                "must be positive."
            )
        return value, "preprocessing metadata"

    return None, "not configured"


def save_run_config(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    device: torch.device,
    energy_max_mwh: float | None,
    energy_max_source: str,
) -> None:
    config = {
        "model_id": MODEL_ID,
        "data_path": str(args.data_path),
        "target_column": args.target_column,
        "target_unit": "MWh/day",
        "origin": str(pd.Timestamp(args.origin).date()),
        "context_len": args.context_len,
        "horizon_len": args.horizon_len,
        "historical_method": args.historical_method,
        "n_scenarios": args.n_scenarios,
        "window_radius_days": args.window_radius_days,
        "min_start_distance_days": args.min_start_distance_days,
        "selection": args.selection,
        "history_start_year": args.history_start_year,
        "history_end_year": args.history_end_year,
        "probability_grid_mode": args.probability_grid_mode,
        "crossing_policy": args.crossing_policy,
        "tie_break": args.tie_break,
        "seed": args.seed,
        "device": str(device),
        "energy_bounds_policy": args.energy_bounds_policy,
        "energy_min_mwh": args.energy_min_mwh,
        "energy_max_mwh": energy_max_mwh,
        "energy_max_source": energy_max_source,
    }

    (
        output_dir
        / "run_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if args.context_len <= 0:
        raise ValueError(
            "--context_len must be positive."
        )
    if args.horizon_len <= 0:
        raise ValueError(
            "--horizon_len must be positive."
        )
    if args.energy_min_mwh < 0:
        raise ValueError(
            "--energy_min_mwh must be >= 0."
        )

    device = resolve_device(
        args.device
    )
    model_dtype = (
        torch.bfloat16
        if device.type == "cuda"
        else torch.float32
    )
    origin = pd.Timestamp(
        args.origin
    ).normalize()

    df = load_timeseries_dataframe(
        args.data_path,
        value_column=args.target_column,
    )

    context, context_dates = extract_context(
        df,
        origin,
        args.context_len,
        value_column=args.target_column,
    )

    energy_max_mwh, energy_max_source = (
        resolve_energy_max_mwh(
            args.energy_max_mwh,
            args.data_path,
        )
    )

    print("\n=== FORECAST ORIGIN ===")
    print(f"Origin:          {origin.date()}")
    print(f"Target variable: {args.target_column}")
    print("Target unit:     MWh/day")
    print(
        f"Context:         "
        f"{context_dates[0].date()} -> "
        f"{context_dates[-1].date()}"
    )
    print(
        f"Context length:  "
        f"{args.context_len} days"
    )
    print(
        f"Horizon length:  "
        f"{args.horizon_len} days"
    )
    print(
        "Future target used in generation: NO"
    )

    print(
        f"\nLoading zero-shot model: "
        f"{MODEL_ID}"
    )
    print(
        f"Execution device: "
        f"{device} (dtype={model_dtype})"
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

    model_max_context = getattr(
        model.config,
        "context_length",
        None,
    )
    model_max_horizon = getattr(
        model.config,
        "horizon_length",
        None,
    )

    if (
        model_max_context is not None
        and args.context_len
        > int(model_max_context)
    ):
        raise ValueError(
            f"Requested context_len={args.context_len}, "
            "but TimesFM supports at most "
            f"{model_max_context}."
        )

    if (
        model_max_horizon is not None
        and args.horizon_len
        > int(model_max_horizon)
    ):
        raise ValueError(
            f"Requested horizon_len={args.horizon_len}, "
            "but this checkpoint produces at most "
            f"{model_max_horizon} steps."
        )

    raw_forecast = forecast_quantiles(
        model=model,
        context=context,
        horizon_len=args.horizon_len,
        model_context_len=args.context_len,
        device=device,
    )
    raw_crossing_rate = (
        quantile_crossing_rate(
            raw_forecast
        )
    )

    forecast = apply_value_bounds(
        raw_forecast,
        lower=args.energy_min_mwh,
        upper=energy_max_mwh,
        policy=args.energy_bounds_policy,
    )

    print(
        "\n=== TIMESFM ZERO-SHOT "
        "DAILY-ENERGY MARGINALS ==="
    )
    print(
        f"Quantiles:       "
        f"{forecast.probabilities}"
    )
    print(
        f"Shape:           "
        f"{forecast.values.shape}"
    )
    print(
        f"Crossing rate:   "
        f"{raw_crossing_rate:.6f}"
    )
    print(
        f"Bounds policy:   "
        f"{args.energy_bounds_policy}"
    )
    print(
        f"Lower bound:     "
        f"{args.energy_min_mwh:.6f} MWh/day"
    )
    print(
        f"Upper bound:     "
        f"{energy_max_mwh if energy_max_mwh is not None else 'None'}"
    )
    print(
        f"Upper source:    "
        f"{energy_max_source}"
    )

    historical = select_historical_trajectories(
        df=df,
        forecast_origin=origin,
        horizon_len=args.horizon_len,
        n_trajectories=args.n_scenarios,
        history_start_year=args.history_start_year,
        history_end_year=args.history_end_year,
        method=args.historical_method,
        window_radius_days=args.window_radius_days,
        selection=args.selection,
        min_start_distance_days=args.min_start_distance_days,
        seed=args.seed,
        strict_n=True,
        value_column=args.target_column,
    )

    print(
        "\n=== HISTORICAL DAILY-GENERATION LIBRARY ==="
    )
    print(
        f"Method:          "
        f"{args.historical_method}"
    )
    print(
        f"Trajectories:    "
        f"{historical.n_trajectories}"
    )
    print(
        f"Horizon:         "
        f"{historical.horizon}"
    )

    result = schaake_shuffle(
        forecast=forecast,
        historical=historical,
        probability_grid_mode=args.probability_grid_mode,
        tie_break=args.tie_break,
        seed=args.seed,
        crossing_policy=args.crossing_policy,
    )
    validate_shuffle_result(
        result
    )

    forecast_dates = pd.date_range(
        origin,
        periods=args.horizon_len,
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
        context,
        context_dates,
        run_output_dir,
        value_column=args.target_column,
    )
    save_timesfm_quantiles(
        forecast,
        forecast_dates,
        run_output_dir,
    )
    save_historical_values(
        historical,
        run_output_dir,
    )
    save_schaake_result(
        result,
        run_output_dir,
        forecast_dates=forecast_dates,
        scenarios_filename="energy_scenarios.csv",
    )
    save_run_config(
        run_output_dir,
        args=args,
        device=device,
        energy_max_mwh=energy_max_mwh,
        energy_max_source=energy_max_source,
    )

    print(
        "\n=== SCHAAKE SHUFFLE: "
        "DAILY GENERATION SCENARIOS ==="
    )
    print(
        f"Scenarios shape: "
        f"{result.scenarios.shape}"
    )
    print(
        "Marginal permutation checks: PASSED"
    )
    print(
        "Probability grid: "
        f"{result.marginal_probability_grid[0]:.4f} "
        "-> "
        f"{result.marginal_probability_grid[-1]:.4f}"
    )
    print(
        f"Saved to:        "
        f"{run_output_dir}"
    )
    print(
        "Scenario file:   "
        "energy_scenarios.csv"
    )


if __name__ == "__main__":
    main()
