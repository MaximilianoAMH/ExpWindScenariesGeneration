from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import TimesFm2_5ModelForPrediction

from wind_scenarios.data import (
    DEFAULT_DATA_PATH,
    ForecastConfig,
    SplitConfig,
    WindMonthlyTestDataset,
    load_wind_dataframe,
)

MODEL_ID = "google/timesfm-2.5-200m-transformers"

# TimesFM 2.5 quantile channels in full_predictions.
Q10 = 0.1
Q90 = 0.9
Q10_CHANNEL = 1
Q90_CHANNEL = 9


def parse_args() -> argparse.Namespace:
    """Configuration for the final zero-shot vs fine-tuned Test comparison."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare TimesFM zero-shot against the Q10/Q90-only LoRA model "
            "on the WIND_AB 2014-2017 Test set."
        )
    )

    parser.add_argument(
        "--data_path",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )

    parser.add_argument(
        "--adapter_dir",
        type=Path,
        default=Path("outputs") / "timesfm25_wind_ab_q10_q90_only",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs") / "test_wind_ab_q10_q90_only",
    )

    return parser.parse_args()


def get_timesfm_config(model):
    """Return TimesFM config for a base or PEFT-wrapped model."""
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "quantiles"):
        return config

    if hasattr(model, "get_base_model"):
        config = getattr(model.get_base_model(), "config", None)
        if config is not None and hasattr(config, "quantiles"):
            return config

    raise RuntimeError("Could not recover TimesFM configuration.")


def quantile_channel_map(model) -> dict[float, int]:
    """
    Map TimesFM quantiles to full_predictions channels:
    1->Q10, 2->Q20, ..., 5->Q50, ..., 9->Q90.
    """
    config = get_timesfm_config(model)
    return {
        float(q): index + 1
        for index, q in enumerate(config.quantiles)
    }


def pinball(
    target: np.ndarray,
    prediction: np.ndarray,
    quantile: float,
) -> float:
    """Pinball loss for one quantile."""
    error = target - prediction
    return float(
        np.mean(
            np.maximum(
                quantile * error,
                (quantile - 1.0) * error,
            )
        )
    )


def coverage(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Empirical coverage of the Q10-Q90 interval."""
    return float(
        np.mean(
            (target >= lower)
            & (target <= upper)
        )
    )


def interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Mean width of the Q10-Q90 interval."""
    return float(np.mean(upper - lower))


def crossing_rate(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of points for which Q10 > Q90."""
    return float(np.mean(lower > upper))


@torch.no_grad()
def forecast_interval(
    model,
    context: torch.Tensor,
    forecast: ForecastConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Generate the Q10 and Q90 forecasts used in this experiment."""
    input_values = context.to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    outputs = model(
        past_values=input_values,
        forecast_context_len=forecast.model_context_len,
    )

    full_predictions = (
        outputs.full_predictions[0, :forecast.horizon_len, :]
        .float()
        .cpu()
        .numpy()
    )

    channel_map = quantile_channel_map(model)

    if (
        channel_map.get(Q10) != Q10_CHANNEL
        or channel_map.get(Q90) != Q90_CHANNEL
    ):
        raise RuntimeError("Unexpected TimesFM Q10/Q90 channel mapping.")

    return {
        "q10": full_predictions[:, Q10_CHANNEL],
        "q90": full_predictions[:, Q90_CHANNEL],
    }


def evaluate_model(
    model,
    dataset,
    model_name: str,
    forecast: ForecastConfig,
    device: torch.device,
):
    """Evaluate Q10/Q90 accuracy and calibration over all Test origins."""
    rows = []
    predictions = {}

    all_targets = []
    all_q10 = []
    all_q90 = []

    for index in range(len(dataset)):
        context, target = dataset[index]
        metadata = dataset.window_metadata(index)

        target_np = target.numpy()

        prediction = forecast_interval(
            model=model,
            context=context,
            forecast=forecast,
            device=device,
        )

        predictions[index] = prediction

        all_targets.append(target_np)
        all_q10.append(prediction["q10"])
        all_q90.append(prediction["q90"])

        pb10 = pinball(target_np, prediction["q10"], Q10)
        pb90 = pinball(target_np, prediction["q90"], Q90)

        rows.append(
            {
                "model": model_name,
                "origin": metadata["origin"].date().isoformat(),
                "year": metadata["year"],
                "month": metadata["month"],
                "month_name": metadata["month_name"],
                "pinball_q10": pb10,
                "pinball_q90": pb90,
                "mean_pinball_q10_q90": 0.5 * (pb10 + pb90),
                "coverage_q10_q90": coverage(
                    target_np,
                    prediction["q10"],
                    prediction["q90"],
                ),
                "width_q10_q90": interval_width(
                    prediction["q10"],
                    prediction["q90"],
                ),
                "crossing_rate_q10_q90": crossing_rate(
                    prediction["q10"],
                    prediction["q90"],
                ),
            }
        )

    target = np.concatenate(all_targets)
    q10 = np.concatenate(all_q10)
    q90 = np.concatenate(all_q90)

    pb10 = pinball(target, q10, Q10)
    pb90 = pinball(target, q90, Q90)

    aggregate = {
        "model": model_name,
        "n_origins": len(dataset),
        "n_forecast_points": len(target),
        "pinball_q10": pb10,
        "pinball_q90": pb90,
        "mean_pinball_q10_q90": 0.5 * (pb10 + pb90),
        "coverage_q10_q90": coverage(target, q10, q90),
        "width_q10_q90": interval_width(q10, q90),
        "crossing_rate_q10_q90": crossing_rate(q10, q90),
    }

    return pd.DataFrame(rows), aggregate, predictions


def load_zero_shot(device: torch.device):
    """Load the untouched pretrained TimesFM 2.5 model."""
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    ).to(device)

    model.eval()
    return model


def load_fine_tuned(
    adapter_dir: Path,
    device: torch.device,
):
    """Load TimesFM 2.5 with the Q10/Q90-only LoRA adapter."""
    base_model = TimesFm2_5ModelForPrediction.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    ).to(device)

    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
    ).to(device)

    model.eval()
    return model


def plot_forecast(
    dataset,
    index: int,
    zero_shot_prediction: dict[str, np.ndarray],
    fine_tuned_prediction: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """
    Plot the contextual month and the 28-day Test forecast.

    The figure contains:
      - observed contextual period before the forecast origin,
      - ground truth over the 28-day forecast horizon,
      - zero-shot Q10-Q90 shaded interval,
      - fine-tuned Q10-Q90 shaded interval.

    Q10 and Q90 are intentionally not plotted as separate boundary lines.
    """
    context, target = dataset[index]
    metadata = dataset.window_metadata(index)

    context_dates = pd.date_range(
        metadata["context_start"],
        metadata["context_end"],
        freq="D",
    )

    target_dates = pd.date_range(
        metadata["target_start"],
        metadata["target_end"],
        freq="D",
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    # Observed contextual month used as input by TimesFM.
    ax.plot(
        context_dates,
        context.numpy(),
        linewidth=1.5,
        label="Observed context",
    )

    # Ground truth over the forecast horizon.
    ax.plot(
        target_dates,
        target.numpy(),
        linewidth=2.0,
        label="Ground truth",
    )

    # Zero-shot 80% nominal prediction interval [Q10, Q90].
    ax.fill_between(
        target_dates,
        zero_shot_prediction["q10"],
        zero_shot_prediction["q90"],
        alpha=0.25,
        label="Zero-shot Q10-Q90",
    )

    # Fine-tuned 80% nominal prediction interval [Q10, Q90].
    ax.fill_between(
        target_dates,
        fine_tuned_prediction["q10"],
        fine_tuned_prediction["q90"],
        alpha=0.25,
        label="Fine-tuned Q10-Q90",
    )

    # Forecast origin: everything to the left is context.
    ax.axvline(
        target_dates[0],
        linestyle="--",
        linewidth=1.0,
        label="Forecast origin",
    )

    ax.set_title(
        f"WIND_AB Q10-Q90 — origin {metadata['origin'].date()}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily mean WIND_AB")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if not args.adapter_dir.exists():
        raise FileNotFoundError(
            f"Fine-tuned adapter not found: {args.adapter_dir}"
        )

    device = torch.device("cuda")

    # Fixed Test configuration.
    split = SplitConfig(
        development_start_year=1980,
        development_end_year=2013,
        test_start_year=2014,
        test_end_year=2017,
        validation_start_month=11,
    )

    forecast = ForecastConfig(
        context_len=32,
        horizon_len=28,
        model_context_len=32,
    )

    dataframe = load_wind_dataframe(args.data_path)

    test_dataset = WindMonthlyTestDataset(
        df=dataframe,
        split=split,
        forecast=forecast,
    )

    if len(test_dataset) != 48:
        raise RuntimeError(
            f"Expected 48 Test origins, got {len(test_dataset)}."
        )

    print("\n=== TEST CONFIGURATION ===")
    print("Variable: WIND_AB daily mean")
    print("Test period: 2014-2017")
    print("Context: 32 days")
    print("Horizon: 28 days")
    print(f"Origins: {len(test_dataset)}")

    # Final comparison: only zero-shot and Q10/Q90 fine-tuned TimesFM.
    print(f"\nLoading zero-shot model: {MODEL_ID}")
    zero_shot = load_zero_shot(device)

    print(f"Loading fine-tuned adapter: {args.adapter_dir}")
    fine_tuned = load_fine_tuned(
        adapter_dir=args.adapter_dir,
        device=device,
    )

    channel_map = quantile_channel_map(zero_shot)
    print(f"Quantile channel map: {channel_map}")

    if (
        channel_map.get(Q10) != Q10_CHANNEL
        or channel_map.get(Q90) != Q90_CHANNEL
    ):
        raise RuntimeError("Unexpected TimesFM Q10/Q90 channel mapping.")

    zero_rows, zero_summary, zero_predictions = evaluate_model(
        model=zero_shot,
        dataset=test_dataset,
        model_name="zero_shot",
        forecast=forecast,
        device=device,
    )

    ft_rows, ft_summary, ft_predictions = evaluate_model(
        model=fine_tuned,
        dataset=test_dataset,
        model_name="q10_q90_ft",
        forecast=forecast,
        device=device,
    )

    # Save aggregate, per-origin, and monthly metrics.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "monthly_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_origin = pd.concat(
        [zero_rows, ft_rows],
        ignore_index=True,
    )
    per_origin.to_csv(
        args.output_dir / "metrics_per_origin.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [zero_summary, ft_summary]
    )

    # Quantify the benefit of fine-tuning relative to zero-shot.
    # Positive Pinball improvement means lower (better) Pinball Loss.
    zero_pb10 = zero_summary["pinball_q10"]
    zero_pb90 = zero_summary["pinball_q90"]
    zero_pb_mean = zero_summary["mean_pinball_q10_q90"]

    summary["pinball_q10_improvement_vs_zero_shot_pct"] = [
        0.0,
        100.0 * (zero_pb10 - ft_summary["pinball_q10"]) / zero_pb10,
    ]
    summary["pinball_q90_improvement_vs_zero_shot_pct"] = [
        0.0,
        100.0 * (zero_pb90 - ft_summary["pinball_q90"]) / zero_pb90,
    ]
    summary["mean_pinball_improvement_vs_zero_shot_pct"] = [
        0.0,
        100.0
        * (zero_pb_mean - ft_summary["mean_pinball_q10_q90"])
        / zero_pb_mean,
    ]

    # Coverage and interval width are reported as changes, not "improvements",
    # because lower width is desirable only if calibration remains acceptable.
    summary["coverage_change_vs_zero_shot_pp"] = [
        0.0,
        100.0
        * (
            ft_summary["coverage_q10_q90"]
            - zero_summary["coverage_q10_q90"]
        ),
    ]
    summary["width_change_vs_zero_shot_pct"] = [
        0.0,
        100.0
        * (
            ft_summary["width_q10_q90"]
            - zero_summary["width_q10_q90"]
        )
        / zero_summary["width_q10_q90"],
    ]

    summary.to_csv(
        args.output_dir / "metrics_summary.csv",
        index=False,
    )

    metrics = [
        "pinball_q10",
        "pinball_q90",
        "mean_pinball_q10_q90",
        "coverage_q10_q90",
        "width_q10_q90",
        "crossing_rate_q10_q90",
    ]

    by_month = (
        per_origin
        .groupby(
            ["model", "month", "month_name"],
            as_index=False,
        )[metrics]
        .mean()
        .sort_values(["month", "model"])
    )
    by_month.to_csv(
        args.output_dir / "metrics_by_month.csv",
        index=False,
    )

    # Generate one figure for each monthly Test origin.
    for index in range(len(test_dataset)):
        metadata = test_dataset.window_metadata(index)

        plot_forecast(
            dataset=test_dataset,
            index=index,
            zero_shot_prediction=zero_predictions[index],
            fine_tuned_prediction=ft_predictions[index],
            output_path=(
                plots_dir
                / f"forecast_{metadata['origin'].date().isoformat()}.png"
            ),
        )

    print("\n=== TEST RESULTS ===")
    print(
        summary[
            [
                "model",
                "pinball_q10",
                "pinball_q90",
                "mean_pinball_q10_q90",
                "coverage_q10_q90",
                "width_q10_q90",
                "crossing_rate_q10_q90",
                "pinball_q10_improvement_vs_zero_shot_pct",
                "pinball_q90_improvement_vs_zero_shot_pct",
                "mean_pinball_improvement_vs_zero_shot_pct",
                "coverage_change_vs_zero_shot_pp",
                "width_change_vs_zero_shot_pct",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
