from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data import DEFAULT_DATA_PATH, load_wind_dataframe
from ..power import (
    DEFAULT_ALTO_BAGUALES_CURVE,
    DEFAULT_RATED_POWER_MW,
    load_power_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize zero-shot TimesFM + Schaake Shuffle outputs. "
            "The TimesFM input context is always plotted."
        )
    )

    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--mode",
        choices=[
            "fan",
            "single",
            "wind_power",
            "power_curve",
        ],
        default="single",
        help=(
            "fan: all wind scenarios; "
            "single: one wind scenario; "
            "wind_power: generated wind and calculated power vs date; "
            "power_curve: generated wind-power pairs over the turbine curve."
        ),
    )
    parser.add_argument("--scenario_id", type=str, default=None)
    parser.add_argument("--scenario_number", type=int, default=None)
    parser.add_argument(
        "--all_quantiles",
        action="store_true",
        help="Plot Q10,...,Q90 instead of only Q10/Q50/Q90.",
    )
    parser.add_argument(
        "--export_all",
        action="store_true",
        help="In single mode, export one figure per scenario.",
    )
    parser.add_argument(
        "--power_curve",
        type=Path,
        default=DEFAULT_ALTO_BAGUALES_CURVE,
        help="Power-curve CSV used by power_curve mode.",
    )
    parser.add_argument(
        "--rated_power_mw",
        type=float,
        default=DEFAULT_RATED_POWER_MW,
        help=(
            "Turbine rated power used to load the curve. "
            f"Default: {DEFAULT_RATED_POWER_MW} MW."
        ),
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_outputs(input_dir: Path) -> dict[str, pd.DataFrame | None]:
    required = {
        "scenarios": input_dir / "scenarios.csv",
        "quantiles": input_dir / "timesfm_quantiles.csv",
        "historical": input_dir / "historical_trajectories.csv",
        "metadata": input_dir / "historical_metadata.csv",
    }

    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    context_path = input_dir / "context.csv"
    power_path = (
        input_dir
        / "power_scenarios_alto_baguales.csv"
    )

    return {
        "scenarios": pd.read_csv(
            required["scenarios"],
            index_col="scenario_id",
        ),
        "quantiles": pd.read_csv(
            required["quantiles"],
            parse_dates=["DATE"],
        ),
        "historical": pd.read_csv(required["historical"]),
        "metadata": pd.read_csv(
            required["metadata"],
            parse_dates=["anchor_date", "start_date", "end_date"],
        ),
        "context": (
            pd.read_csv(context_path, parse_dates=["DATE"])
            if context_path.exists()
            else None
        ),
        "power_scenarios": (
            pd.read_csv(
                power_path,
                index_col="scenario_id",
            )
            if power_path.exists()
            else None
        ),
    }


def recover_context(
    data_path: Path,
    origin: pd.Timestamp,
    context_len: int = 32,
) -> pd.DataFrame:
    df = load_wind_dataframe(data_path)

    dates = pd.date_range(
        end=origin - pd.Timedelta(days=1),
        periods=context_len,
        freq="D",
    )

    indexed = (
        df.loc[:, ["DATE", "WIND"]]
        .copy()
        .assign(DATE=lambda x: pd.to_datetime(x["DATE"]).dt.normalize())
        .set_index("DATE")
    )

    if not dates.isin(indexed.index).all():
        missing = dates[~dates.isin(indexed.index)]
        raise ValueError(
            "Could not recover complete TimesFM context. Missing: "
            f"{[d.date().isoformat() for d in missing]}"
        )

    return pd.DataFrame(
        {
            "DATE": dates,
            "WIND": indexed.loc[dates, "WIND"].to_numpy(dtype=float),
        }
    )


def load_observed(
    data_path: Path,
    forecast_dates: pd.DatetimeIndex,
) -> np.ndarray:
    df = load_wind_dataframe(data_path)

    indexed = (
        df.loc[:, ["DATE", "WIND"]]
        .copy()
        .assign(DATE=lambda x: pd.to_datetime(x["DATE"]).dt.normalize())
        .set_index("DATE")
    )

    if not forecast_dates.isin(indexed.index).all():
        missing = forecast_dates[~forecast_dates.isin(indexed.index)]
        raise ValueError(
            "Observed target is incomplete. Missing: "
            f"{[d.date().isoformat() for d in missing]}"
        )

    return indexed.loc[
        forecast_dates,
        "WIND",
    ].to_numpy(dtype=float)


def normalize_scenario_id(
    scenarios: pd.DataFrame,
    scenario_id: str | None,
    scenario_number: int | None,
) -> str:
    if scenario_id is not None and scenario_number is not None:
        raise ValueError("Use only --scenario_id or --scenario_number.")

    if scenario_number is not None:
        if scenario_number <= 0:
            raise ValueError("--scenario_number must be positive.")
        scenario_id = f"scenario_{scenario_number:03d}"

    if scenario_id is None:
        scenario_id = str(scenarios.index[0])

    if scenario_id not in scenarios.index:
        raise ValueError(f"Scenario {scenario_id} not found.")

    return scenario_id


def plot_quantiles(
    ax,
    quantiles: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    all_quantiles: bool,
) -> None:
    if all_quantiles:
        columns = sorted(
            [
                c
                for c in quantiles.columns
                if c.startswith("q") and c[1:].isdigit()
            ],
            key=lambda c: int(c[1:]),
        )
    else:
        columns = ["q10", "q50", "q90"]

    for column in columns:
        if column not in quantiles.columns:
            raise ValueError(f"Missing quantile column: {column}")

        level = int(column[1:])
        linewidth = 1.2
        linestyle = "-"
        alpha = 0.70

        if level in {10, 90}:
            linewidth = 1.8
            linestyle = "--"
            alpha = 0.90
        elif level == 50:
            linewidth = 2.4
            alpha = 1.00

        ax.plot(
            forecast_dates,
            quantiles[column].to_numpy(dtype=float),
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            label=f"TimesFM {column.upper()}",
        )


def annotate_forecast_origin(
    ax,
    origin: pd.Timestamp,
) -> None:
    """
    Draw the forecast-origin line without adding it to the legend.

    The meaning of the vertical dashed line is communicated directly
    through an annotation placed next to the line.
    """
    ax.axvline(
        origin,
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        label="_nolegend_",
    )

    ax.annotate(
        f"Forecast origin\n{origin.date()}",
        xy=(origin, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(6, -10),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "none",
        },
    )


def place_legend_lower_left(ax) -> None:
    """
    Place the legend inside the plotting area, in the lower-left corner.

    This restores the original compact layout and avoids shrinking the
    plotting area horizontally. The forecast-origin annotation remains
    separate from the legend.
    """
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.01, 0.02),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.90,
        fontsize=9,
    )


def plot_fan(
    scenarios: pd.DataFrame,
    quantiles: pd.DataFrame,
    context: pd.DataFrame,
    observed: np.ndarray,
    output_path: Path,
    all_quantiles: bool,
    show: bool,
) -> None:
    forecast_dates = pd.DatetimeIndex(quantiles["DATE"])
    origin = forecast_dates[0]

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(
        context["DATE"],
        context["WIND"],
        linewidth=2.5,
        label="Observed context (TimesFM input)",
    )

    for row in scenarios.to_numpy(dtype=float):
        ax.plot(
            forecast_dates,
            row,
            linewidth=0.8,
            alpha=0.25,
        )

    plot_quantiles(
        ax=ax,
        quantiles=quantiles,
        forecast_dates=forecast_dates,
        all_quantiles=all_quantiles,
    )

    ax.plot(
        forecast_dates,
        observed,
        linewidth=2.6,
        label="Wind Ground Truth",
    )

    annotate_forecast_origin(
        ax=ax,
        origin=origin,
    )

    ax.set_title(
        "Zero-shot TimesFM + Schaake Shuffle "
        f"— origin {origin.date()} — L={len(scenarios)}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily mean WIND")
    ax.grid(alpha=0.25)

    place_legend_lower_left(ax)

    fig.autofmt_xdate()

    # The legend is inside the axes, so the full horizontal area can be used.
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.18,
        top=0.90,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(fig)


def plot_single(
    scenario_id: str,
    scenarios: pd.DataFrame,
    quantiles: pd.DataFrame,
    historical: pd.DataFrame,
    metadata: pd.DataFrame,
    context: pd.DataFrame,
    observed: np.ndarray,
    output_path: Path,
    all_quantiles: bool,
    show: bool,
) -> None:
    forecast_dates = pd.DatetimeIndex(quantiles["DATE"])
    origin = forecast_dates[0]

    hist_id = f"hist_{scenario_id.split('_')[-1]}"

    historical_row = historical.loc[
        historical["trajectory_id"] == hist_id
    ]
    if historical_row.empty:
        raise ValueError(f"Historical trajectory {hist_id} not found.")

    hist_values = historical_row.iloc[0][
        [c for c in historical.columns if c != "trajectory_id"]
    ].to_numpy(dtype=float)

    meta_row = metadata.loc[
        metadata["trajectory_id"] == hist_id
    ]
    if meta_row.empty:
        raise ValueError(f"Historical metadata {hist_id} not found.")
    meta_row = meta_row.iloc[0]

    scenario_values = scenarios.loc[
        scenario_id
    ].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(
        context["DATE"],
        context["WIND"],
        linewidth=2.5,
        label="Observed context (TimesFM input)",
    )

    plot_quantiles(
        ax=ax,
        quantiles=quantiles,
        forecast_dates=forecast_dates,
        all_quantiles=all_quantiles,
    )

    ax.plot(
        forecast_dates,
        scenario_values,
        linewidth=2.8,
        label=f"{scenario_id} (generated)",
    )

    ax.plot(
        forecast_dates,
        hist_values,
        linewidth=1.8,
        linestyle=":",
        label=f"{hist_id} (source historical trajectory)",
    )

    ax.plot(
        forecast_dates,
        observed,
        linewidth=2.6,
        label="Wind Ground Truth",
    )

    annotate_forecast_origin(
        ax=ax,
        origin=origin,
    )

    quantile_label = "Q10-Q90" if all_quantiles else "Q10/Q50/Q90"

    ax.set_title(
        "Zero-shot TimesFM + Schaake Shuffle "
        f"— {scenario_id} — {quantile_label} — source {hist_id}",
        pad=24,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily mean WIND")
    ax.grid(alpha=0.25)

    # Keep the date metadata separated from the legend and from the
    # forecast-origin annotation.
    subtitle = (
        f"TimesFM context: "
        f"{pd.Timestamp(context['DATE'].iloc[0]).date()} -> "
        f"{pd.Timestamp(context['DATE'].iloc[-1]).date()} | "
        f"Historical source: "
        f"{pd.Timestamp(meta_row['start_date']).date()} -> "
        f"{pd.Timestamp(meta_row['end_date']).date()}"
    )

    ax.text(
        0.0,
        1.01,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
    )

    place_legend_lower_left(ax)

    fig.autofmt_xdate()

    # Keep space only for the title/subtitle; no right-side reservation
    # is needed because the legend is inside the axes.
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.18,
        top=0.84,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(fig)



def require_power_scenarios(
    power_scenarios: pd.DataFrame | None,
    input_dir: Path,
) -> pd.DataFrame:
    """
    Ensure that the post-processed Alto Baguales power scenarios exist.
    """
    if power_scenarios is None:
        expected = (
            input_dir
            / "power_scenarios_alto_baguales.csv"
        )
        raise FileNotFoundError(
            "Power scenarios were not found. Expected: "
            f"{expected}. Run the updated generation pipeline first."
        )

    return power_scenarios


def validate_wind_power_alignment(
    scenario_id: str,
    scenarios: pd.DataFrame,
    power_scenarios: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return aligned wind and power vectors for one scenario.
    """
    if scenario_id not in power_scenarios.index:
        raise ValueError(
            f"{scenario_id} is missing from power scenarios."
        )

    wind_values = scenarios.loc[
        scenario_id
    ].to_numpy(dtype=float)

    power_values = power_scenarios.loc[
        scenario_id
    ].to_numpy(dtype=float)

    H = len(forecast_dates)

    if len(wind_values) != H:
        raise ValueError(
            "Wind scenario length does not match forecast dates."
        )

    if len(power_values) != H:
        raise ValueError(
            "Power scenario length does not match forecast dates."
        )

    return wind_values, power_values


def plot_wind_power(
    scenario_id: str,
    scenarios: pd.DataFrame,
    power_scenarios: pd.DataFrame,
    quantiles: pd.DataFrame,
    output_path: Path,
    show: bool,
) -> None:
    """
    Plot one generated wind trajectory and its calculated power trajectory.

    Both curves share the same forecast dates but use separate y-axes because
    wind speed [m/s] and electrical power [MW] are different physical units.
    """
    forecast_dates = pd.DatetimeIndex(
        quantiles["DATE"]
    )

    wind_values, power_values = (
        validate_wind_power_alignment(
            scenario_id=scenario_id,
            scenarios=scenarios,
            power_scenarios=power_scenarios,
            forecast_dates=forecast_dates,
        )
    )

    fig, ax_wind = plt.subplots(
        figsize=(15, 7)
    )
    ax_power = ax_wind.twinx()

    wind_line = ax_wind.plot(
        forecast_dates,
        wind_values,
        linewidth=2.6,
        label=f"{scenario_id} — generated wind",
    )[0]

    power_line = ax_power.plot(
        forecast_dates,
        power_values,
        linewidth=2.6,
        linestyle="--",
        label=(
            f"{scenario_id} — calculated power "
            "[Alto Baguales]"
        ),
    )[0]

    ax_wind.set_title(
        "Generated wind scenario vs calculated power "
        f"— {scenario_id}"
    )
    ax_wind.set_xlabel("Date")
    ax_wind.set_ylabel("Wind speed [m/s]")
    ax_power.set_ylabel("Power [MW]")

    ax_wind.set_ylim(bottom=0.0)
    ax_power.set_ylim(bottom=0.0)

    ax_wind.grid(alpha=0.25)

    ax_wind.legend(
        handles=[
            wind_line,
            power_line,
        ],
        loc="lower left",
        bbox_to_anchor=(0.01, 0.02),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.90,
        fontsize=9,
    )

    fig.autofmt_xdate()
    fig.subplots_adjust(
        left=0.08,
        right=0.90,
        bottom=0.18,
        top=0.90,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(fig)


def plot_power_curve_comparison(
    scenario_id: str,
    scenarios: pd.DataFrame,
    power_scenarios: pd.DataFrame,
    quantiles: pd.DataFrame,
    output_path: Path,
    power_curve_path: Path,
    rated_power_mw: float,
    show: bool,
) -> None:
    """
    Plot the turbine wind-to-power transfer curve and the generated
    scenario's (wind, power) points.

    This is a diagnostic of the deterministic post-processing step:

        wind speed [m/s] -> calculated power [MW]
    """
    forecast_dates = pd.DatetimeIndex(
        quantiles["DATE"]
    )

    wind_values, power_values = (
        validate_wind_power_alignment(
            scenario_id=scenario_id,
            scenarios=scenarios,
            power_scenarios=power_scenarios,
            forecast_dates=forecast_dates,
        )
    )

    curve = load_power_curve(
        power_curve_path,
        rated_power_mw=rated_power_mw,
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.plot(
        curve.wind_speed_ms,
        curve.power_kw / 1000.0,
        linewidth=2.5,
        label=(
            "Alto Baguales tabulated/interpolated "
            "power curve"
        ),
    )

    ax.scatter(
        wind_values,
        power_values,
        s=45,
        alpha=0.75,
        label=f"{scenario_id} forecast points",
    )

    ax.set_title(
        "Generated wind values on the Alto Baguales "
        f"power curve — {scenario_id}"
    )
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Power [MW]")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)

    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.01, 0.02),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.90,
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.12,
        top=0.90,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    args = parse_args()
    outputs = load_outputs(args.input_dir)

    scenarios = outputs["scenarios"]
    quantiles = outputs["quantiles"]
    historical = outputs["historical"]
    metadata = outputs["metadata"]
    power_scenarios = outputs["power_scenarios"]

    assert isinstance(scenarios, pd.DataFrame)
    assert isinstance(quantiles, pd.DataFrame)
    assert isinstance(historical, pd.DataFrame)
    assert isinstance(metadata, pd.DataFrame)

    forecast_dates = pd.DatetimeIndex(quantiles["DATE"])
    origin = forecast_dates[0]

    context = outputs["context"]
    if context is None:
        context = recover_context(
            data_path=args.data_path,
            origin=origin,
        )
    assert isinstance(context, pd.DataFrame)

    observed = load_observed(
        data_path=args.data_path,
        forecast_dates=forecast_dates,
    )

    if args.mode in {
        "wind_power",
        "power_curve",
    }:
        power_scenarios = require_power_scenarios(
            power_scenarios=power_scenarios,
            input_dir=args.input_dir,
        )

        if args.export_all:
            output_dir = args.input_dir / (
                "wind_power_plots"
                if args.mode == "wind_power"
                else "power_curve_plots"
            )

            for sid in scenarios.index:
                if args.mode == "wind_power":
                    plot_wind_power(
                        scenario_id=str(sid),
                        scenarios=scenarios,
                        power_scenarios=power_scenarios,
                        quantiles=quantiles,
                        output_path=(
                            output_dir
                            / f"{sid}_wind_power.png"
                        ),
                        show=False,
                    )
                else:
                    plot_power_curve_comparison(
                        scenario_id=str(sid),
                        scenarios=scenarios,
                        power_scenarios=power_scenarios,
                        quantiles=quantiles,
                        output_path=(
                            output_dir
                            / f"{sid}_power_curve.png"
                        ),
                        power_curve_path=args.power_curve,
                        rated_power_mw=args.rated_power_mw,
                        show=False,
                    )

            print(
                f"Exported {len(scenarios)} figures to: "
                f"{output_dir}"
            )
            return

        scenario_id = normalize_scenario_id(
            scenarios=scenarios,
            scenario_id=args.scenario_id,
            scenario_number=args.scenario_number,
        )

        if args.mode == "wind_power":
            output_path = (
                args.input_dir
                / f"{scenario_id}_wind_power.png"
            )
            plot_wind_power(
                scenario_id=scenario_id,
                scenarios=scenarios,
                power_scenarios=power_scenarios,
                quantiles=quantiles,
                output_path=output_path,
                show=args.show,
            )
        else:
            output_path = (
                args.input_dir
                / f"{scenario_id}_power_curve.png"
            )
            plot_power_curve_comparison(
                scenario_id=scenario_id,
                scenarios=scenarios,
                power_scenarios=power_scenarios,
                quantiles=quantiles,
                output_path=output_path,
                power_curve_path=args.power_curve,
                rated_power_mw=args.rated_power_mw,
                show=args.show,
            )

        print(
            f"Saved figure: {output_path}"
        )
        return

    if args.mode == "fan":
        output_path = args.input_dir / (
            "scenario_fan_all_quantiles.png"
            if args.all_quantiles
            else "scenario_fan.png"
        )
        plot_fan(
            scenarios=scenarios,
            quantiles=quantiles,
            context=context,
            observed=observed,
            output_path=output_path,
            all_quantiles=args.all_quantiles,
            show=args.show,
        )
        print(f"Saved figure: {output_path}")
        return

    if args.export_all:
        output_dir = args.input_dir / (
            "scenario_plots_all_quantiles"
            if args.all_quantiles
            else "scenario_plots"
        )

        for scenario_id in scenarios.index:
            plot_single(
                scenario_id=str(scenario_id),
                scenarios=scenarios,
                quantiles=quantiles,
                historical=historical,
                metadata=metadata,
                context=context,
                observed=observed,
                output_path=output_dir / f"{scenario_id}.png",
                all_quantiles=args.all_quantiles,
                show=False,
            )

        print(f"Exported {len(scenarios)} figures to: {output_dir}")
        return

    scenario_id = normalize_scenario_id(
        scenarios=scenarios,
        scenario_id=args.scenario_id,
        scenario_number=args.scenario_number,
    )

    suffix = "_all_quantiles" if args.all_quantiles else ""
    output_path = args.input_dir / f"{scenario_id}{suffix}.png"

    plot_single(
        scenario_id=scenario_id,
        scenarios=scenarios,
        quantiles=quantiles,
        historical=historical,
        metadata=metadata,
        context=context,
        observed=observed,
        output_path=output_path,
        all_quantiles=args.all_quantiles,
        show=args.show,
    )

    print("Context plotted: YES")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
