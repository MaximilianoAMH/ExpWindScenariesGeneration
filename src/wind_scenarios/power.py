from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .data import PROJECT_ROOT


DEFAULT_ALTO_BAGUALES_CURVE = (
    PROJECT_ROOT
    / "data"
    / "power_curves"
    / "alto_baguales_3p5mw.csv"
)

# Defaults taken from the Alto Baguales description supplied for the project.
DEFAULT_TURBINE_MODEL = "Enercon E-138 EP3"
DEFAULT_RATED_POWER_MW = 3.5
DEFAULT_HUB_HEIGHT_M = 120.0
DEFAULT_N_TURBINES = 1.0

# The supplied wind_curve.csv itself reaches 3500 kW.
SUPPLIED_CURVE_RATED_POWER_MW = 3.5


@dataclass(frozen=True)
class PowerCurve:
    """
    Parametric turbine power curve.

    ``power_kw`` is the curve actually used in the calculation. It can be
    rescaled from the source CSV maximum to the requested nominal turbine
    rating.

    ``source_power_kw`` preserves the unscaled values from the CSV so the
    transformation is auditable.
    """

    wind_speed_ms: np.ndarray
    power_kw: np.ndarray
    source_power_kw: np.ndarray
    rated_power_kw: float
    source_rated_power_kw: float
    cut_out_speed_ms: float
    turbine_model: str
    hub_height_m: float
    name: str = "power_curve"

    @property
    def rated_power_mw(self) -> float:
        return self.rated_power_kw / 1000.0

    @property
    def source_rated_power_mw(self) -> float:
        return self.source_rated_power_kw / 1000.0

    @property
    def scale_factor(self) -> float:
        return self.rated_power_kw / self.source_rated_power_kw


def load_power_curve(
    path: str | Path = DEFAULT_ALTO_BAGUALES_CURVE,
    *,
    wind_speed_column: str = "v",
    power_column: str = "P",
    power_unit: Literal["kW", "MW"] = "kW",
    rated_power_mw: float = DEFAULT_RATED_POWER_MW,
    cut_out_speed_ms: float | None = None,
    turbine_model: str = DEFAULT_TURBINE_MODEL,
    hub_height_m: float = DEFAULT_HUB_HEIGHT_M,
    rescale_to_rated_power: bool = True,
    name: str | None = None,
) -> PowerCurve:
    """
    Load a tabulated turbine power curve.

    Alto Baguales defaults
    ----------------------
    turbine_model:
        Enercon E-138 EP3

    rated_power_mw:
        3.5 MW per turbine, matching the maximum of the supplied
        ``wind_curve.csv`` used by the source conversion code.

    hub_height_m:
        120 m, following the supplied Alto Baguales text.

    Source convention
    -----------------
    The supplied ``wind_curve.csv`` reaches 3500 kW. Therefore the default
    Alto Baguales turbine rating is set to 3.5 MW so that the repository uses
    the supplied power curve without changing its maximum.

    ``rescale_to_rated_power`` is retained only as an optional parametric
    feature for future experiments with a different nominal rating.
    """
    path = Path(path)
    frame = pd.read_csv(path)

    required = {
        wind_speed_column,
        power_column,
    }
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}."
        )

    frame = frame.loc[
        :,
        [wind_speed_column, power_column],
    ].copy()

    if frame.empty:
        raise ValueError(
            f"{path} contains no power-curve data."
        )

    if frame.isna().any().any():
        raise ValueError(
            f"{path} contains missing values."
        )

    wind = frame[wind_speed_column].to_numpy(dtype=float)
    source_power = frame[power_column].to_numpy(dtype=float)

    if not np.isfinite(wind).all() or not np.isfinite(source_power).all():
        raise ValueError(
            "Power curve contains non-finite values."
        )

    if np.any(wind < 0) or np.any(source_power < 0):
        raise ValueError(
            "Power-curve wind speed and power must be non-negative."
        )

    if np.any(np.diff(wind) <= 0):
        raise ValueError(
            "Power-curve wind speeds must be strictly increasing."
        )

    if power_unit == "MW":
        source_power_kw = source_power * 1000.0
    elif power_unit == "kW":
        source_power_kw = source_power
    else:
        raise ValueError(
            "power_unit must be 'kW' or 'MW'."
        )

    if rated_power_mw <= 0:
        raise ValueError(
            "rated_power_mw must be positive."
        )

    if hub_height_m <= 0:
        raise ValueError(
            "hub_height_m must be positive."
        )

    source_rated_power_kw = float(
        np.max(source_power_kw)
    )
    rated_power_kw = float(
        rated_power_mw * 1000.0
    )

    if rescale_to_rated_power:
        power_kw = (
            source_power_kw
            * rated_power_kw
            / source_rated_power_kw
        )
    else:
        if not np.isclose(
            source_rated_power_kw,
            rated_power_kw,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "Source curve maximum does not match rated_power_mw. "
                f"Source={source_rated_power_kw / 1000.0:.3f} MW, "
                f"requested={rated_power_mw:.3f} MW. "
                "Enable rescale_to_rated_power to rescale the curve."
            )
        power_kw = source_power_kw.copy()

    if cut_out_speed_ms is None:
        cut_out_speed_ms = float(wind[-1])
    elif cut_out_speed_ms <= 0:
        raise ValueError(
            "cut_out_speed_ms must be positive."
        )

    return PowerCurve(
        wind_speed_ms=wind,
        power_kw=power_kw,
        source_power_kw=source_power_kw,
        rated_power_kw=rated_power_kw,
        source_rated_power_kw=source_rated_power_kw,
        cut_out_speed_ms=float(cut_out_speed_ms),
        turbine_model=turbine_model,
        hub_height_m=float(hub_height_m),
        name=name if name is not None else path.stem,
    )


def turbine_power_from_wind_speed(
    wind_speed_ms: np.ndarray | pd.Series | list[float],
    curve: PowerCurve,
    *,
    output_unit: Literal["kW", "MW"] = "MW",
    below_curve_policy: Literal["zero", "first"] = "zero",
    above_cutout_power_kw: float = 0.0,
) -> np.ndarray:
    """
    Convert wind speed [m/s] to electrical power for ONE turbine.

    The mathematical model is piecewise linear. For each interval
    [v_k, v_{k+1}]:

        P(v) = h0_k + h_k v

    where:

        h_k  = (P_{k+1} - P_k) / (v_{k+1} - v_k)
        h0_k = P_k - h_k v_k

    This is equivalent to linear interpolation between consecutive points of
    the tabulated curve.

    Wind speed is assumed to represent the turbine hub height specified in the
    ``PowerCurve`` metadata. No vertical wind-profile correction is applied in
    this module.
    """
    speed = np.asarray(
        wind_speed_ms,
        dtype=float,
    )

    if not np.isfinite(speed).all():
        raise ValueError(
            "Wind speed contains non-finite values."
        )

    if np.any(speed < 0):
        raise ValueError(
            "Wind speed must be non-negative."
        )

    if below_curve_policy not in {"zero", "first"}:
        raise ValueError(
            "below_curve_policy must be 'zero' or 'first'."
        )

    wind = curve.wind_speed_ms
    power_kw = curve.power_kw

    result_kw = np.interp(
        speed,
        wind,
        power_kw,
    )

    if below_curve_policy == "zero":
        result_kw = np.where(
            speed < wind[0],
            0.0,
            result_kw,
        )
    else:
        result_kw = np.where(
            speed < wind[0],
            power_kw[0],
            result_kw,
        )

    result_kw = np.where(
        speed > curve.cut_out_speed_ms,
        float(above_cutout_power_kw),
        result_kw,
    )

    if output_unit == "kW":
        return result_kw

    if output_unit == "MW":
        return result_kw / 1000.0

    raise ValueError(
        "output_unit must be 'kW' or 'MW'."
    )


def plant_power_from_wind_speed(
    wind_speed_ms: np.ndarray | pd.Series | list[float],
    curve: PowerCurve,
    *,
    n_turbines: float = DEFAULT_N_TURBINES,
    availability: float = 1.0,
    electrical_efficiency: float = 1.0,
    output_unit: Literal["kW", "MW"] = "MW",
) -> np.ndarray:
    """
    Convert wind speed to aggregate plant power:

        P_plant(v) =
            P_turbine(v)
            * n_turbines
            * availability
            * electrical_efficiency

    No Alto Baguales turbine count is inferred from the project-capacity text;
    the default therefore remains one turbine until the plant configuration is
    explicitly specified.
    """
    if n_turbines <= 0:
        raise ValueError(
            "n_turbines must be positive."
        )

    if not (0.0 <= availability <= 1.0):
        raise ValueError(
            "availability must be between 0 and 1."
        )

    if not (0.0 <= electrical_efficiency <= 1.0):
        raise ValueError(
            "electrical_efficiency must be between 0 and 1."
        )

    turbine_kw = turbine_power_from_wind_speed(
        wind_speed_ms,
        curve,
        output_unit="kW",
    )

    plant_kw = (
        turbine_kw
        * float(n_turbines)
        * float(availability)
        * float(electrical_efficiency)
    )

    if output_unit == "kW":
        return plant_kw

    if output_unit == "MW":
        return plant_kw / 1000.0

    raise ValueError(
        "output_unit must be 'kW' or 'MW'."
    )


def save_power_scenarios(
    wind_scenarios: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    output_dir: str | Path,
    *,
    curve_path: str | Path = DEFAULT_ALTO_BAGUALES_CURVE,
    curve_power_unit: Literal["kW", "MW"] = "kW",
    rated_power_mw: float = DEFAULT_RATED_POWER_MW,
    cut_out_speed_ms: float | None = None,
    turbine_model: str = DEFAULT_TURBINE_MODEL,
    hub_height_m: float = DEFAULT_HUB_HEIGHT_M,
    rescale_to_rated_power: bool = True,
    n_turbines: float = DEFAULT_N_TURBINES,
    availability: float = 1.0,
    electrical_efficiency: float = 1.0,
    output_unit: Literal["kW", "MW"] = "MW",
    filename: str = "power_scenarios_alto_baguales.csv",
) -> Path:
    """
    Convert final L x H Schaake wind scenarios to Alto Baguales power scenarios.
    """
    wind_scenarios = np.asarray(
        wind_scenarios,
        dtype=float,
    )

    if wind_scenarios.ndim != 2:
        raise ValueError(
            "wind_scenarios must have shape (L, H)."
        )

    forecast_dates = pd.DatetimeIndex(
        forecast_dates
    )

    L, H = wind_scenarios.shape

    if len(forecast_dates) != H:
        raise ValueError(
            "forecast_dates length must match scenario horizon H."
        )

    curve = load_power_curve(
        curve_path,
        power_unit=curve_power_unit,
        rated_power_mw=rated_power_mw,
        cut_out_speed_ms=cut_out_speed_ms,
        turbine_model=turbine_model,
        hub_height_m=hub_height_m,
        rescale_to_rated_power=rescale_to_rated_power,
    )

    power = plant_power_from_wind_speed(
        wind_scenarios,
        curve,
        n_turbines=n_turbines,
        availability=availability,
        electrical_efficiency=electrical_efficiency,
        output_unit=output_unit,
    )

    scenario_ids = [
        f"scenario_{i + 1:03d}"
        for i in range(L)
    ]

    columns = [
        date.date().isoformat()
        for date in forecast_dates
    ]

    frame = pd.DataFrame(
        power,
        index=scenario_ids,
        columns=columns,
    )
    frame.index.name = "scenario_id"

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / filename

    frame.to_csv(
        output_path
    )

    return output_path
