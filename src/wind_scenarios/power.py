from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALTO_BAGUALES_CURVE = (
    PROJECT_ROOT / "data" / "power_curves" / "alto_baguales_3p5mw.csv"
)

DEFAULT_TURBINE_MODEL = "Enercon E-138 EP3"
DEFAULT_RATED_POWER_MW = 3.5
DEFAULT_HUB_HEIGHT_M = 120.0
DEFAULT_N_TURBINES = 1.0


@dataclass(frozen=True)
class PowerCurve:
    """Tabulated turbine power curve used for wind-to-power conversion."""

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


@dataclass(frozen=True)
class PlantPowerConfig:
    """Plant-level parameters applied after the single-turbine power curve."""

    n_turbines: float = DEFAULT_N_TURBINES
    availability: float = 1.0
    electrical_efficiency: float = 1.0

    def validate(self) -> None:
        if self.n_turbines <= 0:
            raise ValueError("n_turbines must be positive.")
        if not (0.0 <= self.availability <= 1.0):
            raise ValueError("availability must be between 0 and 1.")
        if not (0.0 <= self.electrical_efficiency <= 1.0):
            raise ValueError("electrical_efficiency must be between 0 and 1.")


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
    """Load and validate a tabulated turbine power curve."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Power-curve CSV not found: {path}")

    frame = pd.read_csv(path)
    required = {wind_speed_column, power_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}.")

    frame = frame[[wind_speed_column, power_column]].copy()
    if frame.empty:
        raise ValueError(f"{path} contains no power-curve data.")
    if frame.isna().any().any():
        raise ValueError(f"{path} contains missing values.")

    wind = pd.to_numeric(frame[wind_speed_column], errors="raise").to_numpy(dtype=float)
    source_power = pd.to_numeric(frame[power_column], errors="raise").to_numpy(dtype=float)

    if not np.isfinite(wind).all() or not np.isfinite(source_power).all():
        raise ValueError("Power curve contains non-finite values.")
    if np.any(wind < 0) or np.any(source_power < 0):
        raise ValueError("Power-curve wind speed and power must be non-negative.")
    if np.any(np.diff(wind) <= 0):
        raise ValueError("Power-curve wind speeds must be strictly increasing.")

    if power_unit == "kW":
        source_power_kw = source_power
    elif power_unit == "MW":
        source_power_kw = source_power * 1000.0
    else:
        raise ValueError("power_unit must be 'kW' or 'MW'.")

    if rated_power_mw <= 0:
        raise ValueError("rated_power_mw must be positive.")
    if hub_height_m <= 0:
        raise ValueError("hub_height_m must be positive.")

    source_rated_power_kw = float(np.max(source_power_kw))
    if source_rated_power_kw <= 0:
        raise ValueError("The source power curve must contain positive power values.")

    rated_power_kw = float(rated_power_mw * 1000.0)

    if rescale_to_rated_power:
        power_kw = source_power_kw * rated_power_kw / source_rated_power_kw
    else:
        if not np.isclose(source_rated_power_kw, rated_power_kw, rtol=0.0, atol=1e-6):
            raise ValueError(
                "Source curve maximum does not match rated_power_mw. "
                f"Source={source_rated_power_kw / 1000.0:.3f} MW, "
                f"requested={rated_power_mw:.3f} MW."
            )
        power_kw = source_power_kw.copy()

    if cut_out_speed_ms is None:
        cut_out_speed_ms = float(wind[-1])
    elif cut_out_speed_ms <= 0:
        raise ValueError("cut_out_speed_ms must be positive.")

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
    """Convert wind speed to single-turbine power by piecewise-linear interpolation."""
    speed = np.asarray(wind_speed_ms, dtype=float)

    if not np.isfinite(speed).all():
        raise ValueError("Wind speed contains non-finite values.")
    if np.any(speed < 0):
        raise ValueError("Wind speed must be non-negative.")
    if below_curve_policy not in {"zero", "first"}:
        raise ValueError("below_curve_policy must be 'zero' or 'first'.")

    result_kw = np.interp(speed, curve.wind_speed_ms, curve.power_kw)

    if below_curve_policy == "zero":
        result_kw = np.where(speed < curve.wind_speed_ms[0], 0.0, result_kw)
    else:
        result_kw = np.where(speed < curve.wind_speed_ms[0], curve.power_kw[0], result_kw)

    result_kw = np.where(
        speed > curve.cut_out_speed_ms,
        float(above_cutout_power_kw),
        result_kw,
    )

    if output_unit == "kW":
        return result_kw
    if output_unit == "MW":
        return result_kw / 1000.0
    raise ValueError("output_unit must be 'kW' or 'MW'.")


def plant_power_from_wind_speed(
    wind_speed_ms: np.ndarray | pd.Series | list[float],
    curve: PowerCurve,
    *,
    n_turbines: float = DEFAULT_N_TURBINES,
    availability: float = 1.0,
    electrical_efficiency: float = 1.0,
    output_unit: Literal["kW", "MW"] = "MW",
) -> np.ndarray:
    """Convert wind speed to aggregate plant power."""
    config = PlantPowerConfig(
        n_turbines=n_turbines,
        availability=availability,
        electrical_efficiency=electrical_efficiency,
    )
    config.validate()

    turbine_kw = turbine_power_from_wind_speed(
        wind_speed_ms,
        curve,
        output_unit="kW",
    )
    plant_kw = (
        turbine_kw
        * config.n_turbines
        * config.availability
        * config.electrical_efficiency
    )

    if output_unit == "kW":
        return plant_kw
    if output_unit == "MW":
        return plant_kw / 1000.0
    raise ValueError("output_unit must be 'kW' or 'MW'.")


def effective_capacity_mw(
    curve: PowerCurve,
    *,
    n_turbines: float = DEFAULT_N_TURBINES,
    availability: float = 1.0,
    electrical_efficiency: float = 1.0,
) -> float:
    """Maximum plant power implied by the configured curve and plant factors."""
    config = PlantPowerConfig(
        n_turbines=n_turbines,
        availability=availability,
        electrical_efficiency=electrical_efficiency,
    )
    config.validate()
    return float(
        curve.rated_power_mw
        * config.n_turbines
        * config.availability
        * config.electrical_efficiency
    )
