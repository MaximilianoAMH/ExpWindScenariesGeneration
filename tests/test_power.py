from __future__ import annotations

import numpy as np
import pandas as pd

from wind_scenarios.power import (
    DEFAULT_HUB_HEIGHT_M,
    DEFAULT_RATED_POWER_MW,
    DEFAULT_TURBINE_MODEL,
    effective_capacity_mw,
    load_power_curve,
    plant_power_from_wind_speed,
    turbine_power_from_wind_speed,
)


def make_curve(tmp_path):
    path = tmp_path / "curve.csv"
    pd.DataFrame(
        {
            "v": [0.0, 5.0, 10.0, 15.0, 28.0],
            "P": [0.0, 500.0, 3125.0, 3500.0, 3500.0],
        }
    ).to_csv(path, index=False)
    return path


def test_alto_baguales_defaults():
    assert DEFAULT_TURBINE_MODEL == "Enercon E-138 EP3"
    assert np.isclose(DEFAULT_RATED_POWER_MW, 3.5)
    assert np.isclose(DEFAULT_HUB_HEIGHT_M, 120.0)


def test_piecewise_linear_power_and_cutout(tmp_path):
    curve = load_power_curve(make_curve(tmp_path))
    power = turbine_power_from_wind_speed(
        np.array([10.0, 28.5]),
        curve,
        output_unit="MW",
    )
    np.testing.assert_allclose(power, [3.125, 0.0])


def test_plant_scaling_and_capacity(tmp_path):
    curve = load_power_curve(make_curve(tmp_path))
    power = plant_power_from_wind_speed(
        np.array([15.0]),
        curve,
        n_turbines=2,
        availability=0.9,
        electrical_efficiency=0.95,
        output_unit="MW",
    )
    capacity = effective_capacity_mw(
        curve,
        n_turbines=2,
        availability=0.9,
        electrical_efficiency=0.95,
    )
    np.testing.assert_allclose(power, [capacity])
    assert np.isclose(capacity, 3.5 * 2 * 0.9 * 0.95)
