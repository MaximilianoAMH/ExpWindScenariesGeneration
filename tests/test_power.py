import numpy as np

from wind_scenarios.power import (
    DEFAULT_HUB_HEIGHT_M,
    DEFAULT_RATED_POWER_MW,
    DEFAULT_TURBINE_MODEL,
    load_power_curve,
    turbine_power_from_wind_speed,
)


def test_alto_baguales_defaults():
    assert DEFAULT_TURBINE_MODEL == "Enercon E-138 EP3"
    assert np.isclose(DEFAULT_RATED_POWER_MW, 3.5)
    assert np.isclose(DEFAULT_HUB_HEIGHT_M, 120.0)


def test_supplied_curve_is_used_without_default_rescaling():
    curve = load_power_curve()

    assert np.isclose(
        curve.source_rated_power_mw,
        3.5,
    )
    assert np.isclose(
        curve.rated_power_mw,
        3.5,
    )
    assert np.isclose(
        curve.scale_factor,
        1.0,
    )
    assert np.isclose(
        curve.power_kw.max(),
        3500.0,
    )


def test_tabulated_power_at_10_ms():
    curve = load_power_curve()

    actual = turbine_power_from_wind_speed(
        np.array([10.0]),
        curve,
        output_unit="MW",
    )

    np.testing.assert_allclose(
        actual,
        [3.125],
        atol=1e-12,
    )


def test_curve_reaches_3p5mw():
    curve = load_power_curve()

    # The source curve reaches 3500 kW at its rated plateau.
    actual = turbine_power_from_wind_speed(
        np.array([14.5]),
        curve,
        output_unit="MW",
    )

    np.testing.assert_allclose(
        actual,
        [3.5],
        atol=1e-12,
    )
