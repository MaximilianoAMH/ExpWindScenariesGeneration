import numpy as np

from wind_scenarios.forecasting import QuantileForecast, apply_value_bounds


def test_power_bounds_clip():
    forecast = QuantileForecast(
        probabilities=np.array([0.1, 0.5, 0.9]),
        values=np.array(
            [
                [-0.2, 1.0, 4.1],
                [0.0, 2.0, 3.0],
            ]
        ),
    )
    bounded = apply_value_bounds(
        forecast,
        lower=0.0,
        upper=3.5,
        policy="clip",
    )
    np.testing.assert_allclose(
        bounded.values,
        [[0.0, 1.0, 3.5], [0.0, 2.0, 3.0]],
    )
