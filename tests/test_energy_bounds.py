import numpy as np

from wind_scenarios.forecasting import (
    QuantileForecast,
    apply_value_bounds,
)


def test_daily_energy_bounds_clip():
    forecast = QuantileForecast(
        probabilities=np.array(
            [0.1, 0.5, 0.9]
        ),
        values=np.array(
            [
                [-2.0, 40.0, 90.0],
                [0.0, 60.0, 80.0],
            ]
        ),
    )

    bounded = apply_value_bounds(
        forecast,
        lower=0.0,
        upper=84.0,
        policy="clip",
    )

    np.testing.assert_allclose(
        bounded.values,
        [
            [0.0, 40.0, 84.0],
            [0.0, 60.0, 80.0],
        ],
    )
