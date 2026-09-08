import numpy as np

from wind_scenarios.forecasting import QuantileForecast
from wind_scenarios.schaake import schaake_shuffle, validate_shuffle_result


def test_schaake_preserves_discretized_marginals():
    forecast = QuantileForecast(
        probabilities=np.array([0.2, 0.4, 0.6, 0.8]),
        values=np.array(
            [
                [5.0, 6.0, 7.2, 8.6],
                [4.8, 5.9, 7.0, 8.4],
                [5.2, 6.1, 7.4, 9.0],
            ]
        ),
    )

    historical = np.array(
        [
            [7.0, 8.0, 6.0],
            [4.5, 5.0, 7.5],
            [8.2, 7.1, 8.0],
            [6.0, 4.7, 5.5],
        ]
    )

    result = schaake_shuffle(
        forecast=forecast,
        historical=historical,
        probability_grid_mode="schaake",
    )

    validate_shuffle_result(result)

    expected = np.array(
        [
            [7.2, 8.4, 6.1],
            [5.0, 5.9, 7.4],
            [8.6, 7.0, 9.0],
            [6.0, 4.8, 5.2],
        ]
    )

    np.testing.assert_allclose(result.scenarios, expected)
