from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .forecasting import QuantileForecast
from .historical import HistoricalTrajectorySet


@dataclass(frozen=True)
class SchaakeResult:
    """
    Output of the Schaake Shuffle.

    Attributes
    ----------
    scenarios:
        Final scenario matrix, shape (L, H).

    ranks:
        Historical ranks, shape (L, H). Each column is a permutation
        of 1,...,L.

    empirical_pseudo_observations:
        r/(L+1), shape (L, H). These describe the empirical historical
        copula and are independent of the marginal discretization choice.

    assigned_probabilities:
        Probability levels actually assigned to scenarios after mapping
        each historical rank to the marginal probability grid.

    marginal_probability_grid:
        Ordered probability levels used to discretize each TimesFM
        marginal, shape (L,).

    marginal_values:
        TimesFM marginal values evaluated at the probability grid,
        shape (H, L).
    """
    scenarios: np.ndarray
    ranks: np.ndarray
    empirical_pseudo_observations: np.ndarray
    assigned_probabilities: np.ndarray
    marginal_probability_grid: np.ndarray
    marginal_values: np.ndarray
    historical_metadata: pd.DataFrame | None = None

    @property
    def n_scenarios(self) -> int:
        return int(self.scenarios.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.scenarios.shape[1])


def _validate_historical_values(
    historical_values: np.ndarray,
    expected_horizon: int,
) -> np.ndarray:
    historical_values = np.asarray(
        historical_values,
        dtype=float,
    )

    if historical_values.ndim != 2:
        raise ValueError(
            "historical_values must have shape (L, H)."
        )

    if historical_values.shape[1] != expected_horizon:
        raise ValueError(
            "Historical horizon does not match the TimesFM forecast. "
            f"historical H={historical_values.shape[1]}, "
            f"forecast H={expected_horizon}."
        )

    if historical_values.shape[0] < 2:
        raise ValueError(
            "At least two historical trajectories are required."
        )

    if not np.isfinite(historical_values).all():
        raise ValueError(
            "historical_values contains non-finite values."
        )

    return historical_values


def _prepare_forecast_values(
    forecast: QuantileForecast,
    crossing_policy: str,
) -> np.ndarray:
    """
    Return a validated H x K quantile matrix.

    crossing_policy:
        "error" -> fail if any adjacent quantiles cross.
        "sort"  -> explicitly sort quantile values within each day.
    """
    forecast.validate()

    if crossing_policy not in {"error", "sort"}:
        raise ValueError(
            "crossing_policy must be 'error' or 'sort'."
        )

    values = np.asarray(
        forecast.values,
        dtype=float,
    ).copy()

    crossing = np.diff(values, axis=1) < 0

    if crossing.any():
        if crossing_policy == "error":
            n_crossings = int(crossing.sum())
            raise ValueError(
                "TimesFM quantile crossing detected. "
                f"Adjacent crossings: {n_crossings}. "
                "Use crossing_policy='sort' only if you explicitly "
                "want a monotone rearrangement."
            )

        values = np.sort(values, axis=1)

    return values


def _historical_ranks(
    historical_values: np.ndarray,
    tie_break: str = "stable",
    seed: int = 42,
) -> np.ndarray:
    """
    Rank historical trajectories separately at each forecast lead time.

    The result in every column is exactly a permutation of 1,...,L.

    tie_break:
        "stable":
            Equal historical values are broken deterministically by
            trajectory row order.

        "random":
            Equal values are broken randomly but reproducibly using seed.
    """
    if tie_break not in {"stable", "random"}:
        raise ValueError(
            "tie_break must be 'stable' or 'random'."
        )

    L, H = historical_values.shape
    ranks = np.empty((L, H), dtype=np.int64)

    rng = np.random.default_rng(seed)

    for t in range(H):
        column = historical_values[:, t]

        if tie_break == "stable":
            order = np.argsort(
                column,
                kind="stable",
            )
        else:
            # lexsort uses the last key as the primary key:
            # first sort by value, then use a random key only to
            # break exact ties.
            random_key = rng.random(L)
            order = np.lexsort(
                (random_key, column)
            )

        ranks[order, t] = np.arange(
            1,
            L + 1,
            dtype=np.int64,
        )

    return ranks


def build_probability_grid(
    forecast_probabilities: np.ndarray,
    n_scenarios: int,
    mode: str = "timesfm_support",
    custom_grid: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build the L probability levels used to discretize TimesFM marginals.

    mode="timesfm_support"
        Uniformly fills the probability support directly provided by
        TimesFM. For Q10,...,Q90:
            linspace(0.1, 0.9, L)

        This avoids extrapolating unmodelled tails and is the recommended
        default for the current zero-shot experiment.

    mode="schaake"
        Uses the classical plotting positions:
            k/(L+1), k=1,...,L

        This is accepted only if every required probability lies inside
        the TimesFM probability support. No tail extrapolation is done.

    mode="custom"
        Uses custom_grid. It must contain exactly L strictly increasing
        probabilities and must lie inside the TimesFM support.
    """
    probs = np.asarray(
        forecast_probabilities,
        dtype=float,
    )

    if probs.ndim != 1:
        raise ValueError(
            "forecast_probabilities must be one-dimensional."
        )

    if not np.all(np.diff(probs) > 0):
        raise ValueError(
            "forecast_probabilities must be strictly increasing."
        )

    if mode == "timesfm_support":
        grid = np.linspace(
            probs[0],
            probs[-1],
            n_scenarios,
        )

    elif mode == "schaake":
        grid = (
            np.arange(1, n_scenarios + 1, dtype=float)
            / (n_scenarios + 1)
        )

    elif mode == "custom":
        if custom_grid is None:
            raise ValueError(
                "custom_grid is required when mode='custom'."
            )

        grid = np.asarray(
            custom_grid,
            dtype=float,
        )

        if grid.shape != (n_scenarios,):
            raise ValueError(
                "custom_grid must have shape (L,). "
                f"Expected {(n_scenarios,)}, got {grid.shape}."
            )

    else:
        raise ValueError(
            "mode must be 'timesfm_support', 'schaake', or 'custom'."
        )

    if not np.all(np.diff(grid) > 0):
        raise ValueError(
            "The marginal probability grid must be strictly increasing."
        )

    tolerance = 1e-12

    if (
        grid[0] < probs[0] - tolerance
        or grid[-1] > probs[-1] + tolerance
    ):
        raise ValueError(
            "The requested marginal grid extends beyond the "
            "probability support supplied by TimesFM. "
            f"TimesFM support=[{probs[0]:.6f}, {probs[-1]:.6f}], "
            f"requested=[{grid[0]:.6f}, {grid[-1]:.6f}]. "
            "Tail extrapolation is intentionally disabled."
        )

    return grid


def _interpolate_marginals(
    probabilities: np.ndarray,
    quantile_values: np.ndarray,
    probability_grid: np.ndarray,
) -> np.ndarray:
    """
    Linear monotone interpolation of each TimesFM marginal.

    quantile_values:
        shape (H, K)

    returns:
        shape (H, L)
    """
    H = quantile_values.shape[0]
    L = len(probability_grid)

    result = np.empty(
        (H, L),
        dtype=float,
    )

    for t in range(H):
        result[t, :] = np.interp(
            probability_grid,
            probabilities,
            quantile_values[t, :],
        )

    return result


def schaake_shuffle(
    forecast: QuantileForecast,
    historical: HistoricalTrajectorySet | np.ndarray,
    probability_grid_mode: str = "timesfm_support",
    custom_probability_grid: np.ndarray | None = None,
    tie_break: str = "stable",
    seed: int = 42,
    crossing_policy: str = "error",
) -> SchaakeResult:
    """
    Generate temporally dependent scenarios with Schaake Shuffle.

    The historical dependence is represented by complete row trajectories.
    Ranks are computed by column, but the full vector of ranks from each
    historical row stays together when TimesFM marginal values are assigned.

    No future observations are required by this function.
    """
    quantile_values = _prepare_forecast_values(
        forecast=forecast,
        crossing_policy=crossing_policy,
    )

    if isinstance(
        historical,
        HistoricalTrajectorySet,
    ):
        historical_values = historical.values
        historical_metadata = historical.metadata.copy()
    else:
        historical_values = historical
        historical_metadata = None

    historical_values = _validate_historical_values(
        historical_values,
        expected_horizon=forecast.horizon,
    )

    L, H = historical_values.shape

    ranks = _historical_ranks(
        historical_values,
        tie_break=tie_break,
        seed=seed,
    )

    # This is the empirical copula representation:
    # it is always r/(L+1), independent of how TimesFM marginals
    # are discretized.
    empirical_pseudo_observations = (
        ranks.astype(float)
        / (L + 1)
    )

    marginal_probability_grid = build_probability_grid(
        forecast_probabilities=forecast.probabilities,
        n_scenarios=L,
        mode=probability_grid_mode,
        custom_grid=custom_probability_grid,
    )

    marginal_values = _interpolate_marginals(
        probabilities=forecast.probabilities,
        quantile_values=quantile_values,
        probability_grid=marginal_probability_grid,
    )

    scenarios = np.empty(
        (L, H),
        dtype=float,
    )

    assigned_probabilities = np.empty(
        (L, H),
        dtype=float,
    )

    for t in range(H):
        rank_index = ranks[:, t] - 1

        scenarios[:, t] = marginal_values[
            t,
            rank_index,
        ]

        assigned_probabilities[:, t] = (
            marginal_probability_grid[
                rank_index
            ]
        )

    return SchaakeResult(
        scenarios=scenarios,
        ranks=ranks,
        empirical_pseudo_observations=(
            empirical_pseudo_observations
        ),
        assigned_probabilities=assigned_probabilities,
        marginal_probability_grid=(
            marginal_probability_grid
        ),
        marginal_values=marginal_values,
        historical_metadata=historical_metadata,
    )


def validate_shuffle_result(
    result: SchaakeResult,
) -> None:
    """
    Internal consistency checks for the generated scenarios.

    For each forecast day, the final scenario values must be exactly a
    permutation of the L discretized TimesFM marginal values.
    """
    L, H = result.scenarios.shape

    if result.ranks.shape != (L, H):
        raise RuntimeError(
            "Unexpected ranks shape."
        )

    expected_ranks = np.arange(
        1,
        L + 1,
        dtype=np.int64,
    )

    for t in range(H):
        if not np.array_equal(
            np.sort(result.ranks[:, t]),
            expected_ranks,
        ):
            raise RuntimeError(
                f"Ranks are not a permutation of 1,...,L at t={t}."
            )

        if not np.allclose(
            np.sort(result.scenarios[:, t]),
            result.marginal_values[t, :],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Marginal permutation property failed "
                f"at forecast step t={t}."
            )


def save_schaake_result(
    result: SchaakeResult,
    output_dir: str | Path,
    forecast_dates: pd.DatetimeIndex | None = None,
) -> None:
    """
    Save the main Schaake artifacts as CSV files for auditability.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    L, H = result.scenarios.shape

    if forecast_dates is None:
        scenario_columns = [
            f"day_{t + 1:02d}"
            for t in range(H)
        ]
    else:
        forecast_dates = pd.DatetimeIndex(
            forecast_dates
        )

        if len(forecast_dates) != H:
            raise ValueError(
                "forecast_dates length must equal H."
            )

        scenario_columns = [
            date.date().isoformat()
            for date in forecast_dates
        ]

    scenario_index = [
        f"scenario_{i + 1:03d}"
        for i in range(L)
    ]

    scenarios_df = pd.DataFrame(
        result.scenarios,
        index=scenario_index,
        columns=scenario_columns,
    )
    scenarios_df.index.name = "scenario_id"
    scenarios_df.to_csv(
        output_dir / "scenarios.csv"
    )

    ranks_df = pd.DataFrame(
        result.ranks,
        index=scenario_index,
        columns=scenario_columns,
    )
    ranks_df.index.name = "scenario_id"
    ranks_df.to_csv(
        output_dir / "historical_ranks.csv"
    )

    empirical_u_df = pd.DataFrame(
        result.empirical_pseudo_observations,
        index=scenario_index,
        columns=scenario_columns,
    )
    empirical_u_df.index.name = "scenario_id"
    empirical_u_df.to_csv(
        output_dir / "empirical_pseudo_observations.csv"
    )

    assigned_u_df = pd.DataFrame(
        result.assigned_probabilities,
        index=scenario_index,
        columns=scenario_columns,
    )
    assigned_u_df.index.name = "scenario_id"
    assigned_u_df.to_csv(
        output_dir / "assigned_probabilities.csv"
    )

    marginal_df = pd.DataFrame(
        result.marginal_values.T,
        index=[
            f"marginal_order_{i + 1:03d}"
            for i in range(L)
        ],
        columns=scenario_columns,
    )
    marginal_df.insert(
        0,
        "probability",
        result.marginal_probability_grid,
    )
    marginal_df.index.name = "marginal_id"
    marginal_df.to_csv(
        output_dir / "timesfm_discretized_marginals.csv"
    )

    if result.historical_metadata is not None:
        result.historical_metadata.to_csv(
            output_dir / "historical_metadata.csv",
            index=False,
        )
