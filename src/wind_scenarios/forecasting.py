from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class QuantileForecast:
    """
    Container for a probabilistic TimesFM forecast.

    Attributes
    ----------
    probabilities:
        Array of quantile levels, shape (K,).
        Example: [0.1, 0.2, ..., 0.9].

    values:
        Forecast quantiles, shape (H, K).
        values[t, k] = Q_t(probabilities[k]).
    """
    probabilities: np.ndarray
    values: np.ndarray

    @property
    def horizon(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_quantiles(self) -> int:
        return int(self.values.shape[1])

    def validate(self) -> None:
        if self.probabilities.ndim != 1:
            raise ValueError("probabilities must have shape (K,).")

        if self.values.ndim != 2:
            raise ValueError("values must have shape (H, K).")

        if self.values.shape[1] != len(self.probabilities):
            raise ValueError(
                "values.shape[1] must equal len(probabilities)."
            )

        if not np.all(np.diff(self.probabilities) > 0):
            raise ValueError(
                "Quantile probabilities must be strictly increasing."
            )

        if np.any(~np.isfinite(self.values)):
            raise ValueError("Forecast contains non-finite values.")


def get_timesfm_config(model):
    """
    Recover the TimesFM config from either a base model
    or a PEFT-wrapped model.
    """
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "quantiles"):
        return config

    if hasattr(model, "get_base_model"):
        config = getattr(model.get_base_model(), "config", None)
        if config is not None and hasattr(config, "quantiles"):
            return config

    raise RuntimeError(
        "Could not recover TimesFM configuration with quantile metadata."
    )


def quantile_channel_map(model) -> dict[float, int]:
    """
    Map each TimesFM quantile level to its full_predictions channel.

    In TimesFM 2.5, channel 0 is not a quantile channel.
    Quantile channels start at index 1.
    """
    config = get_timesfm_config(model)

    return {
        float(q): index + 1
        for index, q in enumerate(config.quantiles)
    }


@torch.no_grad()
def forecast_quantiles(
    model,
    context: torch.Tensor | np.ndarray,
    horizon_len: int,
    model_context_len: int,
    device: torch.device | str,
) -> QuantileForecast:
    """
    Generate all quantiles exposed by the TimesFM model.

    Important
    ---------
    This function uses only the observed context.
    It does not receive or use the future target.

    Parameters
    ----------
    model:
        TimesFM base model or PEFT-wrapped model.

    context:
        One-dimensional historical context.

    horizon_len:
        Number of future steps to predict.

    model_context_len:
        Context length passed to TimesFM.

    device:
        Torch device.

    Returns
    -------
    QuantileForecast
        probabilities shape: (K,)
        values shape: (H, K)
    """
    if isinstance(context, np.ndarray):
        context_tensor = torch.as_tensor(
            context,
            dtype=torch.float32,
            device=device,
        )
    else:
        context_tensor = context.to(
            device=device,
            dtype=torch.float32,
        )

    if context_tensor.ndim != 1:
        raise ValueError(
            f"context must be 1-D, got shape {tuple(context_tensor.shape)}."
        )

    input_values = context_tensor.unsqueeze(0)

    outputs = model(
        past_values=input_values,
        forecast_context_len=model_context_len,
    )

    full_predictions = (
        outputs.full_predictions[0, :horizon_len, :]
        .float()
        .cpu()
        .numpy()
    )

    channel_map = quantile_channel_map(model)

    probabilities = np.array(
        sorted(channel_map.keys()),
        dtype=float,
    )

    channels = np.array(
        [channel_map[q] for q in probabilities],
        dtype=int,
    )

    values = full_predictions[:, channels]

    result = QuantileForecast(
        probabilities=probabilities,
        values=values,
    )
    result.validate()

    return result


def quantile_crossing_mask(
    forecast: QuantileForecast,
) -> np.ndarray:
    """
    Boolean mask of quantile crossings.

    Returns shape (H, K-1), where True means
    Q_t(p_k) > Q_t(p_{k+1}).
    """
    return np.diff(forecast.values, axis=1) < 0


def quantile_crossing_rate(
    forecast: QuantileForecast,
) -> float:
    """
    Fraction of adjacent quantile pairs that cross.
    """
    mask = quantile_crossing_mask(forecast)
    if mask.size == 0:
        return 0.0
    return float(mask.mean())


def enforce_monotone_quantiles(
    forecast: QuantileForecast,
) -> QuantileForecast:
    """
    Simple non-decreasing rearrangement by sorting quantiles
    independently at each forecast step.

    This is intentionally separate from forecast_quantiles()
    so that any correction is explicit rather than silent.
    """
    corrected = np.sort(forecast.values, axis=1)

    result = QuantileForecast(
        probabilities=forecast.probabilities.copy(),
        values=corrected,
    )
    result.validate()
    return result
