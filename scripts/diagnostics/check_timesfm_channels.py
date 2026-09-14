from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from transformers import TimesFm2_5ModelForPrediction

from wind_scenarios.data import DEFAULT_DATA_PATH, DEFAULT_TARGET_COLUMN, load_power_dataframe


MODEL_ID = "google/timesfm-2.5-200m-transformers"
CONTEXT_LEN = 32
HORIZON_LEN = 28


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = TimesFm2_5ModelForPrediction.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    ).to(device)
    model.eval()

    df = load_power_dataframe(DEFAULT_DATA_PATH)
    forecast_date = pd.Timestamp("2014-01-01")
    matches = df.index[df["DATE"] == forecast_date]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely find forecast origin {forecast_date.date()}.")

    origin_idx = int(matches[0])
    context = df.loc[
        origin_idx - CONTEXT_LEN : origin_idx - 1,
        DEFAULT_TARGET_COLUMN,
    ].to_numpy(dtype=np.float32)

    context_tensor = torch.tensor(
        context,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            past_values=context_tensor,
            forecast_context_len=CONTEXT_LEN,
        )

    full = outputs.full_predictions[0, :HORIZON_LEN].float().cpu().numpy()
    channel_0 = full[:, 0]
    q10 = full[:, 1]
    q50 = full[:, 5]
    q90 = full[:, 9]

    print("\n=== TimesFM channel diagnostic on POWER_MW ===")
    print("full_predictions shape:", full.shape)
    print("channel 0 mean:", float(channel_0.mean()))
    print("Q10 mean:", float(q10.mean()))
    print("Q50 mean:", float(q50.mean()))
    print("Q90 mean:", float(q90.mean()))
    print("mean |channel0-Q50|:", float(np.mean(np.abs(channel_0 - q50))))


if __name__ == "__main__":
    main()
