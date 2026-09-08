import numpy as np
import pandas as pd

from wind_scenarios.data import DEFAULT_DATA_PATH, load_wind_dataframe
import torch

from transformers import TimesFm2_5ModelForPrediction


MODEL_ID = "google/timesfm-2.5-200m-transformers"

CONTEXT_LEN = 32
HORIZON_LEN = 28


# ---------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TimesFm2_5ModelForPrediction.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
).to(device)

model.eval()


# ---------------------------------------------------------
# 2. Load wind data
# ---------------------------------------------------------

df = load_wind_dataframe(DEFAULT_DATA_PATH)

# Example:
# forecast starts on 2014-01-01
forecast_date = pd.Timestamp("2014-01-01")

origin_idx = df.index[df["DATE"] == forecast_date][0]

context = df.loc[
    origin_idx - CONTEXT_LEN : origin_idx - 1,
    "WIND"
].to_numpy(dtype=np.float32)

ground_truth = df.loc[
    origin_idx : origin_idx + HORIZON_LEN - 1,
    "WIND"
].to_numpy(dtype=np.float32)


# ---------------------------------------------------------
# 3. Zero-shot inference
# ---------------------------------------------------------

context_tensor = torch.tensor(
    context,
    dtype=torch.float32,
    device=device,
).unsqueeze(0)

with torch.no_grad():
    outputs = model(
        past_values=context_tensor,
        forecast_context_len=CONTEXT_LEN,
        truncate_negative=False,
        return_dict=True,
    )


full = outputs.full_predictions[0, :HORIZON_LEN].float().cpu().numpy()
mean_predictions = (
    outputs.mean_predictions[0, :HORIZON_LEN]
    .float()
    .cpu()
    .numpy()
)


# ---------------------------------------------------------
# 4. Extract channels
# ---------------------------------------------------------

channel_0 = full[:, 0]
q50 = full[:, 5]

q10 = full[:, 1]
q90 = full[:, 9]


# ---------------------------------------------------------
# 5. Compare
# ---------------------------------------------------------

print("\n===================================")
print("TimesFM 2.5 Zero-Shot")
print("Mean vs median diagnostic")
print("===================================\n")

print("mean_predictions == channel 5?")
print(
    np.allclose(
        mean_predictions,
        q50,
        rtol=1e-5,
        atol=1e-5,
    )
)

print("\nMean absolute difference:")
print(
    f"channel 0 vs channel 5: "
    f"{np.mean(np.abs(channel_0 - q50)):.6f}"
)

print("\nMaximum absolute difference:")
print(
    f"channel 0 vs channel 5: "
    f"{np.max(np.abs(channel_0 - q50)):.6f}"
)

print("\nAverage forecasts:")
print(f"channel 0 : {channel_0.mean():.6f}")
print(f"channel 5 : {q50.mean():.6f}")

print("\nFirst 10 forecast steps:\n")

for h in range(min(10, HORIZON_LEN)):
    print(
        f"h={h+1:02d} | "
        f"channel0={channel_0[h]:8.4f} | "
        f"Q50={q50[h]:8.4f} | "
        f"diff={channel_0[h] - q50[h]:+8.4f}"
    )