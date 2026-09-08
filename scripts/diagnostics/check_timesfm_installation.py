"""Smoke test for the local TimesFM 2.5 installation.

This script does not use the WIND dataset. It only verifies that the pretrained
model can be loaded and that a small artificial inference produces the expected
output tensors.
"""

from __future__ import annotations

import torch
from transformers import TimesFm2_5ModelForPrediction


MODEL_ID = "google/timesfm-2.5-200m-transformers"
CONTEXT_LEN = 512


def main() -> None:
    cuda_available = torch.cuda.is_available()

    print("CUDA available:", cuda_available)
    if cuda_available:
        print("GPU:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if cuda_available else "cpu")
    dtype = torch.bfloat16 if cuda_available else torch.float32

    print(f"\nLoading TimesFM 2.5: {MODEL_ID}")

    model = TimesFm2_5ModelForPrediction.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    ).to(device)
    model.eval()

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    num_params = sum(parameter.numel() for parameter in model.parameters())

    print("Model loaded correctly")
    print("Device:", model_device)
    print("dtype:", model_dtype)
    print(f"Parameters: {num_params / 1e6:.1f} M")

    # Artificial series: this is only a smoke test of inference.
    context = torch.sin(
        torch.linspace(
            0,
            20,
            CONTEXT_LEN,
            device=device,
            dtype=torch.float32,
        )
    ).unsqueeze(0)

    print("\nRunning artificial inference...")

    with torch.no_grad():
        outputs = model(
            past_values=context,
            forecast_context_len=CONTEXT_LEN,
        )

    mean_predictions = outputs.mean_predictions
    full_predictions = outputs.full_predictions

    print("Inference completed")
    print("Shape mean_predictions:", tuple(mean_predictions.shape))
    print("Shape full_predictions:", tuple(full_predictions.shape))
    print("First 5 point-forecast values:")
    print(mean_predictions[0, :5].float().cpu().numpy())


if __name__ == "__main__":
    main()
