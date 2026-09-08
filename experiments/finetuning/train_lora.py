from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import TimesFm2_5ModelForPrediction

from wind_scenarios.data import (
    DEFAULT_DATA_PATH,
    ForecastConfig,
    SamplingConfig,
    SplitConfig,
    build_datasets,
    split_summary,
)

MODEL_ID = "google/timesfm-2.5-200m-transformers"

# TimesFM 2.5 quantile channels in full_predictions.
Q10 = 0.1
Q90 = 0.9
Q10_CHANNEL = 1
Q90_CHANNEL = 9


def parse_args() -> argparse.Namespace:
    """Experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Fine-tune TimesFM 2.5 with LoRA using only Q10 and Q90."
    )
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs") / "timesfm25_wind_ab_q10_q90_only",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.10)

    parser.add_argument("--num_samples", type=int, default=900)
    parser.add_argument("--min_start_distance_days", type=int, default=7)

    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.001)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_timesfm_config(model):
    """Return TimesFM config for a base or PEFT-wrapped model."""
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "quantiles"):
        return config

    if hasattr(model, "get_base_model"):
        config = getattr(model.get_base_model(), "config", None)
        if config is not None and hasattr(config, "quantiles"):
            return config

    raise RuntimeError("Could not recover TimesFM configuration.")


def quantile_channel_map(model) -> dict[float, int]:
    """
    Map TimesFM quantiles to full_predictions channels:
    1->Q10, 2->Q20, ..., 5->Q50, ..., 9->Q90.
    """
    config = get_timesfm_config(model)
    return {
        float(q): index + 1
        for index, q in enumerate(config.quantiles)
    }


def pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Standard pinball loss."""
    error = target - prediction
    return torch.maximum(
        quantile * error,
        (quantile - 1.0) * error,
    ).mean()


def q10_q90_loss(
    model,
    context: torch.Tensor,
    target: torch.Tensor,
    model_context_len: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Training objective:

        L = 0.5 * [PB_0.10(Q10, y) + PB_0.90(Q90, y)]

    No MSE term and no direct loss for any other quantile.
    """
    if context.ndim != 2 or target.ndim != 2:
        raise ValueError("context and target must have shape [batch, time].")

    if context.shape[1] != model_context_len:
        raise ValueError(
            "This experiment assumes context_len == model_context_len."
        )

    outputs = model(
        past_values=context,
        forecast_context_len=model_context_len,
        truncate_negative=False,
    )

    horizon = target.shape[1]
    full_predictions = outputs.full_predictions[:, :horizon, :]

    # Recreate the normalized space used for the training objective.
    mean = context.mean(dim=1, keepdim=True)
    std = context.std(dim=1, keepdim=True)
    safe_std = torch.where(std < 1e-6, torch.ones_like(std), std)

    target_normalized = (target - mean) / safe_std
    predictions_normalized = (
        full_predictions - mean[:, :, None]
    ) / safe_std[:, :, None]

    q10_prediction = predictions_normalized[:, :, Q10_CHANNEL]
    q90_prediction = predictions_normalized[:, :, Q90_CHANNEL]

    pb10 = pinball_loss(q10_prediction, target_normalized, Q10)
    pb90 = pinball_loss(q90_prediction, target_normalized, Q90)

    loss = 0.5 * (pb10 + pb90)

    return loss, {
        "pb10": pb10.detach(),
        "pb90": pb90.detach(),
    }


@torch.no_grad()
def evaluate_validation(
    model,
    loader: DataLoader,
    device: torch.device,
    model_context_len: int,
) -> dict[str, float]:
    """Evaluate validation using the same Q10/Q90 loss as training."""
    model.eval()

    loss_sum = 0.0
    pb10_sum = 0.0
    pb90_sum = 0.0
    n_examples = 0

    for context, target in loader:
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        loss, components = q10_q90_loss(
            model=model,
            context=context,
            target=target,
            model_context_len=model_context_len,
        )

        batch_size = context.shape[0]
        loss_sum += float(loss.item()) * batch_size
        pb10_sum += float(components["pb10"].item()) * batch_size
        pb90_sum += float(components["pb90"].item()) * batch_size
        n_examples += batch_size

    if n_examples == 0:
        raise RuntimeError("Validation loader contains no samples.")

    return {
        "loss": loss_sum / n_examples,
        "pb10": pb10_sum / n_examples,
        "pb90": pb90_sum / n_examples,
    }


def count_parameters(model) -> tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 GPU support is required.")

    device = torch.device("cuda")

    # Fixed split and forecasting task.
    split = SplitConfig(
        development_start_year=1980,
        development_end_year=2013,
        test_start_year=2014,
        test_end_year=2017,
        validation_start_month=11,
    )
    forecast = ForecastConfig(
        context_len=32,
        horizon_len=28,
        model_context_len=32,
    )
    sampling = SamplingConfig(
        num_samples=args.num_samples,
        min_start_distance_days=args.min_start_distance_days,
        seed=args.seed,
    )

    _, train_dataset, validation_dataset, _ = build_datasets(
        data_path=args.data_path,
        split=split,
        forecast=forecast,
        sampling=sampling,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Train batches/epoch: {len(train_loader)}")

    print("\n=== VALIDATION SCHEDULE ===")
    print(split_summary(split).to_string(index=False))

    # Load pretrained TimesFM and verify Q10/Q90 channel positions.
    print(f"\nLoading {MODEL_ID} ...")
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    ).to(device)

    channel_map = quantile_channel_map(model)
    print(f"Quantile channel map: {channel_map}")

    if (
        channel_map.get(Q10) != Q10_CHANNEL
        or channel_map.get(Q90) != Q90_CHANNEL
    ):
        raise RuntimeError("Unexpected TimesFM Q10/Q90 channel mapping.")

    # Add LoRA adapters. Only LoRA parameters are trainable.
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            lora_dropout=args.lora_dropout,
            bias="none",
        ),
    )

    total_parameters, trainable_parameters = count_parameters(model)
    print(f"Trainable parameters: {trainable_parameters:,}/{total_parameters:,}")

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader),
    )

    # Create output files.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "training_history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_pb10",
                "val_pb90",
                "learning_rate",
                "best_so_far",
            ]
        )

    with (args.output_dir / "experiment_config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "model_id": MODEL_ID,
                "objective": "0.5 * (PB10 + PB90)",
                "fine_tuned_quantiles": [Q10, Q90],
                "split": asdict(split),
                "forecast": asdict(forecast),
                "sampling": asdict(sampling),
                "training": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "max_grad_norm": args.max_grad_norm,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "patience": args.patience,
                    "min_delta": args.min_delta,
                    "seed": args.seed,
                },
            },
            file,
            indent=2,
        )

    initial_validation = evaluate_validation(
        model=model,
        loader=validation_loader,
        device=device,
        model_context_len=forecast.model_context_len,
    )
    print(
        "\nInitial validation | "
        f"loss={initial_validation['loss']:.6f} | "
        f"PB10={initial_validation['pb10']:.6f} | "
        f"PB90={initial_validation['pb90']:.6f}"
    )

    # Early stopping is based only on the mean Q10/Q90 validation pinball loss.
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_examples = 0

        for context, target in train_loader:
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            loss, _ = q10_q90_loss(
                model=model,
                context=context,
                target=target,
                model_context_len=forecast.model_context_len,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()

            batch_size = context.shape[0]
            train_loss_sum += float(loss.item()) * batch_size
            train_examples += batch_size

        train_loss = train_loss_sum / train_examples

        validation = evaluate_validation(
            model=model,
            loader=validation_loader,
            device=device,
            model_context_len=forecast.model_context_len,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        improved = (
            validation["loss"]
            < best_validation_loss - args.min_delta
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.6f} | "
            f"val={validation['loss']:.6f} | "
            f"PB10={validation['pb10']:.6f} | "
            f"PB90={validation['pb90']:.6f} | "
            f"lr={current_lr:.3e}"
        )

        with history_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                [
                    epoch,
                    train_loss,
                    validation["loss"],
                    validation["pb10"],
                    validation["pb90"],
                    current_lr,
                    int(improved),
                ]
            )

        if improved:
            best_validation_loss = validation["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0

            model.save_pretrained(args.output_dir)

            with (args.output_dir / "best_metrics.json").open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "best_val_loss": best_validation_loss,
                        "best_val_pb10": validation["pb10"],
                        "best_val_pb90": validation["pb90"],
                    },
                    file,
                    indent=2,
                )

            print("  New best validation loss. Adapter saved.")
        else:
            epochs_without_improvement += 1
            print(
                "  No improvement "
                f"({epochs_without_improvement}/{args.patience})"
            )

        if epochs_without_improvement >= args.patience:
            print("\nEarly stopping triggered.")
            break

    print("\nTraining finished.")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Adapter: {args.output_dir}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
