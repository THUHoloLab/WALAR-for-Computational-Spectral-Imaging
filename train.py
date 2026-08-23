"""Train TASSIR-Net."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from walar.checkpoint import (
    load_checkpoint,
    load_model,
    load_optimizer,
    save_checkpoint,
)
from walar.data import SpectralPairDataset
from walar.engine import EpochMetrics, evaluate, train_one_epoch
from walar.losses import ReconstructionLoss
from walar.model import TASSIRNet


MODEL_CONFIG = {
    "image_size": 490,
    "patch_size": 16,
    "stride": 12,
    "dim": 1024,
    "depth": 7,
    "heads": 8,
    "dim_head": 128,
    "layerscale_init": 1e-4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TASSIR-Net.")
    parser.add_argument("--train-dir", type=Path, default=Path("./data/train"))
    parser.add_argument(
        "--validation-dir", type=Path, default=Path("./data/validation")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")
    return args


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    return device


def print_metrics(epoch: int, train: EpochMetrics, validation: EpochMetrics) -> None:
    print(
        f"Epoch {epoch:02d}: "
        f"train loss {train.loss:.6f} | "
        f"validation PSNR {validation.psnr:.3f} dB, "
        f"SSIM {validation.ssim:.6f}"
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    device = select_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_data = SpectralPairDataset(args.train_dir)
    validation_data = SpectralPairDataset(args.validation_dir)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = TASSIRNet(**MODEL_CONFIG).to(device)
    criterion = ReconstructionLoss(ssim_weight=0.1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )

    start_epoch = 0
    best_psnr = float("-inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        load_model(model, checkpoint)
        if not load_optimizer(optimizer, checkpoint):
            raise KeyError("Optimizer state is missing from the checkpoint.")
        start_epoch = int(checkpoint.get("epoch", 0))
        best_psnr = float(
            checkpoint.get("best_psnr", checkpoint.get("best_validation_metric", -1))
        )

    if start_epoch >= args.epochs:
        raise ValueError("The checkpoint epoch must be lower than --epochs.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.csv"
    append_history = bool(args.resume and history_path.exists())
    fields = [
        "epoch",
        "train_loss",
        "train_psnr",
        "train_ssim",
        "validation_mse",
        "validation_psnr",
        "validation_ssim",
    ]

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}")
    print(f"Trainable parameters: {parameters:,}")
    print(f"Samples: {len(train_data):,} train, {len(validation_data):,} validation")

    mode = "a" if append_history else "w"
    with history_path.open(mode, newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fields)
        if not append_history:
            writer.writeheader()

        for epoch in range(start_epoch + 1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            validation_metrics = evaluate(
                model, validation_loader, criterion, device
            ).overall
            print_metrics(epoch, train_metrics, validation_metrics)

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics.loss,
                    "train_psnr": train_metrics.psnr,
                    "train_ssim": train_metrics.ssim,
                    "validation_mse": validation_metrics.mse,
                    "validation_psnr": validation_metrics.psnr,
                    "validation_ssim": validation_metrics.ssim,
                }
            )
            history_file.flush()

            improved = validation_metrics.psnr > best_psnr
            if improved:
                best_psnr = validation_metrics.psnr
            state = {
                "epoch": epoch,
                "model": model,
                "optimizer": optimizer,
                "best_psnr": best_psnr,
                "model_config": MODEL_CONFIG,
            }
            save_checkpoint(args.output_dir / "last.pt", **state)
            if improved:
                save_checkpoint(args.output_dir / "best.pt", **state)


if __name__ == "__main__":
    main()
