"""Evaluate a trained TASSIR-Net checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from walar.checkpoint import load_checkpoint, load_model
from walar.data import SpectralPairDataset
from walar.engine import evaluate
from walar.losses import ReconstructionLoss
from walar.model import TASSIRNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TASSIR-Net.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("./data/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("./evaluation"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers nonnegative")
    return args


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    return device


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model_config = checkpoint.get("model_config", {})

    model = TASSIRNet(**model_config).to(device)
    load_model(model, checkpoint)
    criterion = ReconstructionLoss(ssim_weight=0.1).to(device)
    dataset = SpectralPairDataset(args.data_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    result = evaluate(
        model,
        data_loader,
        criterion,
        device,
        group_by_wavelength=True,
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "epoch": int(checkpoint.get("epoch", 0)),
        "overall": result.overall.as_dict(),
        "by_wavelength_nm": {
            key: value.as_dict() for key, value in result.by_wavelength.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    metrics = result.overall
    print(
        f"MSE {metrics.mse:.6f} | PSNR {metrics.psnr:.3f} dB | SSIM {metrics.ssim:.6f}"
    )
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
