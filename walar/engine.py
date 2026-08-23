"""Training and evaluation loops."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import ReconstructionLoss


@dataclass(frozen=True)
class EpochMetrics:
    count: int
    loss: float
    mse: float
    psnr: float
    ssim: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    overall: EpochMetrics
    by_wavelength: dict[str, EpochMetrics]


class _Accumulator:
    def __init__(self, ssim_weight: float) -> None:
        self.ssim_weight = ssim_weight
        self.count = 0
        self.loss = 0.0
        self.mse = 0.0
        self.psnr = 0.0
        self.ssim = 0.0

    def update(self, mse: Tensor, ssim: Tensor) -> None:
        mse = mse.detach().float().cpu()
        ssim = ssim.detach().float().cpu()
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
        self.count += mse.numel()
        self.loss += float((mse - self.ssim_weight * ssim).sum())
        self.mse += float(mse.sum())
        self.psnr += float(psnr.sum())
        self.ssim += float(ssim.sum())

    def result(self) -> EpochMetrics:
        if self.count == 0:
            raise RuntimeError("The data loader is empty.")
        return EpochMetrics(
            count=self.count,
            loss=self.loss / self.count,
            mse=self.mse / self.count,
            psnr=self.psnr / self.count,
            ssim=self.ssim / self.count,
        )


def _to_device(
    batch: tuple[Tensor, Tensor, Tensor], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    measurement, wavelength, target = batch
    non_blocking = device.type == "cuda"
    return (
        measurement.to(device, non_blocking=non_blocking),
        wavelength.to(device, non_blocking=non_blocking),
        target.to(device, non_blocking=non_blocking),
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ReconstructionLoss,
    device: torch.device,
) -> EpochMetrics:
    model.train()
    metrics = _Accumulator(criterion.ssim_weight)
    for batch in tqdm(data_loader, desc="train", leave=False):
        measurement, wavelength, target = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(measurement, wavelength)
        loss = criterion(prediction, target)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics.update(loss.mse_per_image, loss.ssim_per_image)
    return metrics.result()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: ReconstructionLoss,
    device: torch.device,
    *,
    group_by_wavelength: bool = False,
) -> EvaluationResult:
    model.eval()
    overall = _Accumulator(criterion.ssim_weight)
    grouped: dict[str, _Accumulator] = {}

    for batch in tqdm(data_loader, desc="evaluate", leave=False):
        measurement, wavelength, target = _to_device(batch, device)
        prediction = model(measurement, wavelength)
        loss = criterion(prediction, target)
        overall.update(loss.mse_per_image, loss.ssim_per_image)

        if group_by_wavelength:
            values = wavelength.detach().flatten().cpu()
            mse = loss.mse_per_image.detach().cpu()
            ssim = loss.ssim_per_image.detach().cpu()
            for index, value in enumerate(values):
                wavelength_nm = round(450.0 + 250.0 * float(value))
                key = str(wavelength_nm)
                grouped.setdefault(key, _Accumulator(criterion.ssim_weight)).update(
                    mse[index : index + 1], ssim[index : index + 1]
                )

    return EvaluationResult(
        overall=overall.result(),
        by_wavelength={key: value.result() for key, value in grouped.items()},
    )
