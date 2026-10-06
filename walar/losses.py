"""Loss functions used to train TASSIR-Net."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _gaussian_window(window_size: int, sigma: float) -> Tensor:
    coordinates = torch.arange(window_size, dtype=torch.float32)
    coordinates -= window_size // 2
    kernel_1d = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d).view(1, 1, window_size, window_size)


class StructuralSimilarity(nn.Module):
    """Single-band structural similarity with a Gaussian local window."""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.register_buffer("window", _gaussian_window(window_size, sigma))

    def per_image(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError("prediction and target must share shape (B, C, H, W).")
        if min(prediction.shape[-2:]) < self.window_size:
            raise ValueError("Images are smaller than the SSIM window.")

        channels = prediction.shape[1]
        window = self.window.to(
            device=prediction.device, dtype=prediction.dtype
        ).expand(channels, 1, -1, -1)

        mean_pred = F.conv2d(prediction, window, groups=channels)
        mean_target = F.conv2d(target, window, groups=channels)
        mean_pred_sq = mean_pred.square()
        mean_target_sq = mean_target.square()
        mean_cross = mean_pred * mean_target

        variance_pred = (
            F.conv2d(prediction.square(), window, groups=channels) - mean_pred_sq
        )
        variance_target = (
            F.conv2d(target.square(), window, groups=channels) - mean_target_sq
        )
        covariance = F.conv2d(prediction * target, window, groups=channels) - mean_cross

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        numerator = (2.0 * mean_cross + c1) * (2.0 * covariance + c2)
        denominator = (mean_pred_sq + mean_target_sq + c1) * (
            variance_pred + variance_target + c2
        )
        return (numerator / denominator).flatten(1).mean(dim=1)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self.per_image(prediction, target).mean()


@dataclass(frozen=True)
class LossTerms:
    total: Tensor
    mse: Tensor
    ssim: Tensor
    mse_per_image: Tensor
    ssim_per_image: Tensor


class ReconstructionLoss(nn.Module):
    """Combine pixel-wise MSE with single-band SSIM."""

    def __init__(self, ssim_weight: float = 0.1) -> None:
        super().__init__()
        self.ssim_weight = ssim_weight
        self.ssim_metric = StructuralSimilarity(window_size=11, data_range=1.0)

    def forward(self, prediction: Tensor, target: Tensor) -> LossTerms:
        mse_per_image = (
            F.mse_loss(prediction, target, reduction="none").flatten(1).mean(dim=1)
        )
        ssim_per_image = self.ssim_metric.per_image(prediction, target)
        mse = mse_per_image.mean()
        ssim = ssim_per_image.mean()
        total = mse - self.ssim_weight * ssim
        return LossTerms(total, mse, ssim, mse_per_image, ssim_per_image)
