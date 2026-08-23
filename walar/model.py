"""Wavelength-conditioned Transformer used by WALAR."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Size2D = int | tuple[int, int]


def _pair(value: Size2D) -> tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


def _same_padding(
    image_size: tuple[int, int], kernel_size: int, stride: int
) -> tuple[int, int, int, int]:
    """Return left, right, top, and bottom padding for full image coverage."""
    height, width = image_size
    pad_h = max(
        (math.ceil(height / stride) - 1) * stride + kernel_size - height,
        0,
    )
    pad_w = max(
        (math.ceil(width / stride) - 1) * stride + kernel_size - width,
        0,
    )
    return (
        pad_w // 2,
        pad_w - pad_w // 2,
        pad_h // 2,
        pad_h - pad_h // 2,
    )


class AutoPaddingUnfold(nn.Module):
    """Extract overlapping patches after symmetric automatic padding."""

    def __init__(self, image_size: Size2D, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.img_size = _pair(image_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = _same_padding(self.img_size, kernel_size, stride)
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)

        padded_h = self.img_size[0] + self.padding[2] + self.padding[3]
        padded_w = self.img_size[1] + self.padding[0] + self.padding[1]
        patches_h = (padded_h - kernel_size) // stride + 1
        patches_w = (padded_w - kernel_size) // stride + 1
        self.num_patches = patches_h * patches_w

    def forward(self, image: Tensor) -> Tensor:
        patches = self.unfold(F.pad(image, self.padding))
        return patches.transpose(1, 2)


class AutoPaddingFold(nn.Module):
    """Merge overlapping patches and average their pixel contributions."""

    def __init__(self, image_size: Size2D, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.img_size = _pair(image_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = _same_padding(self.img_size, kernel_size, stride)

        padded_h = self.img_size[0] + self.padding[2] + self.padding[3]
        padded_w = self.img_size[1] + self.padding[0] + self.padding[1]
        self.padded_size = (padded_h, padded_w)
        self.fold = nn.Fold(
            output_size=self.padded_size,
            kernel_size=kernel_size,
            stride=stride,
        )

        ones = torch.ones(1, 1, *self.img_size)
        patch_ones = nn.Unfold(kernel_size, stride=stride)(F.pad(ones, self.padding))
        overlap = self.fold(patch_ones).clamp_min(1.0)
        self.register_buffer("overlap", overlap)

        self.crop = (
            self.padding[2],
            padded_h - self.padding[3],
            self.padding[0],
            padded_w - self.padding[1],
        )

    def forward(self, patches: Tensor) -> Tensor:
        image = self.fold(patches.transpose(1, 2)) / self.overlap
        top, bottom, left, right = self.crop
        return image[:, :, top:bottom, left:right]


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, tokens: Tensor) -> Tensor:
        return self.fn(self.norm(tokens))


class FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return self.net(tokens)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, tokens: Tensor) -> Tensor:
        batch, length, _ = tokens.shape
        qkv = self.to_qkv(tokens).chunk(3, dim=-1)
        query, key, value = (
            tensor.reshape(batch, length, self.heads, self.dim_head).transpose(1, 2)
            for tensor in qkv
        )
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        return self.to_out(attended)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        layerscale_init: float = 1e-4,
    ) -> None:
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads, dim_head))
        self.ff = PreNorm(dim, FeedForward(dim))
        self.gamma1 = nn.Parameter(layerscale_init * torch.ones(dim))
        self.gamma2 = nn.Parameter(layerscale_init * torch.ones(dim))

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.gamma1 * self.attn(tokens)
        return tokens + self.gamma2 * self.ff(tokens)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        layerscale_init: float = 1e-4,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    layerscale_init=layerscale_init,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, tokens: Tensor) -> Tensor:
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens


class TASSIRNet(nn.Module):
    """Reconstruct one target band from a measurement and wavelength query."""

    def __init__(
        self,
        *,
        image_size: Size2D = 490,
        patch_size: int = 16,
        stride: int = 12,
        dim: int = 1024,
        depth: int = 7,
        heads: int = 8,
        dim_head: int = 128,
        layerscale_init: float = 1e-4,
    ) -> None:
        super().__init__()
        self.image_size = _pair(image_size)
        patch_h, patch_w = _pair(patch_size)
        if patch_h != patch_w:
            raise ValueError("TASSIRNet currently requires square patches.")

        self.img2patch = AutoPaddingUnfold(self.image_size, patch_size, stride)
        self.patch2img = AutoPaddingFold(self.image_size, patch_size, stride)

        # Measurement, wavelength, horizontal position, and vertical position.
        self.in_ch = 4
        patch_dim = self.in_ch * patch_h * patch_w
        self.proj = nn.Linear(patch_dim, dim)
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            layerscale_init=layerscale_init,
        )
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, patch_h * patch_w),
        )
        self._pos_cache: dict[tuple[object, ...], tuple[Tensor, Tensor]] = {}

    def _get_position_maps(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        key = (height, width, device, dtype)
        if key not in self._pos_cache:
            pos_x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(
                1, 1, 1, width
            )
            pos_y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(
                1, 1, height, 1
            )
            self._pos_cache[key] = (
                pos_x.expand(1, 1, height, width),
                pos_y.expand(1, 1, height, width),
            )
        return self._pos_cache[key]

    def forward(self, measurement: Tensor, wavelength: Tensor) -> Tensor:
        if measurement.ndim != 4 or measurement.shape[1] != 1:
            raise ValueError("measurement must have shape (B, 1, H, W).")

        batch, _, height, width = measurement.shape
        if (height, width) != self.image_size:
            raise ValueError(
                f"Expected image size {self.image_size}, got {(height, width)}."
            )
        if wavelength.numel() != batch:
            raise ValueError("wavelength must contain one value per image.")

        wavelength_map = wavelength.to(
            device=measurement.device, dtype=measurement.dtype
        ).reshape(batch, 1, 1, 1)
        wavelength_map = wavelength_map.expand(batch, 1, height, width)

        pos_x, pos_y = self._get_position_maps(
            height, width, measurement.device, measurement.dtype
        )
        model_input = torch.cat(
            [
                measurement,
                wavelength_map,
                pos_x.expand(batch, -1, -1, -1),
                pos_y.expand(batch, -1, -1, -1),
            ],
            dim=1,
        )

        tokens = self.proj(self.img2patch(model_input))
        tokens = self.transformer(tokens)
        return self.patch2img(self.head(tokens))
