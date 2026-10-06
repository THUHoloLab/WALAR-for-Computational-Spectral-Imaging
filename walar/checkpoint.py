"""Checkpoint utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary.")
    return checkpoint


def load_model(model: nn.Module, checkpoint: Mapping[str, Any]) -> None:
    state = None
    for key in ("model", "model_state_dict", "net_cofficient_state_dict"):
        if isinstance(checkpoint.get(key), Mapping):
            state = checkpoint[key]
            break
    if (
        state is None
        and checkpoint
        and all(torch.is_tensor(value) for value in checkpoint.values())
    ):
        state = checkpoint
    if state is None:
        raise KeyError("Model weights were not found in the checkpoint.")
    if all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state)


def load_optimizer(
    optimizer: torch.optim.Optimizer, checkpoint: Mapping[str, Any]
) -> bool:
    for key in (
        "optimizer",
        "optimizer_state_dict",
        "optimizer_cofficient_state_dict",
    ):
        if isinstance(checkpoint.get(key), Mapping):
            optimizer.load_state_dict(checkpoint[key])
            return True
    return False


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_psnr: float,
    model_config: Mapping[str, int | float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_psnr": best_psnr,
            "model_config": dict(model_config),
        },
        path,
    )
