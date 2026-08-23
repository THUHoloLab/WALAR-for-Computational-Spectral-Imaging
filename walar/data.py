"""LMDB reader for synchronized measurement--target pairs."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import lmdb
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _open_database(path: Path) -> lmdb.Environment:
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=512,
    )


class SpectralPairDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        # Relative paths avoid a python-lmdb issue under non-ASCII Windows roots.
        self.path = Path(path).expanduser()
        if not (self.path / "data.mdb").is_file():
            raise FileNotFoundError(f"Missing LMDB file: {self.path / 'data.mdb'}")

        environment = _open_database(self.path)
        with environment.begin() as transaction:
            length = transaction.get(b"__len__")
        environment.close()
        if length is None:
            raise RuntimeError(f"Missing __len__ record in {self.path}")

        self.length = int(length)
        self._environment: lmdb.Environment | None = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)

        if self._environment is None:
            self._environment = _open_database(self.path)
        key = f"{index:08d}".encode("ascii")
        with self._environment.begin() as transaction:
            raw = transaction.get(key)
        if raw is None:
            raise KeyError(f"Missing LMDB key: {key.decode()}")

        buffer = io.BytesIO(raw)
        try:
            sample: dict[str, Any] = torch.load(
                buffer, map_location="cpu", weights_only=False
            )
        except TypeError:
            buffer.seek(0)
            sample = torch.load(buffer, map_location="cpu")

        measurement = torch.as_tensor(sample["measurement"], dtype=torch.float32)
        wavelength = torch.as_tensor(sample["wavelength"], dtype=torch.float32)
        target = torch.as_tensor(sample["truth"], dtype=torch.float32)
        if measurement.ndim != 3 or measurement.shape != target.shape:
            raise ValueError(f"Invalid tensor shapes at sample {index}")
        return measurement, wavelength.reshape(()), target

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_environment"] = None
        return state

    def close(self) -> None:
        if self._environment is not None:
            self._environment.close()
            self._environment = None
