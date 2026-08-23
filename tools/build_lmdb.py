"""Convert wavelength-organized image pairs to LMDB."""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path

import lmdb
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


WAVELENGTH_MIN_NM = 450.0
WAVELENGTH_MAX_NM = 700.0
IMAGE_EXTENSIONS = {".bmp", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImagePair:
    wavelength_nm: float
    measurement: Path
    truth: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a WALAR LMDB split.")
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=490)
    parser.add_argument("--map-size-gb", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return {
        path.relative_to(directory).as_posix().casefold(): path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def find_pairs(sources: list[Path]) -> list[ImagePair]:
    pairs = []
    for source in sources:
        source = source.expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)

        wavelength_dirs = []
        for directory in source.iterdir():
            try:
                wavelength = float(directory.name)
            except ValueError:
                continue
            if (
                directory.is_dir()
                and WAVELENGTH_MIN_NM <= wavelength <= WAVELENGTH_MAX_NM
            ):
                wavelength_dirs.append((wavelength, directory))

        for wavelength, directory in sorted(wavelength_dirs):
            measurements = image_files(directory / "measurement")
            targets = image_files(directory / "truth")
            if measurements.keys() != targets.keys():
                raise RuntimeError(f"Unmatched image names in {directory}")
            pairs.extend(
                ImagePair(wavelength, measurements[name], targets[name])
                for name in sorted(measurements)
            )

    if not pairs:
        raise RuntimeError("No image pairs were found.")
    return pairs


def read_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32)
    if array.shape != (image_size, image_size):
        raise ValueError(f"Expected {image_size} x {image_size}: {path}")
    return torch.from_numpy(array / 255.0).unsqueeze(0)


def serialize(pair: ImagePair, image_size: int) -> bytes:
    wavelength = (pair.wavelength_nm - WAVELENGTH_MIN_NM) / (
        WAVELENGTH_MAX_NM - WAVELENGTH_MIN_NM
    )
    sample = {
        "measurement": read_image(pair.measurement, image_size),
        "truth": read_image(pair.truth, image_size),
        "wavelength": torch.tensor(wavelength, dtype=torch.float32),
        "wavelength_nm": pair.wavelength_nm,
        "filename": pair.measurement.name,
    }
    buffer = io.BytesIO()
    torch.save(sample, buffer)
    return buffer.getvalue()


def prepare_output(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    database_files = [path / "data.mdb", path / "lock.mdb"]
    if any(file.exists() for file in database_files) and not overwrite:
        raise FileExistsError(f"LMDB already exists in {path}; use --overwrite")
    for file in database_files:
        if file.exists():
            file.unlink()


def main() -> None:
    args = parse_args()
    pairs = find_pairs(args.source)
    output = args.output.expanduser()
    prepare_output(output, args.overwrite)

    first_sample = serialize(pairs[0], args.image_size)
    estimated_size = int(len(first_sample) * len(pairs) * 1.2 + 64 * 1024**2)
    map_size = (
        int(args.map_size_gb * 1024**3)
        if args.map_size_gb
        else max(estimated_size, 256 * 1024**2)
    )
    metadata = {
        "sample_count": len(pairs),
        "image_shape": [1, args.image_size, args.image_size],
        "wavelength_range_nm": [WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM],
        "fields": [
            "measurement",
            "truth",
            "wavelength",
            "wavelength_nm",
            "filename",
        ],
    }

    environment = lmdb.open(
        str(output), map_size=map_size, meminit=False, map_async=True
    )
    transaction = environment.begin(write=True)
    try:
        for index, pair in enumerate(tqdm(pairs, desc="LMDB")):
            value = first_sample if index == 0 else serialize(pair, args.image_size)
            transaction.put(f"{index:08d}".encode("ascii"), value)
            if (index + 1) % 500 == 0:
                transaction.commit()
                transaction = environment.begin(write=True)

        transaction.put(b"__len__", str(len(pairs)).encode("ascii"))
        transaction.put(b"__meta__", json.dumps(metadata).encode("utf-8"))
        transaction.commit()
        transaction = None
        environment.sync()
    finally:
        if transaction is not None:
            transaction.abort()
        environment.close()

    print(f"Saved {len(pairs):,} pairs to {output}")


if __name__ == "__main__":
    main()
