from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..architectures import FFHQALAE
from ..data.ffhq_latent import FFHQ_CLASS_NAMES
from ..utils import default_runtime_roots, normalize_path, resolve_latest_alias

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _split_dir(root: Path, split: str) -> Path:
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing FFHQ split directory: {split_dir}")
    return split_dir


def _validate_class_dirs(split_dir: Path) -> None:
    present = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    expected = sorted(FFHQ_CLASS_NAMES)
    missing = [name for name in expected if name not in present]
    extra = [name for name in present if name not in expected]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(f"Invalid FFHQ class directory layout under {split_dir}: {', '.join(details)}")


def _iter_images(class_dir: Path) -> list[Path]:
    images = sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No images found in {class_dir}")
    return images


def _load_image_tensor(path: Path, *, expected_size: int, expected_channels: int) -> torch.Tensor:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB image, received mode={image.mode!r} for {path}")
        if image.size != (expected_size, expected_size):
            raise ValueError(f"Expected image size {(expected_size, expected_size)}, received {image.size} for {path}")
        array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels for {path}, received shape={array.shape}")
    array = array.transpose(2, 0, 1)
    return torch.from_numpy(array).float() / 127.5 - 1.0


@torch.no_grad()
def _encode_paths(
    model: FFHQALAE,
    image_paths: list[Path],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch = torch.stack(
            [
                _load_image_tensor(
                    path,
                    expected_size=model.image_size,
                    expected_channels=model.config.channels,
                )
                for path in batch_paths
            ],
            dim=0,
        ).to(device)
        parts.append(model.encode(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


def encode_ffhq_image_root(
    *,
    alae_ckpt: str | Path,
    image_root: str | Path,
    device: str = "cuda",
    batch_size: int = 8,
    output_dir: str | Path | None = None,
) -> Path:
    runtime = default_runtime_roots()
    model = FFHQALAE.load_project_checkpoint(alae_ckpt)
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    root_dir = resolve_latest_alias(normalize_path(image_root, runtime_roots=runtime, resolve_latest=False))
    output_root = (
        normalize_path(output_dir, runtime_roots=runtime, resolve_latest=False)
        if output_dir is not None
        else normalize_path("{data_root}/ffhq_latents_6class", runtime_roots=runtime, resolve_latest=False)
    )
    output_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        _validate_class_dirs(_split_dir(root_dir, split))

    for split in ("train", "test"):
        split_dir = _split_dir(root_dir, split)
        payload = {
            class_name: _encode_paths(
                model,
                _iter_images(split_dir / class_name),
                device=torch_device,
                batch_size=int(batch_size),
            )
            for class_name in FFHQ_CLASS_NAMES
        }
        np.savez(output_root / f"{split}_latents_by_class.npz", **payload)
    return output_root


def _parse_args() -> argparse.Namespace:
    runtime_roots = default_runtime_roots()
    parser = argparse.ArgumentParser(description="Encode FFHQ train/test class folders into latent NPZ files.")
    parser.add_argument("--alae-ckpt", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default=f"{runtime_roots['data_root']}/ffhq_latents_6class")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = encode_ffhq_image_root(
        alae_ckpt=args.alae_ckpt,
        image_root=args.image_root,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
