from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ..architectures import MnistConvAE
from ..utils import default_runtime_roots, normalize_path, resolve_latest_alias


@torch.no_grad()
def _encode_split(model: MnistConvAE, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    latents, labels = [], []
    for images, batch_labels in loader:
        z = model.encode(images.to(device))
        latents.append(z.cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(latents, axis=0), np.concatenate(labels, axis=0)


def main() -> None:
    runtime_roots = default_runtime_roots()
    parser = argparse.ArgumentParser(description="Encode MNIST train/test splits into latent files.")
    parser.add_argument("--ae-ckpt", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-dir", default=runtime_roots["mnist_root"])
    parser.add_argument("--download", action="store_true", help="Download MNIST if it is missing.")
    args = parser.parse_args()

    device = torch.device(args.device)
    mnist_root = normalize_path(args.data_dir)
    checkpoint_path = resolve_latest_alias(
        normalize_path(args.ae_ckpt, runtime_roots=runtime_roots, resolve_latest=False)
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MnistConvAE(latent_dim=int(checkpoint["latent_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint["model"])
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=mnist_root, train=True, download=args.download, transform=transform)
    test_ds = datasets.MNIST(root=mnist_root, train=False, download=args.download, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_latents, train_labels = _encode_split(model, train_loader, device)
    test_latents, test_labels = _encode_split(model, test_loader, device)

    output_dir = (
        normalize_path(args.output_dir, runtime_roots=runtime_roots, resolve_latest=False)
        if args.output_dir is not None
        else checkpoint_path.resolve().parent
    )
    output_dir = resolve_latest_alias(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_latents.npy", train_latents)
    np.save(output_dir / "train_labels.npy", train_labels)
    np.save(output_dir / "test_latents.npy", test_latents)
    np.save(output_dir / "test_labels.npy", test_labels)
    print(output_dir)


if __name__ == "__main__":
    main()
