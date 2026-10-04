from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from ..architectures import MnistConvAE
from ..utils import (
    build_seed_hash_run_name,
    collect_determinism_metadata,
    configure_determinism,
    default_runtime_roots,
    load_json_file,
    normalize_path,
    make_data_loader_generator,
    seed_data_loader_worker,
    write_json_file,
    write_latest_run_marker,
)

_RUN_CONFIG_FILENAME = "run_config.json"
_RUN_METADATA_FILENAME = "run_metadata.json"


def _run_config_payload(args: argparse.Namespace, *, deterministic: bool) -> dict[str, int | float | str | bool]:
    return {
        "latent_dim": int(args.latent_dim),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "device": str(args.device),
        "seed": int(args.seed),
        "save_every": int(args.save_every),
        "deterministic": bool(deterministic),
    }


def _resolve_run_name(args: argparse.Namespace, *, deterministic: bool) -> str:
    if deterministic:
        return build_seed_hash_run_name(
            seed=args.seed,
            payload=_run_config_payload(args, deterministic=deterministic),
            prefix=f"latent{int(args.latent_dim)}",
        )
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_latent{args.latent_dim}"


def _assert_existing_run_matches_config(run_dir: Path, payload: dict[str, int | float | str | bool]) -> None:
    existing_payload = load_json_file(run_dir / _RUN_CONFIG_FILENAME)
    if existing_payload is not None and existing_payload != payload:
        raise ValueError(
            f"Deterministic MNIST AE run directory {run_dir} already exists with a different run config."
        )


def _run_metadata_payload(
    *,
    seed: int,
    deterministic: bool,
    device: str,
) -> dict[str, object]:
    return {
        "resolved_seed": int(seed),
        "determinism": collect_determinism_metadata(
            seed,
            deterministic=deterministic,
            device=device,
        ),
    }


def main() -> None:
    runtime_roots = default_runtime_roots()
    parser = argparse.ArgumentParser(description="Train the MNIST convolutional autoencoder.")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--data-dir", default=runtime_roots["mnist_root"])
    parser.add_argument("--output-dir", default=str(Path(runtime_roots["runs_root"]) / "mnist_ae"))
    parser.add_argument("--download", action="store_true", help="Download MNIST if it is missing.")
    args = parser.parse_args()

    deterministic = not args.no_deterministic
    configure_determinism(args.seed, deterministic=deterministic, device=args.device)
    device = torch.device(args.device)
    mnist_root = normalize_path(args.data_dir)
    run_root = normalize_path(args.output_dir)
    run_payload = _run_config_payload(args, deterministic=deterministic)
    run_dir = run_root / _resolve_run_name(args, deterministic=deterministic)
    if deterministic and run_dir.exists():
        _assert_existing_run_matches_config(run_dir, run_payload)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_file(run_dir / _RUN_CONFIG_FILENAME, run_payload)
    write_json_file(
        run_dir / _RUN_METADATA_FILENAME,
        _run_metadata_payload(
            seed=args.seed,
            deterministic=deterministic,
            device=str(args.device),
        ),
    )

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=mnist_root, train=True, download=args.download, transform=transform)
    test_ds = datasets.MNIST(root=mnist_root, train=False, download=args.download, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=make_data_loader_generator(seed=args.seed),
        worker_init_fn=seed_data_loader_worker,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MnistConvAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    log_path = run_dir / "logs.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for images, _ in train_loader:
            images = images.to(device)
            recon, _ = model(images)
            loss = F.mse_loss(recon, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * images.shape[0]
            train_count += images.shape[0]

        model.eval()
        test_loss = 0.0
        test_count = 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                recon, _ = model(images)
                loss = F.mse_loss(recon, images)
                test_loss += float(loss.item()) * images.shape[0]
                test_count += images.shape[0]

        record = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_count),
            "test_loss": test_loss / max(1, test_count),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            payload = {
                "latent_dim": args.latent_dim,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "test_loss": record["test_loss"],
            }
            checkpoint_path = run_dir / ("ae_final.pt" if epoch == args.epochs else f"ae_epoch{epoch}.pt")
            torch.save(payload, checkpoint_path)
            with torch.no_grad():
                sample_images = next(iter(test_loader))[0][:16].to(device)
                sample_recon, _ = model(sample_images)
                save_image(torch.cat([sample_images, sample_recon], dim=0), run_dir / f"recon_epoch{epoch}.png", nrow=16)

    write_latest_run_marker(run_root, run_dir)
    print(run_dir)


if __name__ == "__main__":
    main()
