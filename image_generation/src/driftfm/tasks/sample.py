"""Sample a generator checkpoint without loading its training dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..evaluation.latent import load_generation_stack, sample_latents_by_class
from ..training.checkpoints import load_checkpoint
from ..utils import configure_determinism, make_torch_generator, normalize_path


def sample_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_dir: str | Path,
    num_steps: int = 1,
    samples_per_class: int = 8,
    seed: int = 42,
    device: str = "cpu",
    batch_size: int = 64,
    decoder_checkpoint: str | Path | None = None,
) -> Path:
    if min(num_steps, samples_per_class, batch_size) <= 0:
        raise ValueError("steps, samples_per_class, and batch_size must be positive")
    checkpoint_path = normalize_path(checkpoint_path, resolve_latest=True)
    checkpoint = load_checkpoint(checkpoint_path)
    mode = checkpoint["resolved_architecture"]["params"].get("conditioning_mode", "class")
    if mode != "class":
        raise ValueError("This release supports class-conditional DFM checkpoints.")
    configure_determinism(seed, deterministic=True, device=device)
    torch_device = torch.device(device)
    model, method = load_generation_stack(checkpoint, torch_device)
    info = checkpoint["dataset_info"]
    class_names = info["class_names"]
    generated = sample_latents_by_class(
        model, method, class_names=class_names, n_per_class=samples_per_class,
        latent_dim=int(info["data_dim"]), device=torch_device, gen_batch=batch_size,
        num_sampling_steps=num_steps,
        generator=make_torch_generator(torch_device, seed=seed),
    ).fake_by_class
    output = normalize_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "samples.npz", **generated)
    metadata = {
        "checkpoint": str(checkpoint_path), "seed": seed, "nfe": num_steps,
        "samples_per_class": samples_per_class, "class_names": class_names,
        "weights": "ema" if checkpoint.get("ema_state") is not None else "model",
        "decoder_checkpoint": None if decoder_checkpoint is None else str(normalize_path(decoder_checkpoint)),
    }
    if decoder_checkpoint is not None:
        from torchvision.utils import save_image
        dataset_name = checkpoint["experiment"]["dataset"]["name"]
        if dataset_name == "mnist_latent":
            from ..evaluation.mnist import _load_ae
            decoder = _load_ae(normalize_path(decoder_checkpoint), torch_device)
        elif dataset_name == "ffhq_latent":
            from ..architectures import FFHQALAE
            decoder = FFHQALAE.load_project_checkpoint(decoder_checkpoint, map_location="cpu").to(torch_device).eval()
        else:
            raise ValueError(f"No image decoder configured for dataset {dataset_name!r}")
        latents = torch.from_numpy(np.concatenate(list(generated.values())))
        images = []
        with torch.no_grad():
            for chunk in latents.split(batch_size):
                chunk = chunk.to(torch_device)
                decoded = decoder.decode(chunk) if dataset_name == "mnist_latent" else decoder.decode(chunk, noise=False)
                images.append(decoded.cpu() if dataset_name == "mnist_latent" else (decoded.cpu() + 1) / 2)
        save_image(torch.cat(images).clamp(0, 1), output / "samples.png", nrow=samples_per_class)
        metadata["decoder_noise"] = False
    (output / "samples.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output / "samples.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--decoder-checkpoint", default=None, help="MNIST AE or imported FFHQ ALAE checkpoint; also writes an image grid.")
    args = parser.parse_args()
    for steps in args.steps:
        result = sample_checkpoint(
            args.checkpoint, output_dir=Path(args.output_dir) / f"steps_{steps}",
            num_steps=steps, samples_per_class=args.samples_per_class, seed=args.seed,
            device=args.device, batch_size=args.batch_size, decoder_checkpoint=args.decoder_checkpoint,
        )
        print(result)


if __name__ == "__main__":
    main()
