from __future__ import annotations

import argparse
import json
import random
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from ..architectures import MnistConvAE
from ..config import ExperimentConfig, coerce_experiment_config
from ..data import Conditioning
from ..data.mnist_latent import MNIST_CLASS_NAMES
from ..training.checkpoints import load_checkpoint
from ..utils import (
    collect_determinism_metadata,
    configure_determinism,
    default_runtime_roots,
    resolve_runtime_config,
    make_data_loader_generator,
    normalize_path,
    resolve_latest_alias,
    seed_data_loader_worker,
)
from .base import CheckpointEvaluator, EvaluationResult
from .latent import (
    add_projection_artifacts_to_payload,
    load_generation_stack,
    make_torch_generator,
    maybe_write_projection_artifacts,
    resolve_evaluation_seed,
    sample_latents_by_class,
    save_sampled_latents_by_class,
)
from .ot import compute_ot_distance

MNIST_CLASS_GRID_FILENAME = "mnist_class_grid.png"


def _default_classifier_ckpt(runtime_roots: dict[str, str]) -> str:
    return str(Path(runtime_roots["runs_root"]) / "mnist_clf" / "mnist_clf.pt")


class MNISTClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MnistEvaluator(CheckpointEvaluator):
    name = "mnist"
    supported_metrics = ("latent_ot", "image_ot", "accuracy", "pca", "tsne", "umap", "lda")

    def evaluate(
        self,
        checkpoint_path: str | Path,
        *,
        experiment: ExperimentConfig,
        checkpoint: dict[str, Any],
        output_dir: str | Path,
        num_sampling_steps: int,
    ) -> EvaluationResult:
        metrics = experiment.evaluation.metrics or list(self.supported_metrics)
        self.validate_metrics(metrics)
        params = dict(experiment.evaluation.params)
        runtime_roots = resolve_runtime_config(experiment.runtime)
        mnist_root = runtime_roots["mnist_root"]
        device_name = str(params.get("device", experiment.trainer.device))
        device = torch.device(device_name)

        if "ae_ckpt" not in params:
            raise ValueError("MNIST evaluation requires evaluation.params.ae_ckpt")

        ae_ckpt = resolve_latest_alias(params["ae_ckpt"])
        ae = _load_ae(ae_ckpt, device)
        model, method = load_generation_stack(checkpoint, device)
        resolved_seed = resolve_evaluation_seed(experiment)
        configure_determinism(
            resolved_seed,
            deterministic=bool(experiment.trainer.deterministic),
            device=device,
        )
        determinism = collect_determinism_metadata(
            resolved_seed,
            deterministic=bool(experiment.trainer.deterministic),
            device=device,
        )
        latent_dim = int(checkpoint["dataset_info"]["data_dim"])
        num_classes = int(checkpoint["dataset_info"].get("num_classes") or 10)
        class_grid_samples_per_class = int(experiment.evaluation.sample_count_per_class)
        class_grid_path, class_grid_rows = _save_class_grid(
            ae,
            model,
            method,
            latent_dim=latent_dim,
            num_classes=num_classes,
            samples_per_class=class_grid_samples_per_class,
            output_dir=output_dir,
            device=device,
            num_sampling_steps=num_sampling_steps,
            generator=make_torch_generator(device, seed=resolved_seed + 1),
        )

        latent_paths = _resolve_latent_paths(params, ae_ckpt)
        real_latents, real_latent_labels = _load_real_latents_or_encode(
            ae,
            latent_paths=latent_paths,
            mnist_root=mnist_root,
            device=device,
        )

        need_images = "image_ot" in metrics or "accuracy" in metrics
        real_images = None
        real_image_labels = None
        if need_images:
            real_images, real_image_labels = _load_real_images(
                mnist_root=mnist_root
            )

        classifier = None
        if "accuracy" in metrics:
            classifier = _load_or_train_classifier(
                params.get("classifier_ckpt", _default_classifier_ckpt(runtime_roots)),
                mnist_root=mnist_root,
                device=device,
                seed=resolved_seed,
            )

        n_per_class = int(params.get("n_per_class", 100))
        solver = str(params.get("solver", "emd"))
        metric = str(params.get("metric", "l2_sq"))
        sinkhorn_reg = float(params.get("sinkhorn_reg", 0.05))
        ot_iters = int(params.get("ot_iters", 200000))
        gen_batch = int(params.get("gen_batch", 256))

        real_by_class = {
            class_name: real_latents[np.where(real_latent_labels == class_index)[0][:n_per_class]].astype(np.float32)
            for class_index, class_name in enumerate(MNIST_CLASS_NAMES)
        }
        sampled = sample_latents_by_class(
            model,
            method,
            class_names=MNIST_CLASS_NAMES,
            n_per_class=n_per_class,
            latent_dim=latent_dim,
            device=device,
            gen_batch=gen_batch,
            num_sampling_steps=num_sampling_steps,
            generator=make_torch_generator(device, seed=resolved_seed),
        )
        sampled_latents_path = save_sampled_latents_by_class(
            output_dir,
            class_names=MNIST_CLASS_NAMES,
            real_by_class=real_by_class,
            fake_by_class=sampled.fake_by_class,
        )
        projection_artifacts = maybe_write_projection_artifacts(
            metrics=metrics,
            output_dir=output_dir,
            class_names=MNIST_CLASS_NAMES,
            fake_by_class=sampled.fake_by_class,
            real_by_class=real_by_class,
            random_state=resolved_seed,
        )

        per_class_results: list[dict[str, float | int]] = []
        for class_index, class_name in enumerate(MNIST_CLASS_NAMES):
            generated_latents_np = sampled.fake_by_class[class_name]
            row: dict[str, float | int] = {"class": class_index}
            if "latent_ot" in metrics:
                row["latent_ot"] = compute_ot_distance(
                    generated_latents_np,
                    real_by_class[class_name],
                    solver=solver,
                    metric=metric,
                    sinkhorn_reg=sinkhorn_reg,
                    ot_iters=ot_iters,
                )

            generated_images = None
            if need_images:
                with torch.no_grad():
                    generated_images = ae.decode(torch.from_numpy(generated_latents_np).to(device)).cpu()

            if "image_ot" in metrics:
                assert real_images is not None and real_image_labels is not None and generated_images is not None
                image_indices = np.where(real_image_labels == class_index)[0][:n_per_class]
                row["image_ot"] = compute_ot_distance(
                    generated_images.view(generated_images.shape[0], -1).numpy(),
                    real_images[image_indices],
                    solver=solver,
                    metric=metric,
                    sinkhorn_reg=sinkhorn_reg,
                    ot_iters=ot_iters,
                )

            if "accuracy" in metrics:
                assert classifier is not None and generated_images is not None
                predicted = classifier(generated_images.to(device)).argmax(dim=1).cpu()
                row["accuracy"] = float((predicted == class_index).float().mean().item())

            per_class_results.append(row)

        summary: dict[str, float] = {}
        payload: dict[str, Any] = {
            "checkpoint": str(resolve_latest_alias(checkpoint_path)),
            "metrics": metrics,
            "num_sampling_steps": num_sampling_steps,
            "resolved_seed": resolved_seed,
            "determinism": determinism,
            "per_class": per_class_results,
            "sampled_latents_npz": str(sampled_latents_path),
            "class_grid_path": str(class_grid_path),
            "class_grid_samples_per_class": class_grid_samples_per_class,
            "class_grid_rows": class_grid_rows,
        }
        add_projection_artifacts_to_payload(payload, projection_artifacts)
        if "latent_ot" in metrics:
            summary["mean_latent_ot"] = float(np.mean([float(row["latent_ot"]) for row in per_class_results]))
            payload["mean_latent_ot"] = summary["mean_latent_ot"]
        if "image_ot" in metrics:
            summary["mean_image_ot"] = float(np.mean([float(row["image_ot"]) for row in per_class_results]))
            payload["mean_image_ot"] = summary["mean_image_ot"]
        if "accuracy" in metrics:
            summary["mean_accuracy"] = float(np.mean([float(row["accuracy"]) for row in per_class_results]))
            payload["mean_accuracy"] = summary["mean_accuracy"]

        output_path = Path(output_dir) / "mnist_eval.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return EvaluationResult(output_path=output_path, summary=summary)


def _parse_args() -> argparse.Namespace:
    runtime_roots = default_runtime_roots()
    classifier_ckpt = _default_classifier_ckpt(runtime_roots)
    parser = argparse.ArgumentParser(description="Evaluate MNIST latent Drift Flow Matching checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ae-ckpt", required=True)
    parser.add_argument("--latent-root", default=None)
    parser.add_argument("--test-latents-path", default=None)
    parser.add_argument("--test-labels-path", default=None)
    parser.add_argument("--classifier-ckpt", default=classifier_ckpt)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-per-class", type=int, default=100)
    parser.add_argument("--solver", choices=["emd", "sinkhorn"], default="emd")
    parser.add_argument("--metric", choices=["l2", "l2_sq"], default="l2_sq")
    parser.add_argument("--sinkhorn-reg", type=float, default=0.05)
    parser.add_argument("--ot-iters", type=int, default=200000)
    parser.add_argument("--gen-batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-sampling-steps", nargs="+", type=int, default=[1, 2, 5, 10, 20, 50])
    parser.add_argument("--metrics", nargs="*", default=list(MnistEvaluator.supported_metrics))
    parser.add_argument("--no-deterministic", action="store_true")
    return parser.parse_args()


@contextmanager
def _temporary_random_seed(seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _load_ae(path: str | Path, device: torch.device) -> MnistConvAE:
    checkpoint = load_checkpoint(path, map_location="cpu")
    model = MnistConvAE(latent_dim=int(checkpoint["latent_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint["model"])
    model.eval()
    return model


def _resolve_latent_paths(params: dict[str, Any], ae_ckpt: str | Path) -> tuple[Path | None, Path | None]:
    if params.get("test_latents_path") and params.get("test_labels_path"):
        return resolve_latest_alias(params["test_latents_path"]), resolve_latest_alias(params["test_labels_path"])
    if params.get("latent_root") is not None:
        root = resolve_latest_alias(params["latent_root"])
        return root / "test_latents.npy", root / "test_labels.npy"
    ae_dir = resolve_latest_alias(ae_ckpt).resolve().parent
    latents_path = ae_dir / "test_latents.npy"
    labels_path = ae_dir / "test_labels.npy"
    if latents_path.exists() and labels_path.exists():
        return latents_path, labels_path
    return None, None


def _load_real_latents_or_encode(
    ae: MnistConvAE,
    *,
    latent_paths: tuple[Path | None, Path | None],
    mnist_root: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    latents_path, labels_path = latent_paths
    if latents_path is not None and labels_path is not None and latents_path.exists() and labels_path.exists():
        return np.load(latents_path), np.load(labels_path)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root=mnist_root, train=False, download=False, transform=transform)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    all_latents, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            latents = ae.encode(images.to(device))
            all_latents.append(latents.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_latents, axis=0), np.concatenate(all_labels, axis=0)


def _load_real_images(*, mnist_root: str) -> tuple[np.ndarray, np.ndarray]:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root=mnist_root, train=False, download=False, transform=transform)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    images, labels = [], []
    for batch_images, batch_labels in loader:
        images.append(batch_images.view(batch_images.shape[0], -1).numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)


def _save_class_grid(
    ae: MnistConvAE,
    model: torch.nn.Module,
    method: Any,
    *,
    latent_dim: int,
    num_classes: int,
    samples_per_class: int,
    output_dir: str | Path,
    device: torch.device,
    num_sampling_steps: int,
    generator: torch.Generator,
) -> tuple[Path, list[int]]:
    class_rows = list(range(num_classes))
    labels = torch.repeat_interleave(
        torch.tensor(class_rows, dtype=torch.long),
        repeats=samples_per_class,
    )
    noise = torch.randn((labels.shape[0], latent_dim), device=device, generator=generator)
    conditioning = Conditioning(
        class_labels=labels.to(device),
    )
    with torch.no_grad():
        generated_latents = method.sample(
            model,
            noise,
            conditioning,
            num_steps=num_sampling_steps,
        )
        generated_images = ae.decode(generated_latents.to(device)).cpu().clamp(0.0, 1.0)

    grid_path = Path(output_dir) / MNIST_CLASS_GRID_FILENAME
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(generated_images, grid_path, nrow=samples_per_class)
    return grid_path.resolve(), class_rows


def _load_or_train_classifier(
    path: str | Path,
    *,
    mnist_root: str,
    device: torch.device,
    seed: int,
) -> MNISTClassifier:
    checkpoint_path = resolve_latest_alias(path)
    classifier = MNISTClassifier().to(device)
    if checkpoint_path.exists():
        classifier.load_state_dict(load_checkpoint(checkpoint_path, map_location=device))
        classifier.eval()
        return classifier

    with _temporary_random_seed(seed):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST(root=mnist_root, train=True, download=False, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=True,
            num_workers=0,
            generator=make_data_loader_generator(seed=seed),
            worker_init_fn=seed_data_loader_worker,
        )
        classifier = MNISTClassifier().to(device)
        optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        classifier.train()
        for _ in range(2):
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                loss = loss_fn(classifier(images), labels)
                loss.backward()
                optimizer.step()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.state_dict(), checkpoint_path)
    classifier.eval()
    return classifier


def evaluate_mnist_checkpoint(
    checkpoint_path: str | Path,
    *,
    ae_ckpt: str | Path,
    latent_root: str | None = None,
    test_latents_path: str | None = None,
    test_labels_path: str | None = None,
    classifier_ckpt: str | Path | None = None,
    device: str = "cuda",
    n_per_class: int = 100,
    solver: str = "emd",
    metric: str = "l2_sq",
    sinkhorn_reg: float = 0.05,
    ot_iters: int = 200000,
    gen_batch: int = 256,
    seed: int | None = None,
    deterministic: bool | None = None,
    num_sampling_steps: list[int] | None = None,
    metrics: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    resolved_checkpoint_path = normalize_path(
        checkpoint_path,
        resolve_latest=False,
    )
    checkpoint = load_checkpoint(resolved_checkpoint_path)
    experiment = coerce_experiment_config(checkpoint["experiment"])
    if deterministic is not None:
        experiment.trainer.deterministic = bool(deterministic)
    runtime_roots = resolve_runtime_config(experiment.runtime)
    classifier_ckpt = _default_classifier_ckpt(runtime_roots) if classifier_ckpt is None else classifier_ckpt
    experiment.evaluation.evaluator = "mnist"
    experiment.evaluation.metrics = list(metrics or MnistEvaluator.supported_metrics)
    experiment.evaluation.params = {
        "ae_ckpt": str(normalize_path(ae_ckpt, runtime_roots=runtime_roots, resolve_latest=True)),
        "latent_root": None
        if latent_root is None
        else str(normalize_path(latent_root, runtime_roots=runtime_roots, resolve_latest=True)),
        "test_latents_path": None
        if test_latents_path is None
        else str(normalize_path(test_latents_path, runtime_roots=runtime_roots, resolve_latest=True)),
        "test_labels_path": None
        if test_labels_path is None
        else str(normalize_path(test_labels_path, runtime_roots=runtime_roots, resolve_latest=True)),
        "classifier_ckpt": str(
            normalize_path(classifier_ckpt, runtime_roots=runtime_roots, resolve_latest=True)
        ),
        "device": device,
        "n_per_class": n_per_class,
        "solver": solver,
        "metric": metric,
        "sinkhorn_reg": sinkhorn_reg,
        "ot_iters": ot_iters,
        "gen_batch": gen_batch,
    }
    if seed is not None:
        experiment.evaluation.params["seed"] = int(seed)
    experiment.evaluation = replace(
        experiment.evaluation,
        num_sampling_steps=[1, 2, 5, 10, 20, 50] if num_sampling_steps is None else list(num_sampling_steps),
    )
    from .registry import evaluate_checkpoint

    result = evaluate_checkpoint(
        resolved_checkpoint_path,
        experiment=experiment,
        output_dir=(
            normalize_path(output_dir, runtime_roots=runtime_roots, resolve_latest=False)
            if output_dir is not None
            else resolve_latest_alias(resolved_checkpoint_path).resolve().parent
        ),
    )
    return result.output_path


def main() -> None:
    args = _parse_args()
    output_path = evaluate_mnist_checkpoint(
        args.checkpoint,
        ae_ckpt=args.ae_ckpt,
        latent_root=args.latent_root,
        test_latents_path=args.test_latents_path,
        test_labels_path=args.test_labels_path,
        classifier_ckpt=args.classifier_ckpt,
        device=args.device,
        n_per_class=args.n_per_class,
        solver=args.solver,
        metric=args.metric,
        sinkhorn_reg=args.sinkhorn_reg,
        ot_iters=args.ot_iters,
        gen_batch=args.gen_batch,
        seed=args.seed,
        deterministic=not args.no_deterministic,
        num_sampling_steps=args.num_sampling_steps,
        metrics=args.metrics,
    )
    print(output_path)
