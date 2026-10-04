from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..architectures import FFHQALAE
from ..config import ExperimentConfig, coerce_experiment_config
from ..data.ffhq_latent import FFHQ_CLASS_NAMES
from ..training.checkpoints import load_checkpoint
from ..utils import (
    collect_determinism_metadata,
    configure_determinism,
    default_runtime_roots,
    resolve_runtime_config,
    normalize_path,
    resolve_latest_alias,
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

_DEFAULT_FFHQ_EVAL_METRICS = ["latent_ot", "fid", "pca", "tsne", "umap", "lda"]
_DEFAULT_ALAE_CKPT = "{runs_root}/ffhq_alae/latest/alae_ffhq.pt"
_DEFAULT_REAL_SPLIT = "test"
_VALID_REAL_SPLITS = {"train", "test"}
_FID_ARTIFACTS_FILENAME = "fid_artifacts.json"


class FFHQEvaluator(CheckpointEvaluator):
    name = "ffhq"
    supported_metrics = ("latent_ot", "fid", "pca", "tsne", "umap", "lda")

    def evaluate(
        self,
        checkpoint_path: str | Path,
        *,
        experiment: ExperimentConfig,
        checkpoint: dict[str, Any],
        output_dir: str | Path,
        num_sampling_steps: int,
    ) -> EvaluationResult:
        metrics = experiment.evaluation.metrics or list(_DEFAULT_FFHQ_EVAL_METRICS)
        self.validate_metrics(metrics)
        params = dict(experiment.evaluation.params)

        device_name = str(params.get("device", experiment.trainer.device))
        device = torch.device(device_name)
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
        latent_dim = int(checkpoint["resolved_architecture"]["params"]["data_dim"])
        resolved_real_split, real_npz, real_input_source = _resolve_real_latent_source(
            experiment=experiment,
            params=params,
        )
        real_latents = _load_real_npz(real_npz)
        n_per_class = int(params.get("n_per_class", 128))
        solver = str(params.get("solver", "emd"))
        metric = str(params.get("metric", "l2_sq"))
        sinkhorn_reg = float(params.get("sinkhorn_reg", 0.05))
        ot_iters = int(params.get("ot_iters", 200000))
        gen_batch = int(params.get("gen_batch", 256))

        real_by_class = {
            class_name: real_latents[class_name][:n_per_class].astype(np.float32)
            for class_name in FFHQ_CLASS_NAMES
        }
        sampled = sample_latents_by_class(
            model,
            method,
            class_names=FFHQ_CLASS_NAMES,
            n_per_class=n_per_class,
            latent_dim=latent_dim,
            device=device,
            gen_batch=gen_batch,
            num_sampling_steps=num_sampling_steps,
            generator=make_torch_generator(device, seed=resolved_seed),
        )

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        sampled_latents_path = save_sampled_latents_by_class(
            output_root,
            class_names=FFHQ_CLASS_NAMES,
            real_by_class=real_by_class,
            fake_by_class=sampled.fake_by_class,
        )
        projection_artifacts = maybe_write_projection_artifacts(
            metrics=metrics,
            output_dir=output_root,
            class_names=FFHQ_CLASS_NAMES,
            fake_by_class=sampled.fake_by_class,
            real_by_class=real_by_class,
            random_state=resolved_seed,
        )

        per_class: list[dict[str, Any]] = []
        for class_name in FFHQ_CLASS_NAMES:
            row: dict[str, Any] = {"class_name": class_name}
            real = real_by_class[class_name]
            fake = sampled.fake_by_class[class_name]
            if "latent_ot" in metrics:
                row["latent_ot"] = compute_ot_distance(
                    fake,
                    real,
                    solver=solver,
                    metric=metric,
                    sinkhorn_reg=sinkhorn_reg,
                    ot_iters=ot_iters,
                )
            per_class.append(row)

        if "fid" in metrics:
            alae_model, alae_source = _load_configured_alae_model(device, params)
            real_base = _shared_real_images_root(output_root, params)
            fake_base = output_root / "fake_images"
            decode_batch = int(params.get("decode_batch", 8))
            fid_batch = int(params.get("fid_batch", 64))
            fid_num_workers = int(params.get("fid_num_workers", 0))
            save_size = int(params.get("save_size", 1024))
            reuse_images = bool(params.get("reuse_images", False))
            fid_decode_noise = bool(params.get("fid_decode_noise", True))
            allow_real_cache_reuse = bool(params.get("_allow_real_cache_reuse", reuse_images))
            real_image_counts = {class_name: int(real_by_class[class_name].shape[0]) for class_name in FFHQ_CLASS_NAMES}
            real_artifacts_metadata = _build_real_artifacts_metadata(
                resolved_real_split=resolved_real_split,
                resolved_real_npz=real_npz,
                real_input_source=real_input_source,
                alae_source=alae_source,
                save_size=save_size,
                n_per_class=n_per_class,
                per_class_image_counts=real_image_counts,
                fid_decode_noise=fid_decode_noise,
                resolved_seed=resolved_seed,
            )
            real_artifacts_path = real_base / _FID_ARTIFACTS_FILENAME
            if not (allow_real_cache_reuse and _can_reuse_real_image_cache(real_base, real_artifacts_metadata)):
                for class_index, class_name in enumerate(FFHQ_CLASS_NAMES):
                    _decode_and_save(
                        alae_model,
                        real_by_class[class_name],
                        real_base / class_name,
                        device=device,
                        batch_size=decode_batch,
                        save_size=save_size,
                        noise=fid_decode_noise,
                        generator=_make_decode_generator(
                            device,
                            seed=resolved_seed,
                            class_index=class_index,
                            stage_offset=0,
                        ),
                    )
                _write_json(real_artifacts_path, real_artifacts_metadata)
            fid_scores: dict[str, float] = {}
            for class_index, class_name in enumerate(FFHQ_CLASS_NAMES):
                real_dir = real_base / class_name
                fake_dir = fake_base / class_name
                _decode_and_save(
                    alae_model,
                    sampled.fake_by_class[class_name],
                    fake_dir,
                    device=device,
                    batch_size=decode_batch,
                    save_size=save_size,
                    noise=fid_decode_noise,
                    generator=_make_decode_generator(
                        device,
                        seed=resolved_seed,
                        class_index=class_index,
                        stage_offset=10_000,
                    ),
                )
                fid_scores[class_name] = _compute_fid(
                    real_dir,
                    fake_dir,
                    fid_batch=fid_batch,
                    fid_num_workers=fid_num_workers,
                    device=device,
                )
            for row in per_class:
                row["fid"] = fid_scores[row["class_name"]]
            fake_artifacts_metadata = {
                "kind": "ffhq_fake_images_v1",
                "checkpoint": str(resolve_latest_alias(checkpoint_path)),
                "num_sampling_steps": int(num_sampling_steps),
                "resolved_seed": int(resolved_seed),
                "fid_decode_noise": fid_decode_noise,
                "root_dir": str(fake_base.resolve()),
                "alae_source": dict(alae_source),
                "save_size": save_size,
                "n_per_class": n_per_class,
                "per_class_image_counts": _count_class_images(fake_base),
                "sampled_latents_npz": str(sampled_latents_path),
            }
            fid_artifacts = {
                "kind": "ffhq_fid_artifacts_v1",
                "fid_batch": fid_batch,
                "fid_num_workers": fid_num_workers,
                "real": {
                    **real_artifacts_metadata,
                    "root_dir": str(real_base.resolve()),
                    "artifacts_path": str(real_artifacts_path.resolve()),
                    "per_class_image_counts": _count_class_images(real_base),
                },
                "fake": fake_artifacts_metadata,
            }
            fid_artifacts_path = output_root / _FID_ARTIFACTS_FILENAME
            _write_json(fid_artifacts_path, fid_artifacts)

        summary: dict[str, float] = {}
        payload: dict[str, Any] = {
            "checkpoint": str(resolve_latest_alias(checkpoint_path)),
            "real_npz": str(real_npz),
            "resolved_real_npz": str(real_npz),
            "resolved_real_split": resolved_real_split,
            "real_input_source": real_input_source,
            "metrics": metrics,
            "num_sampling_steps": num_sampling_steps,
            "resolved_seed": resolved_seed,
            "determinism": determinism,
            "fid_decode_noise": bool(params.get("fid_decode_noise", True)),
            "sampled_latents_npz": str(sampled_latents_path),
            "per_class": per_class,
        }
        add_projection_artifacts_to_payload(payload, projection_artifacts)
        if "latent_ot" in metrics:
            summary["mean_latent_ot"] = float(np.mean([float(row["latent_ot"]) for row in per_class]))
            payload["mean_latent_ot"] = summary["mean_latent_ot"]
        if "fid" in metrics:
            summary["mean_fid"] = float(np.mean([float(row["fid"]) for row in per_class]))
            payload["mean_fid"] = summary["mean_fid"]
            payload["real_images_dir"] = str(real_base.resolve())
            payload["fake_images_dir"] = str(fake_base.resolve())
            payload["fid_artifacts_path"] = str(fid_artifacts_path.resolve())

        output_path = output_root / "ffhq_eval.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return EvaluationResult(output_path=output_path, summary=summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FFHQ latent Drift Flow Matching checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--real-npz", default=None)
    parser.add_argument("--real-split", choices=sorted(_VALID_REAL_SPLITS), default=_DEFAULT_REAL_SPLIT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-per-class", type=int, default=128)
    parser.add_argument("--solver", choices=["emd", "sinkhorn"], default="emd")
    parser.add_argument("--metric", choices=["l2", "l2_sq"], default="l2_sq")
    parser.add_argument("--sinkhorn-reg", type=float, default=0.05)
    parser.add_argument("--ot-iters", type=int, default=200000)
    parser.add_argument("--alae-ckpt", default=_DEFAULT_ALAE_CKPT)
    parser.add_argument("--decode-batch", type=int, default=8)
    parser.add_argument("--gen-batch", type=int, default=256)
    parser.add_argument("--fid-batch", type=int, default=64)
    parser.add_argument("--fid-num-workers", type=int, default=0)
    parser.add_argument("--save-size", type=int, default=1024)
    parser.add_argument("--reuse-images", action="store_true")
    parser.add_argument("--fid-decode-noise", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-sampling-steps", nargs="+", type=int, default=[1, 2, 5, 10, 20, 50])
    parser.add_argument("--metrics", nargs="*", default=list(_DEFAULT_FFHQ_EVAL_METRICS))
    parser.add_argument("--no-deterministic", action="store_true")
    return parser.parse_args()


def _make_decode_generator(
    device: torch.device,
    *,
    seed: int,
    class_index: int,
    stage_offset: int,
) -> torch.Generator:
    return make_torch_generator(device, seed=seed + stage_offset + class_index)


def _load_real_npz(path: str | Path) -> dict[str, np.ndarray]:
    arrays = np.load(path, allow_pickle=True)
    return {name: np.asarray(arrays[name], dtype=np.float32) for name in FFHQ_CLASS_NAMES}


def _normalize_real_split(value: Any | None) -> str:
    token = _DEFAULT_REAL_SPLIT if value in (None, "") else str(value).strip().lower()
    if token not in _VALID_REAL_SPLITS:
        supported = ", ".join(sorted(_VALID_REAL_SPLITS))
        raise ValueError(f"FFHQ evaluation real_split must be one of {{{supported}}}, got {value!r}")
    return token


def _resolve_real_latent_source(
    *,
    experiment: ExperimentConfig,
    params: dict[str, Any],
) -> tuple[str | None, Path, str]:
    runtime_roots = resolve_runtime_config(experiment.runtime)
    explicit_real_npz = params.get("real_npz")
    if explicit_real_npz not in (None, ""):
        resolved_explicit = resolve_latest_alias(
            normalize_path(
                str(explicit_real_npz),
                runtime_roots=runtime_roots,
                resolve_latest=True,
            )
        )
        if not resolved_explicit.exists():
            raise FileNotFoundError(f"Missing explicit FFHQ evaluation NPZ: {resolved_explicit}")
        return None, resolved_explicit, "explicit_npz"

    resolved_real_split = _normalize_real_split(params.get("real_split", _DEFAULT_REAL_SPLIT))
    dataset_npz = experiment.dataset.params.get(f"{resolved_real_split}_npz_path")
    if dataset_npz not in (None, ""):
        resolved_dataset_npz = resolve_latest_alias(
            normalize_path(
                str(dataset_npz),
                runtime_roots=runtime_roots,
                resolve_latest=True,
            )
        )
        if not resolved_dataset_npz.exists():
            raise FileNotFoundError(
                f"Missing FFHQ evaluation NPZ for real_split={resolved_real_split}: {resolved_dataset_npz}"
            )
        return resolved_real_split, resolved_dataset_npz, f"dataset_{resolved_real_split}"

    canonical = resolve_latest_alias(
        normalize_path(
            f"{{data_root}}/ffhq_latents_6class/{resolved_real_split}_latents_by_class.npz",
            runtime_roots=runtime_roots,
            resolve_latest=True,
        )
    )
    if not canonical.exists():
        raise FileNotFoundError(f"Missing canonical FFHQ evaluation NPZ for real_split={resolved_real_split}: {canonical}")
    return resolved_real_split, canonical, f"canonical_{resolved_real_split}"


def _shared_real_images_root(output_dir: str | Path, params: dict[str, Any]) -> Path:
    evaluation_root = (
        Path(str(params["_evaluation_root_dir"]))
        if params.get("_evaluation_root_dir") not in (None, "")
        else Path(output_dir)
    )
    return evaluation_root.resolve() / "real_images"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _count_class_images(root_dir: Path) -> dict[str, int]:
    return {class_name: _count_images(root_dir / class_name) for class_name in FFHQ_CLASS_NAMES}


def _build_real_artifacts_metadata(
    *,
    resolved_real_split: str | None,
    resolved_real_npz: Path,
    real_input_source: str,
    alae_source: dict[str, Any],
    save_size: int,
    n_per_class: int,
    per_class_image_counts: dict[str, int],
    fid_decode_noise: bool,
    resolved_seed: int,
) -> dict[str, Any]:
    return {
        "kind": "ffhq_real_images_v1",
        "resolved_real_split": resolved_real_split,
        "resolved_real_npz": str(resolved_real_npz),
        "real_input_source": real_input_source,
        "alae_source": dict(alae_source),
        "save_size": int(save_size),
        "n_per_class": int(n_per_class),
        "per_class_image_counts": {class_name: int(per_class_image_counts[class_name]) for class_name in FFHQ_CLASS_NAMES},
        "fid_decode_noise": bool(fid_decode_noise),
        "resolved_seed": int(resolved_seed) if fid_decode_noise else None,
    }


def _can_reuse_real_image_cache(root_dir: Path, expected_metadata: dict[str, Any]) -> bool:
    actual_metadata = _load_json(root_dir / _FID_ARTIFACTS_FILENAME)
    if actual_metadata != expected_metadata:
        return False
    return _count_class_images(root_dir) == expected_metadata["per_class_image_counts"]


def _load_internal_alae_model(device: torch.device, alae_ckpt: str):
    model = FFHQALAE.load_project_checkpoint(alae_ckpt, map_location="cpu")
    model = model.to(device)
    model.eval()
    return model


def _resolve_alae_source_descriptor(params: dict[str, Any]) -> dict[str, Any]:
    if params.get("alae_ckpt"):
        return {
            "type": "project_checkpoint",
            "alae_ckpt": str(resolve_latest_alias(str(params["alae_ckpt"]))),
        }
    raise ValueError(
        "FFHQ FID evaluation requires evaluation.params.alae_ckpt; "
        f"use an imported project checkpoint such as {_DEFAULT_ALAE_CKPT}"
    )


def _load_configured_alae_model(device: torch.device, params: dict[str, Any]):
    descriptor = _resolve_alae_source_descriptor(params)
    return _load_internal_alae_model(device, str(descriptor["alae_ckpt"])), descriptor


def _decode_latents(
    alae_model: FFHQALAE,
    latents: torch.Tensor,
    *,
    device: torch.device,
    noise: bool | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    x = latents.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        return alae_model.decode(x, noise=noise, generator=generator).detach().cpu()


def _decode_and_save(
    alae_model,
    latents: np.ndarray,
    output_dir: Path,
    *,
    device: torch.device,
    batch_size: int,
    save_size: int,
    noise: bool | str,
    generator: torch.Generator | None = None,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(latents)
    for start in range(0, tensor.shape[0], batch_size):
        batch = tensor[start : start + batch_size]
        images = _decode_latents(alae_model, batch, device=device, noise=noise, generator=generator)
        array = ((images.clamp(-1, 1) + 1.0) / 2.0 * 255.0).byte().permute(0, 2, 3, 1).numpy()
        for offset, image in enumerate(array):
            pil = Image.fromarray(image)
            if save_size != 1024:
                pil = pil.resize((save_size, save_size), Image.LANCZOS)
            pil.save(output_dir / f"{start + offset:06d}.png")


def _count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.iterdir() if path.suffix.lower() == ".png")


def _compute_fid(
    real_dir: Path,
    fake_dir: Path,
    *,
    fid_batch: int,
    fid_num_workers: int,
    device: torch.device,
) -> float:
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError as exc:
        raise ImportError("pytorch-fid is required for FFHQ FID evaluation. Install the `eval` extra.") from exc
    return float(
        calculate_fid_given_paths(
            [str(real_dir), str(fake_dir)],
            batch_size=fid_batch,
            device=device,
            dims=2048,
            num_workers=fid_num_workers,
        )
    )


def evaluate_ffhq_checkpoint(
    checkpoint_path: str | Path,
    *,
    real_npz: str | Path | None = None,
    real_split: str = _DEFAULT_REAL_SPLIT,
    device: str = "cuda",
    n_per_class: int = 128,
    solver: str = "emd",
    metric: str = "l2_sq",
    sinkhorn_reg: float = 0.05,
    ot_iters: int = 200000,
    alae_ckpt: str = _DEFAULT_ALAE_CKPT,
    decode_batch: int = 8,
    gen_batch: int = 256,
    fid_batch: int = 64,
    fid_num_workers: int = 0,
    save_size: int = 1024,
    reuse_images: bool = False,
    fid_decode_noise: bool = True,
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
    experiment.evaluation.evaluator = "ffhq"
    experiment.evaluation.metrics = list(metrics or _DEFAULT_FFHQ_EVAL_METRICS)
    experiment.evaluation.params = {
        "real_split": _normalize_real_split(real_split),
        "device": device,
        "n_per_class": n_per_class,
        "solver": solver,
        "metric": metric,
        "sinkhorn_reg": sinkhorn_reg,
        "ot_iters": ot_iters,
        "decode_batch": decode_batch,
        "gen_batch": gen_batch,
        "fid_batch": fid_batch,
        "fid_num_workers": fid_num_workers,
        "save_size": save_size,
        "reuse_images": reuse_images,
        "fid_decode_noise": fid_decode_noise,
    }
    if real_npz not in (None, ""):
        experiment.evaluation.params["real_npz"] = str(
            normalize_path(real_npz, runtime_roots=runtime_roots, resolve_latest=True)
        )
    if seed is not None:
        experiment.evaluation.params["seed"] = int(seed)
    if alae_ckpt != "":
        experiment.evaluation.params["alae_ckpt"] = str(
            normalize_path(
                _DEFAULT_ALAE_CKPT if alae_ckpt == _DEFAULT_ALAE_CKPT else alae_ckpt,
                runtime_roots=runtime_roots,
                resolve_latest=True,
            )
        )
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
    output_path = evaluate_ffhq_checkpoint(
        args.checkpoint,
        real_npz=args.real_npz,
        real_split=args.real_split,
        device=args.device,
        n_per_class=args.n_per_class,
        solver=args.solver,
        metric=args.metric,
        sinkhorn_reg=args.sinkhorn_reg,
        ot_iters=args.ot_iters,
        alae_ckpt=args.alae_ckpt,
        decode_batch=args.decode_batch,
        gen_batch=args.gen_batch,
        fid_batch=args.fid_batch,
        fid_num_workers=args.fid_num_workers,
        save_size=args.save_size,
        reuse_images=args.reuse_images,
        fid_decode_noise=args.fid_decode_noise,
        seed=args.seed,
        deterministic=not args.no_deterministic,
        num_sampling_steps=args.num_sampling_steps,
        metrics=args.metrics,
    )
    print(output_path)
