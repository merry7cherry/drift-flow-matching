from __future__ import annotations

import json
import random
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..architectures import create_architecture
from ..config import (
    ArchitectureSection,
    DatasetSection,
    ExperimentConfig,
    coerce_experiment_config,
    dump_experiment_config,
    load_experiment_config,
)
from ..data import Conditioning, TransportDataset, create_dataset
from ..evaluation.registry import evaluate_checkpoint
from ..methods import create_method
from ..methods.drift_flow_matching import conditioning_mode_for_drift_form
from ..utils import (
    build_seed_hash_run_name,
    collect_determinism_metadata,
    configure_determinism,
    make_torch_generator,
    normalize_path,
    write_json_file,
    write_latest_run_marker,
)
from .checkpoints import find_latest_checkpoint, load_checkpoint, save_checkpoint
from .ema import EMA

_RUN_METADATA_FILENAME = "run_metadata.json"


def _seed_all(seed: int, *, deterministic: bool, device: str | torch.device) -> None:
    configure_determinism(seed, deterministic=deterministic, device=device)


def _dataset_section_with_default_seed(config: ExperimentConfig) -> DatasetSection:
    params = dict(config.dataset.params)
    params.setdefault("seed", int(config.trainer.seed))
    config.dataset.params = params
    return DatasetSection(name=config.dataset.name, params=params)


def _deterministic_run_name(config: ExperimentConfig) -> str:
    payload = config.to_dict()
    payload["output"] = {**dict(payload.get("output", {})), "run_name": None}
    return build_seed_hash_run_name(seed=config.trainer.seed, payload=payload)


def _assert_existing_run_matches_config(run_dir: Path, config: ExperimentConfig) -> None:
    resolved_config_path = run_dir / "resolved_config.yaml"
    if not resolved_config_path.exists():
        return
    with resolved_config_path.open("r", encoding="utf-8") as handle:
        existing_payload = yaml.safe_load(handle) or {}
    if existing_payload != config.to_dict():
        raise ValueError(
            f"Deterministic run directory {run_dir} already exists with a different resolved config."
        )


def _build_run_dir(config: ExperimentConfig) -> Path:
    run_name = config.output.run_name
    auto_named = run_name is None
    if run_name is None and config.trainer.deterministic:
        run_name = _deterministic_run_name(config)
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output.run_name = run_name
    run_dir = Path(config.output.root_dir) / run_name
    if config.trainer.deterministic and auto_named and run_dir.exists():
        _assert_existing_run_matches_config(run_dir, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _materialize_architecture_section(
    config: ExperimentConfig,
    dataset: TransportDataset,
) -> ArchitectureSection:
    params = dict(config.architecture.params)
    params.setdefault("data_dim", dataset.data_dim)
    if dataset.num_classes is not None:
        params.setdefault("num_classes", dataset.num_classes)
    params.setdefault(
        "conditioning_mode",
        conditioning_mode_for_drift_form(str(config.method.params.get("drift_form", "split_v0"))),
    )
    return ArchitectureSection(name=config.architecture.name, params=params)


def _build_sampling_conditioning(
    dataset: TransportDataset,
    samples_per_class: int,
    device: torch.device,
) -> Conditioning | None:
    return dataset.sample_generation_conditioning(samples_per_class, device)


def _save_samples(
    output_path: Path,
    samples: torch.Tensor,
    conditioning: Conditioning | None,
) -> None:
    payload: dict[str, Any] = {"samples": samples.detach().cpu().numpy()}
    if conditioning is not None and conditioning.class_labels is not None:
        payload["class_labels"] = conditioning.class_labels.detach().cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _run_metadata_payload(
    *,
    seed: int,
    dataset_seed: int,
    deterministic: bool,
    device: str | torch.device,
) -> dict[str, Any]:
    return {
        "resolved_seed": int(seed),
        "dataset_seed": int(dataset_seed),
        "determinism": collect_determinism_metadata(
            seed,
            deterministic=deterministic,
            device=device,
        ),
    }


def _save_sampling_previews(
    run_dir: Path,
    *,
    epoch: int,
    method: Any,
    model: torch.nn.Module,
    dataset: TransportDataset,
    conditioning: Conditioning | None,
    sample_count: int,
    device: torch.device,
    num_sampling_steps: list[int],
    seed: int,
) -> Path:
    preview_root = run_dir / "samples" / f"epoch{epoch:03d}"
    preview_generator = make_torch_generator(device, seed=seed + 10_000 + epoch)
    noise = dataset.make_noise(sample_count, device, generator=preview_generator)
    step_entries: list[dict[str, object]] = []

    for steps in num_sampling_steps:
        step_dir = preview_root / f"steps_{steps}"
        samples = method.sample(model, noise.clone(), conditioning, num_steps=steps)
        sample_path = step_dir / "samples.npz"
        _save_samples(sample_path, samples, conditioning)
        step_entries.append({
            "num_sampling_steps": steps,
            "output_dir": str(step_dir.resolve()), "output_path": str(sample_path.resolve()),
        })

    index_path = preview_root / "samples_index.json"
    _write_json(
        index_path,
        {
            "epoch": epoch,
            "num_sampling_steps": list(num_sampling_steps),
            "steps": step_entries,
        },
    )
    return index_path


def train_from_config_path(
    config_path: str | Path,
    *,
    resume_from: str | Path | None = None,
) -> Path:
    return train_from_config(
        load_experiment_config(config_path),
        config_path=config_path,
        resume_from=resume_from,
    )


def _normalize_resume_path(path: str | Path) -> Path:
    return normalize_path(path, resolve_latest=True)


def _coerce_rng_state_tensor(value: Any) -> Any:
    tensor = value.detach() if hasattr(value, "detach") else torch.as_tensor(value)
    if getattr(getattr(tensor, "device", None), "type", "cpu") != "cpu" and hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if getattr(tensor, "dtype", None) != torch.uint8 and hasattr(tensor, "to"):
        tensor = tensor.to(dtype=torch.uint8)
    return tensor


def _restore_rng_state(
    checkpoint: dict[str, Any],
    *,
    dataset: TransportDataset,
) -> None:
    python_rng_state = checkpoint.get("python_rng_state")
    if python_rng_state is not None:
        random.setstate(python_rng_state)
    numpy_rng_state = checkpoint.get("numpy_rng_state")
    if numpy_rng_state is not None:
        np.random.set_state(numpy_rng_state)
    torch_rng_state = checkpoint.get("torch_rng_state")
    if torch_rng_state is not None:
        torch.set_rng_state(_coerce_rng_state_tensor(torch_rng_state))
    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_coerce_rng_state_tensor(state) for state in cuda_rng_state_all])
    dataset_generator_state = checkpoint.get("dataset_generator_state")
    if dataset_generator_state is not None and hasattr(dataset, "_generator"):
        dataset._generator.set_state(_coerce_rng_state_tensor(dataset_generator_state))


def _snapshot_rng_state(dataset: TransportDataset) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "dataset_generator_state": dataset._generator.get_state() if hasattr(dataset, "_generator") else None,
    }
    payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return payload


@contextmanager
def _preserve_training_rng_state(dataset: TransportDataset):
    rng_state = _snapshot_rng_state(dataset)
    try:
        yield
    finally:
        _restore_rng_state(rng_state, dataset=dataset)


def _inline_evaluation_output_dir(
    config: ExperimentConfig,
    *,
    run_dir: Path,
    epoch: int,
) -> Path | None:
    metrics_scheduled = config.evaluation.metrics_every > 0 and epoch % config.evaluation.metrics_every == 0
    final_metrics = epoch == config.trainer.epochs and config.evaluation.run_final
    if not config.evaluation.metrics or not (metrics_scheduled or final_metrics):
        return None
    return run_dir / "evals" / ("final" if final_metrics else f"epoch{epoch:03d}")


def _append_eval_log_if_missing(eval_log_path: Path, payload: dict[str, Any]) -> None:
    identity = (
        int(payload["epoch"]),
        str(payload["checkpoint"]),
        str(payload["output_path"]),
    )
    if eval_log_path.exists():
        for line in eval_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                existing = json.loads(line)
            except json.JSONDecodeError:
                continue
            existing_identity = (
                int(existing.get("epoch", -1)),
                str(existing.get("checkpoint", "")),
                str(existing.get("output_path", "")),
            )
            if existing_identity == identity:
                return
    with eval_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _build_checkpoint_payload(
    *,
    epoch: int,
    step: int,
    model: torch.nn.Module,
    ema: EMA | None,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    resolved_architecture: dict[str, Any],
    dataset: TransportDataset,
    last_metrics: dict[str, float],
    pending_inline_evaluation_dir: Path | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "step": step,
        "model_state": model.state_dict(),
        "ema_state": None if ema is None else {name: value.clone() for name, value in ema.shadow.items()},
        "optimizer_state": optimizer.state_dict(),
        "experiment": config.to_dict(),
        "resolved_architecture": resolved_architecture,
        "dataset_info": {
            "data_shape": dataset.data_shape,
            "data_dim": dataset.data_dim,
            "num_classes": dataset.num_classes,
            "class_names": dataset.class_names,
        },
        "last_metrics": last_metrics,
        "pending_inline_evaluation": pending_inline_evaluation_dir is not None,
        "pending_inline_evaluation_epoch": epoch if pending_inline_evaluation_dir is not None else None,
        "pending_inline_evaluation_output_dir": None if pending_inline_evaluation_dir is None else str(pending_inline_evaluation_dir),
    }
    payload.update(_snapshot_rng_state(dataset))
    return payload


def _run_inline_evaluation(
    checkpoint_path: Path,
    *,
    dataset: TransportDataset,
    experiment: ExperimentConfig,
    epoch: int,
    step: int,
    output_dir: Path,
    eval_log_path: Path,
) -> None:
    with _preserve_training_rng_state(dataset):
        result = evaluate_checkpoint(checkpoint_path, experiment=experiment, output_dir=output_dir)
    _append_eval_log_if_missing(
        eval_log_path,
        {
            "epoch": epoch,
            "step": step,
            "checkpoint": str(checkpoint_path),
            "output_path": str(result.output_path),
            **result.summary,
        },
    )


def train_from_config(
    config: ExperimentConfig,
    *,
    config_path: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> Path:
    if config_path is not None:
        config = coerce_experiment_config(config.to_dict(), source_path=config_path)
    dataset_section = _dataset_section_with_default_seed(config)
    _seed_all(config.trainer.seed, deterministic=config.trainer.deterministic, device=config.trainer.device)
    device = torch.device(config.trainer.device)
    dataset_seed = int(dataset_section.params["seed"])

    run_dir = _build_run_dir(config)
    if resume_from is None and any(run_dir.glob("checkpoint*.pt")):
        raise FileExistsError(
            f"Run {run_dir} already contains checkpoints. Use --resume-from auto or choose a new output.run_name."
        )
    write_json_file(
        run_dir / _RUN_METADATA_FILENAME,
        _run_metadata_payload(
            seed=config.trainer.seed,
            dataset_seed=dataset_seed,
            deterministic=config.trainer.deterministic,
            device=config.trainer.device,
        ),
    )
    resolved_resume = None if resume_from is None else (
        find_latest_checkpoint(run_dir) if str(resume_from).strip().lower() == "auto" else _normalize_resume_path(resume_from)
    )

    dataset = create_dataset(dataset_section)
    method = create_method(config.method)
    architecture_section = _materialize_architecture_section(config, dataset)
    model = create_architecture(architecture_section, dataset).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.trainer.learning_rate,
        weight_decay=config.trainer.weight_decay,
    )
    ema = EMA(model, decay=method.ema_decay) if method.uses_ema else None

    dump_experiment_config(config, run_dir / "resolved_config.yaml")

    log_path = run_dir / "logs.jsonl"
    eval_log_path = run_dir / "evals.jsonl"
    resolved_architecture = {"name": architecture_section.name, "params": dict(architecture_section.params)}

    start_epoch = 1
    global_step = 0
    if resolved_resume is not None:
        checkpoint = load_checkpoint(resolved_resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if ema is not None and checkpoint.get("ema_state") is not None:
            ema.shadow = {
                name: value.detach().clone().to(device)
                for name, value in checkpoint["ema_state"].items()
            }
        global_step = int(checkpoint.get("step", 0))
        _restore_rng_state(checkpoint, dataset=dataset)
        if bool(checkpoint.get("pending_inline_evaluation", False)):
            pending_epoch = int(checkpoint.get("pending_inline_evaluation_epoch") or checkpoint.get("epoch", 0))
            pending_output_dir_value = checkpoint.get("pending_inline_evaluation_output_dir")
            pending_output_dir = (
                Path(pending_output_dir_value)
                if pending_output_dir_value is not None
                else _inline_evaluation_output_dir(config, run_dir=run_dir, epoch=pending_epoch)
            )
            if pending_output_dir is not None:
                _run_inline_evaluation(
                    resolved_resume,
                    dataset=dataset,
                    experiment=config,
                    epoch=pending_epoch,
                    step=global_step,
                    output_dir=pending_output_dir,
                    eval_log_path=eval_log_path,
                )
            checkpoint["pending_inline_evaluation"] = False
            checkpoint["pending_inline_evaluation_epoch"] = None
            checkpoint["pending_inline_evaluation_output_dir"] = None
            checkpoint.update(_snapshot_rng_state(dataset))
            save_checkpoint(resolved_resume, checkpoint)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    if start_epoch > config.trainer.epochs:
        write_latest_run_marker(config.output.root_dir, run_dir)
        return run_dir

    for epoch in range(start_epoch, config.trainer.epochs + 1):
        model.train()
        epoch_loss = 0.0
        last_metrics: dict[str, float] = {}
        for _ in range(config.trainer.steps_per_epoch):
            batch = dataset.sample_batch(batch_size=config.trainer.batch_size, device=device)
            output = method.compute_loss(model, batch)
            optimizer.zero_grad()
            output.loss.backward()
            if config.trainer.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            global_step += 1
            epoch_loss += float(output.loss.detach().item())
            last_metrics = dict(output.metrics)

        average_loss = epoch_loss / max(1, config.trainer.steps_per_epoch)
        record = {"epoch": epoch, "step": global_step, "avg_loss": average_loss, **last_metrics}
        if epoch % config.trainer.log_every == 0 or epoch == 1 or epoch == config.trainer.epochs:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

        active_model = create_architecture(architecture_section, dataset).to(device)
        active_model.load_state_dict(model.state_dict())
        if ema is not None:
            ema.copy_to(active_model)

        if config.evaluation.sample_every > 0 and (
            epoch % config.evaluation.sample_every == 0 or epoch == config.trainer.epochs
        ):
            conditioning = _build_sampling_conditioning(
                dataset,
                config.evaluation.sample_count_per_class,
                device,
            )
            sample_count = (
                config.evaluation.sample_count_per_class * dataset.num_classes
                if dataset.num_classes is not None
                else config.evaluation.sample_count_per_class
            )
            _save_sampling_previews(
                run_dir,
                epoch=epoch,
                method=method,
                model=active_model,
                dataset=dataset,
                conditioning=conditioning,
                sample_count=sample_count,
                device=device,
                num_sampling_steps=config.evaluation.num_sampling_steps,
                seed=config.trainer.seed,
            )

        if epoch % config.trainer.checkpoint_every == 0 or epoch == config.trainer.epochs:
            filename = "checkpoint_final.pt" if epoch == config.trainer.epochs else f"checkpoint_epoch{epoch}.pt"
            checkpoint_path = run_dir / filename
            pending_eval_dir = _inline_evaluation_output_dir(config, run_dir=run_dir, epoch=epoch)
            save_checkpoint(
                checkpoint_path,
                _build_checkpoint_payload(
                    epoch=epoch,
                    step=global_step,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    config=config,
                    resolved_architecture=resolved_architecture,
                    dataset=dataset,
                    last_metrics=record,
                    pending_inline_evaluation_dir=pending_eval_dir,
                ),
            )

            if pending_eval_dir is not None:
                _run_inline_evaluation(
                    checkpoint_path,
                    dataset=dataset,
                    experiment=config,
                    epoch=epoch,
                    step=global_step,
                    output_dir=pending_eval_dir,
                    eval_log_path=eval_log_path,
                )
                save_checkpoint(
                    checkpoint_path,
                    _build_checkpoint_payload(
                        epoch=epoch,
                        step=global_step,
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        config=config,
                        resolved_architecture=resolved_architecture,
                        dataset=dataset,
                        last_metrics=record,
                        pending_inline_evaluation_dir=None,
                    ),
                )

    write_latest_run_marker(config.output.root_dir, run_dir)
    return run_dir
