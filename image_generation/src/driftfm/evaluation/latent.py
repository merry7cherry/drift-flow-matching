from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..architectures import create_architecture
from ..config import ArchitectureSection, ExperimentConfig, MethodSection
from ..data import Conditioning
from ..methods import create_method
from ..methods.drift_flow_matching import conditioning_mode_for_drift_form
from ..utils import make_torch_generator
from .projection import PROJECTION_METHODS, ProjectionArtifacts, write_projection_artifacts


@dataclass(slots=True)
class LatentSampleSet:
    fake_by_class: dict[str, np.ndarray]


def resolve_evaluation_seed(experiment: ExperimentConfig) -> int:
    raw_seed = experiment.evaluation.params.get("seed", experiment.trainer.seed)
    if raw_seed in (None, ""):
        raw_seed = experiment.trainer.seed
    if isinstance(raw_seed, bool):
        raise ValueError("evaluation.params.seed must be an integer >= 0")
    if isinstance(raw_seed, float) and not raw_seed.is_integer():
        raise ValueError("evaluation.params.seed must be an integer >= 0")
    try:
        seed = int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation.params.seed must be an integer >= 0") from exc
    if seed < 0:
        raise ValueError("evaluation.params.seed must be an integer >= 0")
    return seed

def resolve_evaluation_conditioning_mode(
    experiment: ExperimentConfig,
    checkpoint: dict[str, Any],
) -> str:
    resolved = checkpoint.get("resolved_architecture", {})
    resolved_params = dict(resolved.get("params", {}))
    if resolved_params.get("conditioning_mode") is not None:
        return str(resolved_params["conditioning_mode"])

    if experiment.architecture.params.get("conditioning_mode") is not None:
        return str(experiment.architecture.params["conditioning_mode"])

    method_payload = checkpoint.get("experiment", {}).get("method", {})
    drift_form = str(
        method_payload.get("params", {}).get(
            "drift_form",
            experiment.method.params.get("drift_form", "split_v0"),
        )
    )
    return conditioning_mode_for_drift_form(drift_form)


def load_generation_stack(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, Any]:
    architecture_section = checkpoint["resolved_architecture"]
    model = create_architecture(
        ArchitectureSection(
            name=architecture_section["name"],
            params=dict(architecture_section["params"]),
        )
    ).to(device)
    state = checkpoint["ema_state"] if checkpoint.get("ema_state") is not None else checkpoint["model_state"]
    model.load_state_dict(state)
    model.eval()

    method_section = checkpoint["experiment"]["method"]
    method = create_method(MethodSection(name=method_section["name"], params=dict(method_section.get("params", {}))))
    return model, method


def sample_latents_by_class(
    model: torch.nn.Module,
    method: Any,
    *,
    class_names: list[str],
    n_per_class: int,
    latent_dim: int,
    device: torch.device,
    gen_batch: int,
    num_sampling_steps: int,
    generator: torch.Generator | None = None,
) -> LatentSampleSet:
    fake_by_class: dict[str, np.ndarray] = {}
    for class_index, class_name in enumerate(class_names):
        fake_pieces: list[np.ndarray] = []
        remaining = n_per_class
        while remaining > 0:
            current = min(gen_batch, remaining)
            labels = torch.full((current,), class_index, dtype=torch.long, device=device)
            conditioning = Conditioning(class_labels=labels)
            noise = torch.randn((current, latent_dim), device=device, generator=generator)
            with torch.no_grad():
                samples = method.sample(
                    model,
                    noise,
                    conditioning,
                    num_steps=num_sampling_steps,
                ).cpu().numpy()
            fake_pieces.append(samples.astype(np.float32, copy=False))
            remaining -= current
        fake_by_class[class_name] = np.concatenate(fake_pieces, axis=0).astype(np.float32, copy=False)
    return LatentSampleSet(fake_by_class=fake_by_class)


def save_sampled_latents_by_class(
    output_dir: str | Path,
    *,
    class_names: list[str],
    real_by_class: dict[str, np.ndarray],
    fake_by_class: dict[str, np.ndarray],
) -> Path:
    output_path = Path(output_dir) / "sampled_latents_by_class.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **{f"real_{class_name}": np.asarray(real_by_class[class_name], dtype=np.float32) for class_name in class_names},
        **{f"fake_{class_name}": np.asarray(fake_by_class[class_name], dtype=np.float32) for class_name in class_names},
    )
    return output_path.resolve()


def maybe_write_projection_artifacts(
    *,
    metrics: list[str],
    output_dir: str | Path,
    class_names: list[str],
    fake_by_class: dict[str, np.ndarray],
    real_by_class: dict[str, np.ndarray],
    random_state: int,
) -> dict[str, ProjectionArtifacts]:
    methods = [metric for metric in metrics if metric in PROJECTION_METHODS]
    if not methods:
        return {}
    return write_projection_artifacts(
        methods=methods,
        fake_by_class=fake_by_class,
        real_by_class=real_by_class,
        class_names=class_names,
        output_dir=output_dir,
        random_state=random_state,
    )


def add_projection_artifacts_to_payload(
    payload: dict[str, Any],
    artifacts_by_method: dict[str, ProjectionArtifacts],
) -> None:
    for method_name, artifacts in artifacts_by_method.items():
        payload[f"{method_name}_plot_path"] = str(artifacts.plot_path)
        payload[f"{method_name}_real_only_plot_path"] = str(artifacts.real_only_plot_path)
        payload[f"{method_name}_generated_only_plot_path"] = str(artifacts.generated_only_plot_path)
        payload[f"{method_name}_projection_path"] = str(artifacts.projection_path)
        if artifacts.real_fit_plot_path is None:
            continue
        assert artifacts.real_fit_real_only_plot_path is not None
        assert artifacts.real_fit_generated_only_plot_path is not None
        assert artifacts.real_fit_projection_path is not None
        payload[f"{method_name}_real_fit_plot_path"] = str(artifacts.real_fit_plot_path)
        payload[f"{method_name}_real_fit_real_only_plot_path"] = str(artifacts.real_fit_real_only_plot_path)
        payload[f"{method_name}_real_fit_generated_only_plot_path"] = str(artifacts.real_fit_generated_only_plot_path)
        payload[f"{method_name}_real_fit_projection_path"] = str(artifacts.real_fit_projection_path)
