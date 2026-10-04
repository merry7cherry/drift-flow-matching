from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..configs import (
    DatasetConfig, DriftFlowMatchingConfig, IntegratorConfig, MeanFlowConfig,
    TrainingConfig, VisualizationEvalConfig,
)
from ..pipelines.checkpoints import load_checkpoint, save_checkpoint
from ..pipelines.inference import (
    compute_drift_flow_matching_trajectories, compute_mean_flow_trajectories,
    compute_model_trajectories,
)
from ..pipelines.training import (
    train_drift_flow_matching, train_flow_matching, train_mean_flow_matching,
)
from ..seed import seed_all
from ..visualization.plotting import create_2d_trajectory_figure, save_figure

ALL_METHODS = ("drift_flow_matching", "flow_matching", "mean_flow")
METHOD_LABELS = {
    "drift_flow_matching": "Drift Flow Matching",
    "flow_matching": "Flow Matching",
    "mean_flow": "MeanFlow",
}


@dataclass(frozen=True)
class VisualizationRunConfig:
    output_dir: Path
    seed: int
    training_config: TrainingConfig
    integrator_config: IntegratorConfig
    drift_flow_config: DriftFlowMatchingConfig
    mean_flow_config: MeanFlowConfig
    eval_config: VisualizationEvalConfig
    methods: tuple[str, ...] = ("drift_flow_matching",)
    load_dir: Path | None = None


def run_visualization_experiments(
    dataset_configs: Sequence[DatasetConfig], run_config: VisualizationRunConfig,
) -> None:
    """Train or load each method, then export matched-source trajectories and figures.

    Every method and NFE uses the same evaluation source points. The target samples
    show the target distribution; they are not pathwise ground truth.
    """
    device = torch.device(run_config.training_config.device)
    output_dir = Path(run_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    eval_config = run_config.eval_config
    for dataset_config in dataset_configs:
        dataset = dataset_config.create_dataset(run_config.seed)
        key = dataset_config.name
        dataset.reset_rng(run_config.seed)
        eval_batch = dataset.sample_pairs(eval_config.eval_samples, device)
        for method in run_config.methods:
            if method not in ALL_METHODS:
                raise ValueError(f"Unknown method: {method}")
            filename = f"{key}_{method}"
            load_dir = run_config.load_dir
            if load_dir:
                checkpoint = Path(load_dir) / f"{filename}.pt"
                model, metadata = load_checkpoint(
                    checkpoint, device, expected_method=method, expected_dataset=key,
                )
                print(f"Loaded {checkpoint}", flush=True)
            else:
                seed_all(run_config.seed)
                dataset.reset_rng(run_config.seed)
                print(f"Training {METHOD_LABELS[method]} on {dataset_config.label}...", flush=True)
                if method == "drift_flow_matching":
                    config = run_config.drift_flow_config
                    artifacts = train_drift_flow_matching(dataset, run_config.training_config, config)
                    method_config = asdict(config)
                    hidden_sizes = config.velocity_hidden_sizes
                elif method == "mean_flow":
                    config = run_config.mean_flow_config
                    artifacts = train_mean_flow_matching(dataset, run_config.training_config, config)
                    method_config = asdict(config)
                    hidden_sizes = config.velocity_hidden_sizes
                else:
                    artifacts = train_flow_matching(dataset, run_config.training_config)
                    method_config = {}
                    hidden_sizes = (128, 128, 128)
                model = artifacts.model
                checkpoint = checkpoint_dir / f"{filename}.pt"
                save_checkpoint(
                    checkpoint, model, method=method, dataset_key=key,
                    hidden_sizes=hidden_sizes, training_config=run_config.training_config,
                    method_config=method_config, seed=run_config.seed, losses=artifacts.history.losses,
                )
                _, metadata = load_checkpoint(
                    checkpoint, device, expected_method=method, expected_dataset=key,
                )
            nfe_values = (run_config.integrator_config.num_steps,) if method == "flow_matching" else eval_config.nfe
            trajectories = []
            for nfe in nfe_values:
                if method == "drift_flow_matching":
                    states, times = compute_drift_flow_matching_trajectories(model, eval_batch.x0, device, steps=nfe)
                elif method == "mean_flow":
                    states, times = compute_mean_flow_trajectories(model, eval_batch.x0, device, steps=nfe)
                else:
                    states, times = compute_model_trajectories(model, eval_batch.x0, device, IntegratorConfig(nfe))
                if not torch.isfinite(states).all():
                    raise RuntimeError(f"Non-finite trajectory: {filename}, NFE={nfe}")
                stem = f"{filename}_nfe_{nfe}"
                np.savez_compressed(
                    output_dir / f"{stem}.npz",
                    times=times.detach().cpu().numpy(),
                    states=states.detach().cpu().numpy(),
                    target_samples=eval_batch.x1.detach().cpu().numpy(),
                )
                trajectories.append((stem, states, times))
            # Fix the axes across NFE to make visual comparisons meaningful.
            points = torch.cat([eval_batch.x1, *[states.flatten(0, 1) for _, states, _ in trajectories]])
            lower = points.amin(0).detach().cpu().numpy()
            upper = points.amax(0).detach().cpu().numpy()
            padding = np.maximum(upper - lower, 1.0) * 0.05
            bounds = (lower - padding, upper + padding)
            for stem, states, times in trajectories:
                fig = create_2d_trajectory_figure(
                    states, f"{METHOD_LABELS[method]} · NFE {len(times) - 1}",
                    target_samples=eval_batch.x1, max_display=eval_config.max_display, bounds=bounds,
                )
                save_figure(fig, output_dir / f"{stem}.png")
            manifest = {
                "schema_version": 1, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "checkpoint_name": checkpoint.name, "checkpoint_metadata": metadata,
                "evaluation": {"seed": run_config.seed, "samples": eval_config.eval_samples, "nfe": list(nfe_values), "device": str(device)},
                "dataset": {"name": key, "kwargs": dict(dataset_config.kwargs)},
                "traces": [f"{stem}.npz" for stem, _, _ in trajectories],
                "trace_semantics": "states[k] is the learned sampler state at times[k]; target_samples are unpaired reference samples, not pathwise ground truth",
                "provenance": "New run from the public implementation; not a recovered original paper checkpoint.",
            }
            (output_dir / f"{filename}.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Saved checkpoint metadata, NPZ traces, and PNG figures to {output_dir}", flush=True)
