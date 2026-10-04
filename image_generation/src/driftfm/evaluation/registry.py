from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from ..config import (
    ExperimentConfig,
    RuntimeConfig,
    coerce_evaluation_config,
    coerce_experiment_config,
    load_evaluation_config,
    load_experiment_config,
)
from ..training.checkpoints import load_checkpoint
from ..utils import normalize_path, resolve_latest_alias, resolve_runtime_config
from .base import CheckpointEvaluator, EvaluationResult
from .ffhq import FFHQEvaluator
from .mnist import MnistEvaluator

DATASET_EVALUATORS: dict[str, str] = {
    "mnist_latent": "mnist",
    "ffhq_latent": "ffhq",
}

EVALUATORS: dict[str, CheckpointEvaluator] = {
    "mnist": MnistEvaluator(),
    "ffhq": FFHQEvaluator(),
}


def resolve_evaluator_name(experiment: ExperimentConfig) -> str:
    if experiment.evaluation.evaluator:
        return experiment.evaluation.evaluator
    try:
        return DATASET_EVALUATORS[experiment.dataset.name]
    except KeyError as exc:
        raise KeyError(
            f"Unable to infer evaluator for dataset={experiment.dataset.name}. "
            "Set evaluation.evaluator explicitly."
        ) from exc


def get_evaluator(name: str) -> CheckpointEvaluator:
    try:
        return EVALUATORS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown evaluator: {name}") from exc


def _merge_evaluation_override(
    experiment: ExperimentConfig,
    config_path: str | Path | None,
) -> ExperimentConfig:
    if config_path is None:
        return experiment
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if {"dataset", "architecture", "method"}.issubset(raw):
        return load_experiment_config(config_path)
    runtime_payload = asdict(experiment.runtime)
    authored_runtime = dict(raw.get("runtime", {}))
    runtime_payload.update(authored_runtime)
    if "data_root" in authored_runtime and "mnist_root" not in authored_runtime:
        runtime_payload["mnist_root"] = None
    experiment.runtime = RuntimeConfig(**resolve_runtime_config(runtime_payload, source_path=path))
    experiment.evaluation = load_evaluation_config(config_path, runtime=experiment.runtime)
    return experiment


def _step_dir_name(num_sampling_steps: int) -> str:
    return f"steps_{num_sampling_steps}"


def _flatten_step_summary(num_sampling_steps: int, summary: dict[str, float]) -> dict[str, float]:
    return {f"steps_{num_sampling_steps}.{key}": value for key, value in summary.items()}


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    experiment: ExperimentConfig | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> EvaluationResult:
    resolved_checkpoint = resolve_latest_alias(normalize_path(checkpoint_path, resolve_latest=False))
    checkpoint = load_checkpoint(resolved_checkpoint)
    active_experiment = experiment if experiment is not None else coerce_experiment_config(checkpoint["experiment"])
    active_experiment = _merge_evaluation_override(active_experiment, config_path)

    evaluator = get_evaluator(resolve_evaluator_name(active_experiment))
    active_experiment.evaluation.metrics = evaluator.validate_metrics(
        active_experiment.evaluation.metrics or list(evaluator.supported_metrics)
    )
    output_root = Path(output_dir) if output_dir is not None else resolved_checkpoint.resolve().parent
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = output_root.resolve()

    step_entries: list[dict[str, object]] = []
    summary: dict[str, float] = {}
    base_eval_params = dict(active_experiment.evaluation.params)
    for num_sampling_steps in active_experiment.evaluation.num_sampling_steps:
        step_output_dir = output_root / _step_dir_name(num_sampling_steps)
        eval_params = dict(base_eval_params)
        eval_params["_evaluation_root_dir"] = str(output_root)
        eval_params["_allow_real_cache_reuse"] = bool(base_eval_params.get("reuse_images", False)) or bool(
            step_entries
        )
        eval_experiment = replace(
            active_experiment,
            evaluation=replace(
                active_experiment.evaluation,
                params=eval_params,
            ),
        )
        result = evaluator.evaluate(
            resolved_checkpoint,
            experiment=eval_experiment,
            checkpoint=checkpoint,
            output_dir=step_output_dir,
            num_sampling_steps=num_sampling_steps,
        )
        step_entry: dict[str, object] = {
            "num_sampling_steps": num_sampling_steps,
            "output_dir": str(step_output_dir.resolve()),
            "output_path": str(result.output_path.resolve()),
            "summary": dict(result.summary),
        }
        with result.output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "resolved_seed" in payload:
            step_entry["resolved_seed"] = payload["resolved_seed"]
        if "determinism" in payload:
            step_entry["determinism"] = payload["determinism"]
        if evaluator.name == "ffhq":
            if "resolved_real_split" in payload:
                step_entry["resolved_real_split"] = payload["resolved_real_split"]
            if "resolved_real_npz" in payload:
                step_entry["resolved_real_npz"] = payload["resolved_real_npz"]
            if "fid_artifacts_path" in payload:
                step_entry["fid_artifacts_path"] = payload["fid_artifacts_path"]
            if "fid_decode_noise" in payload:
                step_entry["fid_decode_noise"] = payload["fid_decode_noise"]
        step_entries.append(step_entry)
        summary.update(_flatten_step_summary(num_sampling_steps, result.summary))

    index_path = output_root / "evaluation_index.json"
    payload = {
        "checkpoint": str(resolved_checkpoint),
        "evaluator": evaluator.name,
        "metrics": list(active_experiment.evaluation.metrics),
        "num_sampling_steps": list(active_experiment.evaluation.num_sampling_steps),
        "steps": step_entries,
    }
    if step_entries and "resolved_seed" in step_entries[0]:
        payload["resolved_seed"] = step_entries[0]["resolved_seed"]
    if step_entries and "determinism" in step_entries[0]:
        payload["determinism"] = step_entries[0]["determinism"]
    if step_entries and "resolved_real_split" in step_entries[0]:
        payload["resolved_real_split"] = step_entries[0]["resolved_real_split"]
    if step_entries and "resolved_real_npz" in step_entries[0]:
        payload["resolved_real_npz"] = step_entries[0]["resolved_real_npz"]
    if step_entries and "fid_decode_noise" in step_entries[0]:
        payload["fid_decode_noise"] = step_entries[0]["fid_decode_noise"]
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return EvaluationResult(output_path=index_path, summary=summary)
