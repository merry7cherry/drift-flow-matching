from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from ..utils import normalize_path, normalize_path_fields, resolve_runtime_config
from .schemas import (
    ArchitectureSection,
    DatasetSection,
    EvaluationConfig,
    ExperimentConfig,
    MethodSection,
    OutputConfig,
    RuntimeConfig,
    TrainerConfig,
)

def _coerce_named_section(section_type: type, payload: dict[str, Any] | None) -> Any:
    payload = payload or {}
    return section_type(
        name=str(payload["name"]),
        params=dict(payload.get("params", {})),
    )


def coerce_evaluation_config(
    raw: dict[str, Any],
    *,
    runtime: RuntimeConfig | dict[str, Any] | None = None,
    source_path: str | Path | None = None,
) -> EvaluationConfig:
    runtime_payload = resolve_runtime_config(runtime, source_path=source_path)
    normalized = normalize_path_fields(
        dict(raw),
        runtime_roots=runtime_payload,
        base_dir=source_path,
    )
    return EvaluationConfig(**normalized)


def coerce_experiment_config(
    raw: dict[str, Any],
    *,
    source_path: str | Path | None = None,
) -> ExperimentConfig:
    payload = dict(raw)
    runtime_payload = resolve_runtime_config(payload.get("runtime"), source_path=source_path)
    payload["runtime"] = runtime_payload
    output_payload = dict(payload.get("output", {}))
    output_payload.setdefault("root_dir", OutputConfig().root_dir)
    payload["output"] = output_payload
    payload = normalize_path_fields(payload, runtime_roots=runtime_payload, base_dir=source_path)
    experiment = ExperimentConfig(
        dataset=_coerce_named_section(DatasetSection, payload.get("dataset")),
        architecture=_coerce_named_section(ArchitectureSection, payload.get("architecture")),
        method=_coerce_named_section(MethodSection, payload.get("method")),
        runtime=RuntimeConfig(**runtime_payload),
        trainer=TrainerConfig(**dict(payload.get("trainer", {}))),
        evaluation=EvaluationConfig(**dict(payload.get("evaluation", {}))),
        output=OutputConfig(**dict(payload.get("output", {}))),
    )
    return experiment


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = normalize_path(path, resolve_latest=False)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    experiment = coerce_experiment_config(raw, source_path=config_path)

    if experiment.output.run_name is None:
        experiment.output.run_name = config_path.stem
    return experiment


def load_evaluation_config(
    path: str | Path,
    *,
    runtime: RuntimeConfig | dict[str, Any] | None = None,
) -> EvaluationConfig:
    config_path = normalize_path(path, resolve_latest=False)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    evaluation_payload = dict(raw.get("evaluation", raw))
    runtime_payload = asdict(runtime) if isinstance(runtime, RuntimeConfig) else dict(runtime or {})
    authored_runtime = dict(raw.get("runtime", {}))
    runtime_payload.update(authored_runtime)
    if "data_root" in authored_runtime and "mnist_root" not in authored_runtime:
        runtime_payload["mnist_root"] = None
    return coerce_evaluation_config(
        evaluation_payload,
        runtime=runtime_payload,
        source_path=config_path,
    )


def dump_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    output_path = normalize_path(path, resolve_latest=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
