from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _normalize_num_sampling_steps(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("evaluation.num_sampling_steps must be a non-empty list of distinct positive integers")
    if len(value) == 0:
        raise ValueError("evaluation.num_sampling_steps must be a non-empty list of distinct positive integers")

    normalized: list[int] = []
    seen: set[int] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("evaluation.num_sampling_steps must contain only integers")
        if item <= 0:
            raise ValueError("evaluation.num_sampling_steps must contain only positive integers")
        if item in seen:
            raise ValueError("evaluation.num_sampling_steps must contain distinct integers")
        normalized.append(item)
        seen.add(item)
    return normalized


@dataclass(slots=True)
class RuntimeConfig:
    project_name: str | None = None
    data_root: str | None = None
    runs_root: str | None = None
    mnist_root: str | None = None


@dataclass(slots=True)
class NamedSection:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetSection(NamedSection):
    pass


@dataclass(slots=True)
class ArchitectureSection(NamedSection):
    pass


@dataclass(slots=True)
class MethodSection(NamedSection):
    pass


@dataclass(slots=True)
class TrainerConfig:
    epochs: int = 200
    batch_size: int = 256
    steps_per_epoch: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    device: str = "cuda"
    seed: int = 42
    deterministic: bool = True
    log_every: int = 10
    checkpoint_every: int = 25


@dataclass(slots=True)
class EvaluationConfig:
    evaluator: str | None = None
    num_sampling_steps: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50])
    sample_every: int = 25
    sample_count_per_class: int = 8
    metrics_every: int = 0
    run_final: bool = True
    metrics: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.num_sampling_steps = _normalize_num_sampling_steps(self.num_sampling_steps)


@dataclass(slots=True)
class OutputConfig:
    root_dir: str = "{runs_root}"
    run_name: str | None = None


@dataclass(slots=True)
class ExperimentConfig:
    dataset: DatasetSection
    architecture: ArchitectureSection
    method: MethodSection
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
