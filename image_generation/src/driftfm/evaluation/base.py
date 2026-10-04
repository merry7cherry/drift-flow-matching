from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig


@dataclass(slots=True)
class EvaluationResult:
    output_path: Path
    summary: dict[str, float]


class CheckpointEvaluator(ABC):
    name: str
    supported_metrics: tuple[str, ...]

    def validate_metrics(self, metrics: list[str]) -> list[str]:
        unsupported = sorted(set(metrics).difference(self.supported_metrics))
        if unsupported:
            raise ValueError(
                f"Unsupported metrics for evaluator={self.name}: {', '.join(unsupported)}. "
                f"Supported metrics: {', '.join(self.supported_metrics)}"
            )
        return list(metrics)

    @abstractmethod
    def evaluate(
        self,
        checkpoint_path: str | Path,
        *,
        experiment: ExperimentConfig,
        checkpoint: dict[str, Any],
        output_dir: str | Path,
        num_sampling_steps: int,
    ) -> EvaluationResult:
        raise NotImplementedError
