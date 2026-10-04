from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    steps_per_epoch: int = 50
    learning_rate: float = 1e-3
    device: str = "cpu"


@dataclass
class IntegratorConfig:
    num_steps: int = 50


@dataclass
class VisualizationEvalConfig:
    nfe: tuple[int, ...] = (1, 20)
    eval_samples: int = 512
    max_display: int = 128
