from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrainingHistory:
    losses: list[float]


@dataclass
class ExperimentArtifacts:
    model: torch.nn.Module
    history: TrainingHistory
