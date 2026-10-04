from __future__ import annotations

from dataclasses import dataclass

import torch

from ..training.trainer import TrainingHistory


@dataclass
class ExperimentArtifacts:
    model: torch.nn.Module
    history: TrainingHistory


@dataclass
class MeanFlowExperimentArtifacts:
    model: torch.nn.Module
    history: TrainingHistory


@dataclass
class DriftFlowMatchingExperimentArtifacts:
    model: torch.nn.Module
    history: TrainingHistory


__all__ = [
    "ExperimentArtifacts",
    "MeanFlowExperimentArtifacts",
    "DriftFlowMatchingExperimentArtifacts",
]
