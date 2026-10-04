from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import torch


class TimeSampler(Protocol):
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        ...


class UniformTimeSampler:
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, 1, device=device)


class FlowMatchingObjective(ABC):
    def __init__(self, time_sampler: TimeSampler | None = None) -> None:
        self.time_sampler = time_sampler or UniformTimeSampler()

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.time_sampler.sample(batch_size, device)

    @abstractmethod
    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return x_t given x0, x1 and interpolation time."""

    @abstractmethod
    def target_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return the target velocity used for training."""


__all__ = [
    "FlowMatchingObjective",
    "UniformTimeSampler",
]
