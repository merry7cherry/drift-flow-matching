from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class SampleBatch:
    x0: torch.Tensor
    x1: torch.Tensor


class PairDataset(ABC):
    """Abstract base class for synthetic datasets that provide (x0, x1) pairs."""

    def __init__(self, dim: int, seed: int = 42) -> None:
        self.dim = dim
        self.seed = seed
        self._generator = torch.Generator(device="cpu").manual_seed(seed)

    @abstractmethod
    def sample_base(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample from the base (noise) distribution."""

    @abstractmethod
    def sample_target(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample from the target data distribution."""

    def sample_pairs(self, batch_size: int, device: torch.device) -> SampleBatch:
        x0 = self.sample_base(batch_size, device)
        x1 = self.sample_target(batch_size, device)
        return SampleBatch(x0=x0, x1=x1)

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset the internal RNG to a deterministic state."""

        new_seed = self.seed if seed is None else seed
        self._generator = torch.Generator(device="cpu").manual_seed(new_seed)
