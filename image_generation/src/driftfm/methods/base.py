from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from ..architectures import VelocityModel
from ..data import Conditioning, TransportBatch


@dataclass(slots=True)
class LossOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


class GenerativeMethod(ABC):
    @property
    @abstractmethod
    def uses_ema(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def ema_decay(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(
        self,
        model: VelocityModel,
        batch: TransportBatch,
    ) -> LossOutput:
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        model: VelocityModel,
        x0: torch.Tensor,
        conditioning: Conditioning | None = None,
        *,
        num_steps: int,
    ) -> torch.Tensor:
        raise NotImplementedError
