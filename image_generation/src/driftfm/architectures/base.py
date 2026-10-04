from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn

from ..data import Conditioning

ConditioningMode = Literal["class"]


def resolve_conditioning_mode(
    conditioning_mode: str | None,
    *,
    num_classes: int | None,
) -> ConditioningMode | None:
    if num_classes is None:
        return None
    if conditioning_mode is None:
        return "class"
    if conditioning_mode not in {"class"}:
        raise ValueError(f"Unsupported conditioning_mode: {conditioning_mode}")
    return conditioning_mode


class VelocityModel(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        conditioning: Conditioning | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
