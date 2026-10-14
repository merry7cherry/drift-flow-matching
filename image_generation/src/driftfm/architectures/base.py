from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..data import Conditioning

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
