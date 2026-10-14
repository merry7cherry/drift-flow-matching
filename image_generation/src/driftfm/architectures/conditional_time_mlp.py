from __future__ import annotations

import torch
import torch.nn as nn

from ..data import Conditioning
from .base import VelocityModel


def _build_activation(name: str) -> nn.Module:
    lowered = name.lower()
    if lowered == "relu":
        return nn.ReLU(inplace=True)
    if lowered == "gelu":
        return nn.GELU()
    if lowered == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class ConditionalTimeMLP(VelocityModel):
    def __init__(
        self,
        *,
        data_dim: int,
        hidden_sizes: list[int] | tuple[int, ...],
        num_classes: int,
        class_embedding_dim: int = 32,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.num_classes = num_classes
        self.class_embedding_dim = class_embedding_dim
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)
        input_dim = data_dim + 2 + self.class_embedding_dim
        layer_sizes = [input_dim, *hidden_sizes, data_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != layer_sizes[-1]:
                layers.append(_build_activation(activation))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        conditioning: Conditioning | None = None,
    ) -> torch.Tensor:
        if conditioning is None or conditioning.class_labels is None:
            raise ValueError("conditioning.class_labels is required for class-conditional generation")

        original_shape = x_t.shape
        x_flat = x_t.reshape(x_t.shape[0], -1)
        t_flat = t.reshape(t.shape[0], -1)
        h_flat = h.reshape(h.shape[0], -1)
        if t_flat.shape[0] != x_flat.shape[0] or h_flat.shape[0] != x_flat.shape[0]:
            raise ValueError(
                "x_t, t, and h must share the same batch axis, "
                f"got x_t={tuple(x_t.shape)}, t={tuple(t.shape)}, h={tuple(h.shape)}"
            )
        pieces = [x_flat, t_flat, h_flat]

        # Loss/sampling entrypoints validate labels once, outside the NFE loop.
        class_features = self.class_embedding(conditioning.class_labels).to(dtype=x_flat.dtype)
        pieces.append(class_features)
        output = self.network(torch.cat(pieces, dim=-1))
        return output.reshape(original_shape)
