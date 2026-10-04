from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layer_sizes, activation: type[nn.Module] = nn.SiLU) -> None:
        super().__init__()
        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != layer_sizes[-1]:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VelocityMLP(nn.Module):
    def __init__(self, dim: int, hidden_sizes: list[int] | None = None) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 128, 128]
        layer_sizes = [dim + 1, *hidden_sizes, dim]
        self.mlp = MLP(layer_sizes)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        t_flat = t.view(t.shape[0], -1)
        return self.mlp(torch.cat((x_flat, t_flat), dim=1))


class MeanVelocityMLP(nn.Module):
    """Velocity network that conditions on both t and the offset h = t - r."""

    def __init__(self, dim: int, hidden_sizes: list[int] | None = None) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 128, 128]
        layer_sizes = [dim + 2, *hidden_sizes, dim]
        self.mlp = MLP(layer_sizes)

    def forward(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        t_flat = t.view(t.shape[0], -1)
        h_flat = h.view(h.shape[0], -1)
        velocity = self.mlp(torch.cat((x_flat, t_flat, h_flat), dim=1))
        return velocity.reshape_as(x)


class DriftFlowVelocityMLP(nn.Module):
    """Predict drift velocity u from x_t conditioned on t and h = r - t."""

    def __init__(self, dim: int, hidden_sizes: list[int] | None = None) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 128, 128]
        layer_sizes = [dim + 2, *hidden_sizes, dim]
        self.mlp = MLP(layer_sizes)

    def forward(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        t_flat = t.view(t.shape[0], -1)
        h_flat = h.view(h.shape[0], -1)
        predicted = self.mlp(torch.cat((x_flat, t_flat, h_flat), dim=1))
        return predicted.reshape_as(x)


__all__ = [
    "VelocityMLP",
    "MeanVelocityMLP",
    "DriftFlowVelocityMLP",
]
