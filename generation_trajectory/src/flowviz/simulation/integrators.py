from __future__ import annotations

from typing import Tuple

import torch


class EulerIntegrator:
    """Explicit Euler integrator for flow trajectories."""

    def __init__(self, num_steps: int = 50) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        self.num_steps = num_steps

    @torch.no_grad()
    def integrate(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, dim = x0.shape
        dt = 1.0 / self.num_steps
        times = torch.linspace(0.0, 1.0, self.num_steps + 1, device=device)

        trajectory = torch.zeros(self.num_steps + 1, batch_size, dim, device=device)
        trajectory[0] = x0
        x = x0
        for step in range(self.num_steps):
            t = torch.full((batch_size, 1), times[step], device=device)
            velocity = model(x, t)
            x = x + dt * velocity
            trajectory[step + 1] = x
        return trajectory, times


__all__ = ["EulerIntegrator"]
