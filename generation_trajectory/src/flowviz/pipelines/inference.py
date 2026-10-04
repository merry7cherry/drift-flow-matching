from __future__ import annotations

from typing import Tuple

import torch

from ..configs import IntegratorConfig
from ..models.mlp import (
    DriftFlowVelocityMLP,
    MeanVelocityMLP,
    VelocityMLP,
)
from ..simulation.integrators import EulerIntegrator


def compute_model_trajectories(
    model: torch.nn.Module,
    x0: torch.Tensor,
    device: torch.device,
    integrator_config: IntegratorConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    integrator = EulerIntegrator(num_steps=integrator_config.num_steps)
    with torch.no_grad():
        trajectory, times = integrator.integrate(model, x0.to(device), device)
    return trajectory, times


def compute_drift_flow_matching_trajectories(
    model: DriftFlowVelocityMLP,
    x0: torch.Tensor,
    device: torch.device,
    *,
    steps: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate Drift FM trajectories with x_r = x_t + (r - t)u(x_t, t, r)."""

    if steps < 1:
        raise ValueError("steps must be a positive integer")

    model.eval()
    with torch.no_grad():
        base = x0.to(device)
        batch_size = base.shape[0]
        times = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=base.dtype)

        states = [base]
        current = base

        def u_func(
            xt_in: torch.Tensor, t_in: torch.Tensor, r_in: torch.Tensor
        ) -> torch.Tensor:
            h_in = r_in - t_in
            return model(xt_in, t_in, h_in)

        for idx in range(steps):
            t_scalar = times[idx]
            r_scalar = times[idx + 1]
            t = torch.full(
                (batch_size, 1),
                float(t_scalar.item()),
                device=device,
                dtype=base.dtype,
            )
            r = torch.full(
                (batch_size, 1),
                float(r_scalar.item()),
                device=device,
                dtype=base.dtype,
            )
            velocity = u_func(current, t, r)
            current = current + (r - t) * velocity
            states.append(current)

        trajectory = torch.stack(states, dim=0)

    return trajectory, times


def compute_mean_flow_trajectories(
    model: MeanVelocityMLP,
    x0: torch.Tensor,
    device: torch.device,
    *,
    steps: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate mean-flow trajectories using uniformly spaced inference steps."""

    if steps < 1:
        raise ValueError("steps must be a positive integer")

    model.eval()
    with torch.no_grad():
        base = x0.to(device)
        batch_size = base.shape[0]
        times = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=base.dtype)

        states = [base]
        current = base

        for idx in range(steps):
            current_time = times[idx]
            evaluation_time = 1.0 - current_time
            next_time = times[idx + 1]
            reference_time = 1.0 - next_time
            t = torch.ones((batch_size, 1), device=device, dtype=base.dtype) * evaluation_time
            r = torch.ones_like(t) * reference_time
            h = t - r

            velocity = model(current, t, h)
            dt = times[idx + 1] - current_time
            current = current - velocity * dt
            states.append(current)

        trajectory = torch.stack(states, dim=0)

    return trajectory, times


__all__ = [
    "compute_model_trajectories",
    "compute_drift_flow_matching_trajectories",
    "compute_mean_flow_trajectories",
]
