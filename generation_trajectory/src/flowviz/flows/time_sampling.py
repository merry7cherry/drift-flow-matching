from __future__ import annotations

import torch

from ..configs import DriftFlowMatchingConfig, MeanFlowConfig


def logit_normal_timestep_sample(
    P_mean: float, P_std: float, num_samples: int, device: torch.device
) -> torch.Tensor:
    rnd_normal = torch.randn((num_samples,), device=device)
    time = torch.sigmoid(rnd_normal * P_std + P_mean)
    time = torch.clamp(time, min=0.0, max=1.0)
    return time


def sample_two_timesteps_t_r_v1(
    config: MeanFlowConfig,
    num_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (t, r) pairs with post-processing to enforce t >= r."""

    t = logit_normal_timestep_sample(config.P_mean_t, config.P_std_t, num_samples, device)
    r = logit_normal_timestep_sample(config.P_mean_r, config.P_std_r, num_samples, device)

    prob = torch.rand(num_samples, device=device)
    mask = prob < 1 - config.ratio
    r = torch.where(mask, t, r)

    r = torch.minimum(t, r)
    return t, r


def sample_drift_timestep(
    config: DriftFlowMatchingConfig,
    num_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (t, r) for drift flow matching with 0 < t <= r < 1."""

    t = logit_normal_timestep_sample(config.P_mean_t, config.P_std_t, num_samples, device)
    r = logit_normal_timestep_sample(config.P_mean_r, config.P_std_r, num_samples, device)

    t, r = torch.minimum(t, r), torch.maximum(t, r)

    return t, r


__all__ = [
    "logit_normal_timestep_sample",
    "sample_two_timesteps_t_r_v1",
    "sample_drift_timestep",
]
