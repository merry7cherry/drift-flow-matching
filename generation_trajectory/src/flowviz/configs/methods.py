from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MeanFlowConfig:
    """Hyper-parameters for training the mean flow objective."""

    velocity_hidden_sizes: tuple[int, ...] = (128, 128, 128)
    P_mean_t: float = -0.6
    P_std_t: float = 1.6
    P_mean_r: float = -2.0
    P_std_r: float = 1.6
    ratio: float = 0.75
    norm_eps: float = 1e-4
    norm_p: float = 0.75


@dataclass
class DriftFlowMatchingConfig:
    """Hyper-parameters for training the drift flow matching objective."""

    velocity_hidden_sizes: tuple[int, ...] = (128, 128, 128)
    batch_groups: int = 4
    use_ema: bool = True
    ema_decay: float = 0.999
    P_mean_t: float = -1.0
    P_std_t: float = 2.5
    P_mean_r: float = 1.0
    P_std_r: float = 2.5
    norm_eps: float = 1e-4
    norm_p: float = 0.0
    kernel_temp_pos: float = 0.01
    kernel_temp_neg: float = 0.01
    sinkhorn_iters: int = 5


__all__ = [
    "DriftFlowMatchingConfig",
    "MeanFlowConfig",
]
