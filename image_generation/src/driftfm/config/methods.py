from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DriftFlowMatchingConfig:
    groups_per_class: int = 4
    drift_form: str = "split_v0"
    use_ema: bool = True
    ema_decay: float = 0.999
    P_mean_t: float = -1.0
    P_std_t: float = 2.5
    P_mean_r: float = 1.0
    P_std_r: float = 2.5
    norm_eps: float = 1e-4
    norm_p: float = 0.0
    kernel_temp_pos: float = 1
    kernel_temp_neg: float = 1
    sinkhorn_iters: int = 20
