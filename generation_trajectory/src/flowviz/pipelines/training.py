from __future__ import annotations

from copy import deepcopy
from typing import Callable, List

import torch
from torch.func import jvp

from ..configs import (
    DriftFlowMatchingConfig,
    MeanFlowConfig,
    TrainingConfig,
)
from ..data.base import PairDataset
from ..flows.linear import LinearInterpolationFlow
from ..flows.time_sampling import (
    sample_drift_timestep,
    sample_two_timesteps_t_r_v1,
)
from ..models.mlp import (
    DriftFlowVelocityMLP,
    MeanVelocityMLP,
    VelocityMLP,
)
from ..training.trainer import TrainingHistory, train_model
from .artifacts import (
    DriftFlowMatchingExperimentArtifacts,
    ExperimentArtifacts,
    MeanFlowExperimentArtifacts,
)


def _sinkhorn_from_logits_batched(
    logits: torch.Tensor,
    r: torch.Tensor,
    c: torch.Tensor,
    iters: int = 20,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Solve a batch of entropic OT plans in log space.

    Args:
        logits: [B, Nx, Ny] pairwise transport logits.
        r: [Nx] source marginal shared by the whole batch.
        c: [Ny] target marginal shared by the whole batch.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, Nx, Ny]")
    if iters <= 0:
        raise ValueError("iters must be > 0")

    dtype = logits.dtype
    device = logits.device
    r = r.to(device=device, dtype=dtype)
    c = c.to(device=device, dtype=dtype)

    log_r = torch.log(r.clamp_min(eps))
    log_c = torch.log(c.clamp_min(eps))
    log_u = torch.zeros_like(logits[..., 0])
    log_v = torch.zeros_like(logits[:, 0, :])

    for _ in range(iters):
        log_u = log_r.unsqueeze(0) - torch.logsumexp(
            logits + log_v.unsqueeze(-2),
            dim=-1,
        )
        log_v = log_c.unsqueeze(0) - torch.logsumexp(
            logits + log_u.unsqueeze(-1),
            dim=-2,
        )

    return torch.exp(logits + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)).clamp_min(0.0)


def _row_normalize_plan(plan: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Turn a nonnegative transport plan [B, Nx, Ny] into row-stochastic weights."""

    if plan.ndim != 3:
        raise ValueError("plan must have shape [B, Nx, Ny]")
    return plan / plan.sum(dim=-1, keepdim=True).clamp_min(eps)


def compute_drift_batched_sinkhorn(
    gen: torch.Tensor,
    pos: torch.Tensor,
    temp_pos: float = 0.05,
    temp_neg: float = 0.05,
    sinkhorn_iters: int = 20,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute Sinkhorn split-form drift for grouped batches.

    Args:
        gen: [G, B, D] predicted query states grouped along the first axis.
        pos: [G, B_pos, D] positive reference states grouped the same way.
    """

    if gen.ndim != 3 or pos.ndim != 3:
        raise ValueError("gen and pos must be 3D tensors")
    if gen.shape[0] != pos.shape[0]:
        raise ValueError("gen and pos must share the same number of groups")
    if gen.shape[2] != pos.shape[2]:
        raise ValueError("gen and pos must share the same feature dimension")

    _, group_batch_size, _ = gen.shape
    if group_batch_size < 2:
        return torch.zeros_like(gen)

    pos_group_size = pos.shape[1]
    if pos_group_size < 1:
        raise ValueError("pos must be non-empty within each group")

    work_dtype = (
        torch.float32 if gen.dtype in (torch.float16, torch.bfloat16) else gen.dtype
    )
    query_points = gen.to(dtype=work_dtype)
    positive_points = pos.to(dtype=work_dtype)

    dist_pos = torch.cdist(query_points, positive_points)
    logits_pos = -dist_pos / float(temp_pos)
    row_marginal_pos = torch.full(
        (group_batch_size,),
        1.0 / float(group_batch_size),
        device=query_points.device,
        dtype=work_dtype,
    )
    col_marginal_pos = torch.full(
        (pos_group_size,),
        1.0 / float(pos_group_size),
        device=query_points.device,
        dtype=work_dtype,
    )
    plan_pos = _sinkhorn_from_logits_batched(
        logits_pos,
        r=row_marginal_pos,
        c=col_marginal_pos,
        iters=sinkhorn_iters,
        eps=eps,
    )
    weights_pos = _row_normalize_plan(plan_pos, eps=eps)
    drift_pos = torch.matmul(weights_pos, positive_points)

    # Each group defines an independent negative OT problem on its own queries.
    dist_neg = torch.cdist(query_points, query_points)
    logits_neg = -dist_neg / float(temp_neg)
    marginal_neg = torch.full(
        (group_batch_size,),
        1.0 / float(group_batch_size),
        device=query_points.device,
        dtype=work_dtype,
    )
    plan_neg = _sinkhorn_from_logits_batched(
        logits_neg,
        r=marginal_neg,
        c=marginal_neg,
        iters=sinkhorn_iters,
        eps=eps,
    )
    weights_neg = _row_normalize_plan(plan_neg, eps=eps)
    drift_neg = torch.matmul(weights_neg, query_points)

    return (drift_pos - drift_neg).to(dtype=gen.dtype)


def _reshape_grouped_batch(tensor: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Reshape a flat batch [B, ...] into grouped form [G, b, ...]."""

    if tensor.shape[0] % num_groups != 0:
        raise ValueError(
            "The batched drift implementation requires batch_size to be divisible "
            f"by batch_groups, but got batch_size={tensor.shape[0]} and "
            f"batch_groups={num_groups}."
        )

    group_batch_size = tensor.shape[0] // num_groups
    return tensor.reshape(num_groups, group_batch_size, *tensor.shape[1:])


def _build_ema_model(model: torch.nn.Module) -> torch.nn.Module:
    """Create a frozen shadow copy used for EMA evaluation."""

    ema_model = deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()
    return ema_model


@torch.no_grad()
def _update_ema_model(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    decay: float,
) -> None:
    """Update EMA parameters after one optimizer step."""

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param, alpha=1.0 - decay)

    # Keep buffers synchronized even though the current MLPs have no stateful buffers.
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def train_flow_matching(
    dataset: PairDataset,
    training_config: TrainingConfig,
    hidden_sizes: list[int] | None = None,
) -> ExperimentArtifacts:
    model = VelocityMLP(dim=dataset.dim, hidden_sizes=hidden_sizes)
    objective = LinearInterpolationFlow()
    history = train_model(model, dataset, objective, training_config)
    return ExperimentArtifacts(model=model, history=history)


def _compute_adaptive_matching_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    norm_eps: float,
    norm_p: float,
) -> torch.Tensor:
    """Apply the adaptive residual reduction to a batch of predictions [B, ...]."""

    loss_terms = (prediction - target) ** 2
    loss_terms = loss_terms.reshape(loss_terms.shape[0], -1).sum(dim=1)
    adaptive_weight = (loss_terms.detach() + norm_eps) ** norm_p
    return torch.mean(loss_terms / adaptive_weight)


def _expand_groupwise_timesteps(
    t_values: torch.Tensor,
    r_values: torch.Tensor,
    group_batch_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast one sampled (t, r) pair per group.

    Args:
        t_values: [] for one group or [G] for G groups.
        r_values: [] for one group or [G] for G groups.

    Returns:
        t, r: [B, 1] for scalar inputs or [G, B, 1] for grouped inputs.
    """

    if t_values.shape != r_values.shape:
        raise ValueError("t_values and r_values must have matching shapes")
    if group_batch_size <= 0:
        raise ValueError("group_batch_size must be positive")

    base_shape = (*t_values.shape, 1, 1) if t_values.ndim > 0 else (1, 1)
    expanded_shape = (
        (*t_values.shape, group_batch_size, 1)
        if t_values.ndim > 0
        else (group_batch_size, 1)
    )
    t = t_values.to(dtype=dtype).reshape(base_shape).expand(expanded_shape)
    r = r_values.to(dtype=dtype).reshape(base_shape).expand(expanded_shape)
    return t, r


def _compute_drift_interpolation_terms(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the drift FM interpolation states.

    Args:
        x0, x1: [B, D] or [G, B, D] paired endpoints.
        t, r: [B, 1] or [G, B, 1] group-aligned timesteps with t <= r.

    Returns:
        xt, xr, delta with the same leading dimensions as x0/x1.
    """

    xt = t * x1 + (1.0 - t) * x0
    xr = r * x1 + (1.0 - r) * x0
    delta = r - t
    return xt, xr, delta


def _flatten_grouped_samples(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten grouped tensors from [G, B, ...] to [G * B, ...]."""

    if tensor.ndim < 2:
        raise ValueError("tensor must have at least two leading dimensions [G, B, ...]")
    return tensor.reshape(-1, *tensor.shape[2:])


def _predict_drift_reference_state(
    model: DriftFlowVelocityMLP,
    xt: torch.Tensor,
    t: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict the reference state from flat drift inputs.

    Args:
        xt: [B, D] interpolated query states at time t.
        t: [B, 1] query times.
        delta: [B, 1] horizon h = r - t.
    """

    predicted_velocity = model(xt, t, delta)
    predicted_xr = xt + delta * predicted_velocity
    return predicted_velocity, predicted_xr


def _build_drift_matching_target(
    predicted_xr: torch.Tensor,
    xr: torch.Tensor,
    drift_kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Build the detached residual target used by drift flow matching.

    Args:
        predicted_xr: [B, D] or [G, B, D] current model prediction at time r.
        xr: [B, D] or [G, B, D] linear interpolation target at time r.
    """

    # Detach the query branch so Sinkhorn only defines the target, not the gradient path.
    drift_update = drift_kernel(predicted_xr.detach(), xr)
    return (predicted_xr + drift_update).detach()


def _run_optimizer_step(
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
    ema_decay: float,
) -> None:
    """Apply one optimizer step and keep EMA in sync when enabled."""

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if ema_model is not None:
        _update_ema_model(ema_model, model, ema_decay)


def _train_drift_flow_matching_batched(
    dataset: PairDataset,
    training_config: TrainingConfig,
    drift_config: DriftFlowMatchingConfig,
    model: DriftFlowVelocityMLP,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ema_model: torch.nn.Module | None = None,
) -> list[float]:
    """Batched drift FM path that evaluates all group drifts in one tensor pass."""

    if training_config.batch_size % drift_config.batch_groups != 0:
        raise ValueError(
            "drift_impl='batched' requires batch_size to be divisible by "
            f"batch_groups, but got batch_size={training_config.batch_size} "
            f"and batch_groups={drift_config.batch_groups}."
        )

    num_groups = drift_config.batch_groups
    group_batch_size = training_config.batch_size // num_groups
    losses: List[float] = []

    def compute_group_drift(
        predicted_xr_group: torch.Tensor,
        xr_group: torch.Tensor,
    ) -> torch.Tensor:
        return compute_drift_batched_sinkhorn(
            predicted_xr_group,
            xr_group,
            temp_pos=drift_config.kernel_temp_pos,
            temp_neg=drift_config.kernel_temp_neg,
            sinkhorn_iters=drift_config.sinkhorn_iters,
        )

    model.train()
    for _ in range(training_config.epochs):
        epoch_loss = 0.0
        for _ in range(training_config.steps_per_epoch):
            batch = dataset.sample_pairs(training_config.batch_size, device)
            x0 = batch.x0
            x1 = batch.x1

            group_t_raw, group_r_raw = sample_drift_timestep(
                drift_config, num_groups, device
            )

            x0_group = _reshape_grouped_batch(x0, num_groups)
            x1_group = _reshape_grouped_batch(x1, num_groups)

            t_group, r_group = _expand_groupwise_timesteps(
                group_t_raw,
                group_r_raw,
                group_batch_size,
                x0.dtype,
            )
            xt_group, xr_group, delta_group = _compute_drift_interpolation_terms(
                x0_group,
                x1_group,
                t_group,
                r_group,
            )

            # Flatten [G, b, ...] to [G * b, ...] so the model interface stays unchanged.
            xt_flat = _flatten_grouped_samples(xt_group)
            t_flat = _flatten_grouped_samples(t_group)
            delta_flat = _flatten_grouped_samples(delta_group)

            _, predicted_xr_flat = _predict_drift_reference_state(
                model,
                xt_flat,
                t_flat,
                delta_flat,
            )
            predicted_xr_group = predicted_xr_flat.reshape_as(x0_group)
            target_group = _build_drift_matching_target(
                predicted_xr_group,
                xr_group,
                compute_group_drift,
            )
            loss = _compute_adaptive_matching_loss(
                predicted_xr_flat, _flatten_grouped_samples(target_group),
                drift_config.norm_eps, drift_config.norm_p,
            )

            _run_optimizer_step(
                optimizer,
                loss,
                model,
                ema_model,
                drift_config.ema_decay,
            )

            epoch_loss += loss.item()

        steps = max(1, training_config.steps_per_epoch)
        losses.append(epoch_loss / steps)

    return losses


def train_drift_flow_matching(
    dataset: PairDataset,
    training_config: TrainingConfig,
    drift_config: DriftFlowMatchingConfig,
) -> DriftFlowMatchingExperimentArtifacts:
    """Train Drift Flow Matching with grouped, batched Sinkhorn drift."""

    device = torch.device(training_config.device)
    model = DriftFlowVelocityMLP(
        dim=dataset.dim, hidden_sizes=list(drift_config.velocity_hidden_sizes)
    ).to(device)
    ema_model = _build_ema_model(model) if drift_config.use_ema else None
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    losses = _train_drift_flow_matching_batched(
        dataset, training_config, drift_config, model, optimizer, device, ema_model,
    )

    history = TrainingHistory(losses=losses)
    artifact_model = ema_model if ema_model is not None else model
    return DriftFlowMatchingExperimentArtifacts(model=artifact_model, history=history)


def train_mean_flow_matching(
    dataset: PairDataset,
    training_config: TrainingConfig,
    mean_config: MeanFlowConfig,
) -> MeanFlowExperimentArtifacts:
    device = torch.device(training_config.device)
    model = MeanVelocityMLP(
        dim=dataset.dim, hidden_sizes=list(mean_config.velocity_hidden_sizes)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    losses: List[float] = []

    model.train()
    for _ in range(training_config.epochs):
        epoch_loss = 0.0
        for _ in range(training_config.steps_per_epoch):
            batch = dataset.sample_pairs(training_config.batch_size, device)
            t_raw, r_raw = sample_two_timesteps_t_r_v1(
                mean_config, training_config.batch_size, device
            )
            t = t_raw.view(-1, 1)
            r = r_raw.view(-1, 1)

            x0 = batch.x0
            x1 = batch.x1
            z = (1.0 - t) * x1 + t * x0
            v = x0 - x1

            def u_func(z_in: torch.Tensor, t_in: torch.Tensor, r_in: torch.Tensor) -> torch.Tensor:
                h_in = t_in - r_in
                return model(z_in, t_in, h_in)

            dtdt = torch.ones_like(t)
            drdt = torch.zeros_like(r)
            state_tangent = v

            predicted_velocity, velocity_time_derivative = jvp(
                u_func,
                (z, t, r),
                (state_tangent, dtdt, drdt),
            )

            u_target = (v - (t - r) * velocity_time_derivative).detach()
            loss_terms = (predicted_velocity - u_target) ** 2
            loss_terms = loss_terms.view(loss_terms.shape[0], -1).sum(dim=1)
            adaptive_weight = (loss_terms.detach() + mean_config.norm_eps) ** mean_config.norm_p
            loss = torch.mean(loss_terms / adaptive_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        steps = max(1, training_config.steps_per_epoch)
        losses.append(epoch_loss / steps)

    history = TrainingHistory(losses=losses)
    return MeanFlowExperimentArtifacts(model=model, history=history)


__all__ = ["train_flow_matching", "train_drift_flow_matching", "train_mean_flow_matching"]
