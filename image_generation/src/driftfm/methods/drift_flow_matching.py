from __future__ import annotations

from dataclasses import asdict

import torch

from ..architectures import VelocityModel
from ..config import DriftFlowMatchingConfig
from ..data import Conditioning, TransportBatch
from .base import GenerativeMethod, LossOutput

# Tensor shape conventions used throughout this module:
# N: flat batch size
# C: number of classes in the balanced batch
# S: subgroups per class
# B: samples per subgroup
# G: flattened group count, equal to C * S
# D: flattened feature dimension
# *shape: original data shape, such as image or latent dimensions

_SUPPORTED_DRIFT_FORMS = {"split_v0"}


def conditioning_mode_for_drift_form(drift_form: str) -> str:
    if drift_form != "split_v0":
        raise ValueError("This release supports the class-conditional split_v0 formulation only")
    return "class"


def logit_normal_timestep_sample(
    P_mean: float,
    P_std: float,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample bounded timesteps from a logit-normal law.

    Args:
        P_mean: Logit-space mean.
        P_std: Logit-space standard deviation.
        num_samples: Number of samples `N`.
        device: Output device.

    Returns:
        Timesteps with shape `[N]` in `[0, 1]`.
    """
    random_values = torch.randn((num_samples,), device=device)
    sampled = torch.sigmoid(random_values * P_std + P_mean)
    return torch.clamp(sampled, min=0.0, max=1.0)


def sample_drift_timestep(
    config: DriftFlowMatchingConfig,
    num_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ordered drift endpoints.

    Args:
        config: Drift Flow Matching config.
        num_samples: Number of samples `N`.
        device: Output device.

    Returns:
        `(t, r)`, each with shape `[N]` and `t <= r`.
    """
    t = logit_normal_timestep_sample(config.P_mean_t, config.P_std_t, num_samples, device)
    r = logit_normal_timestep_sample(config.P_mean_r, config.P_std_r, num_samples, device)
    return torch.minimum(t, r), torch.maximum(t, r)


def _expand_marginals(
    marginals: torch.Tensor,
    *,
    expected_size: int,
    leading_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Broadcast Sinkhorn marginals to `leading_shape + [expected_size]`.

    Args:
        marginals: `[expected_size]` or `leading_shape + [expected_size]`.
        expected_size: Required final axis size.
        leading_shape: Broadcasted batch shape.
        device: Target device.
        dtype: Target dtype.
        name: Error-label for validation.

    Returns:
        Expanded marginals with shape `leading_shape + [expected_size]`.
    """
    marginals = marginals.to(device=device, dtype=dtype)
    if marginals.ndim == 1:
        if marginals.shape[0] != expected_size:
            raise ValueError(f"{name} must have shape [{expected_size}], got {tuple(marginals.shape)}")
        view_shape = (1,) * len(leading_shape) + (expected_size,)
        return marginals.reshape(view_shape).expand(*leading_shape, expected_size)
    if marginals.shape[:-1] != leading_shape or marginals.shape[-1] != expected_size:
        raise ValueError(
            f"{name} must have shape {leading_shape + (expected_size,)}, got {tuple(marginals.shape)}"
        )
    return marginals


def _sinkhorn_from_logits(
    logits: torch.Tensor,
    *,
    row_marginals: torch.Tensor,
    col_marginals: torch.Tensor,
    num_iters: int = 20,
    eps: float = 1e-12,
    return_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Solve Sinkhorn transport plans from pairwise logits.

    Args:
        logits: Pairwise scores with shape `[..., N_src, N_tgt]`.
        row_marginals: Source marginals with shape `[N_src]` or `[..., N_src]`.
        col_marginals: Target marginals with shape `[N_tgt]` or `[..., N_tgt]`.
        num_iters: Sinkhorn iterations.
        eps: Numerical clamp.
        return_dtype: Optional output dtype.

    Returns:
        Transport plan with shape `[..., N_src, N_tgt]`.
    """
    if logits.ndim < 2:
        raise ValueError(f"logits must have shape [..., N_src, N_tgt], got {tuple(logits.shape)}")
    if num_iters <= 0:
        raise ValueError(f"num_iters must be > 0, got {num_iters}")

    num_source_points, num_target_points = logits.shape[-2], logits.shape[-1]
    orig_dtype = logits.dtype
    work_dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    if return_dtype is None:
        return_dtype = orig_dtype

    logits = logits.to(dtype=work_dtype)
    leading_shape = logits.shape[:-2]
    row_marginals = _expand_marginals(
        row_marginals,
        expected_size=num_source_points,
        leading_shape=leading_shape,
        device=logits.device,
        dtype=work_dtype,
        name="row_marginals",
    )
    col_marginals = _expand_marginals(
        col_marginals,
        expected_size=num_target_points,
        leading_shape=leading_shape,
        device=logits.device,
        dtype=work_dtype,
        name="col_marginals",
    )

    log_row_marginals = torch.log(row_marginals.clamp_min(eps))
    log_col_marginals = torch.log(col_marginals.clamp_min(eps))
    log_u = torch.zeros_like(logits[..., :, 0])
    log_v = torch.zeros_like(logits[..., 0, :])

    for _ in range(int(num_iters)):
        log_u = log_row_marginals - torch.logsumexp(logits + log_v.unsqueeze(-2), dim=-1)
        log_v = log_col_marginals - torch.logsumexp(logits + log_u.unsqueeze(-1), dim=-2)

    plan = torch.exp(logits + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)).clamp_min(0.0)
    return plan.to(dtype=return_dtype)


def _row_normalize_plan(plan: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize a transport plan row-wise.

    Args:
        plan: Transport plan with shape `[..., N_src, N_tgt]`.
        eps: Numerical clamp.

    Returns:
        Row-normalized weights with shape `[..., N_src, N_tgt]`.
    """
    return plan / plan.sum(dim=-1, keepdim=True).clamp_min(eps)


def _compute_sinkhorn_barycentric_projection(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    kernel_temp: float,
    num_iters: int,
    eps: float,
    row_marginals: torch.Tensor | None = None,
    col_marginals: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project grouped source points onto Sinkhorn barycenters.

    Args:
        source_points: Source block with shape `[G, N_src, D]`.
        target_points: Target block with shape `[G, N_tgt, D]`.
        kernel_temp: Sinkhorn kernel temperature.
        num_iters: Sinkhorn iterations.
        eps: Numerical clamp.
        row_marginals: Optional source marginals with shape `[N_src]` or `[G, N_src]`.
        col_marginals: Optional target marginals with shape `[N_tgt]` or `[G, N_tgt]`.

    Returns:
        Barycentric projections with shape `[G, N_src, D]`.
    """
    if source_points.ndim != 3 or target_points.ndim != 3:
        raise ValueError("source_points and target_points must have shape [G, N, D]")
    if source_points.shape[0] != target_points.shape[0]:
        raise ValueError("source_points and target_points must share the same G axis")
    if target_points.shape[1] == 0:
        return torch.zeros_like(source_points)

    orig_dtype = source_points.dtype
    work_dtype = torch.float32 if source_points.dtype in (torch.float16, torch.bfloat16) else source_points.dtype
    source_points = source_points.to(dtype=work_dtype)
    target_points = target_points.to(dtype=work_dtype)
    if row_marginals is None:
        row_marginals = torch.full(
            (source_points.shape[1],),
            1.0 / float(source_points.shape[1]),
            device=source_points.device,
            dtype=work_dtype,
        )
    if col_marginals is None:
        col_marginals = torch.full(
            (target_points.shape[1],),
            1.0 / float(target_points.shape[1]),
            device=target_points.device,
            dtype=work_dtype,
        )
    distances = torch.cdist(source_points, target_points)
    logits = -distances / float(kernel_temp)
    plan = _sinkhorn_from_logits(
        logits,
        row_marginals=row_marginals,
        col_marginals=col_marginals,
        num_iters=num_iters,
        eps=eps,
    )
    weights = _row_normalize_plan(plan, eps=eps)
    return torch.matmul(weights, target_points).to(dtype=orig_dtype)


def compute_drift_grouped_sinkhorn(
    gen: torch.Tensor,
    pos: torch.Tensor,
    *,
    drift_form: str = "split_v0",
    kernel_temp_pos: float = 0.05,
    kernel_temp_neg: float = 0.05,
    num_sinkhorn_iters: int = 20,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Same-class attraction minus generated-sample repulsion, shape [G, B, D]."""
    conditioning_mode_for_drift_form(drift_form)
    if gen.ndim != 3 or pos.shape != gen.shape:
        raise ValueError("gen and pos must share shape [G, B, D]")
    if kernel_temp_pos <= 0 or kernel_temp_neg <= 0:
        raise ValueError("Sinkhorn kernel temperatures must be positive")
    pos_proj = _compute_sinkhorn_barycentric_projection(
        gen, pos, kernel_temp=kernel_temp_pos, num_iters=num_sinkhorn_iters, eps=eps,
    )
    neg_proj = _compute_sinkhorn_barycentric_projection(
        gen, gen, kernel_temp=kernel_temp_neg, num_iters=num_sinkhorn_iters, eps=eps,
    )
    return pos_proj - neg_proj


def _adaptive_matching_loss(
    predicted_points: torch.Tensor,
    target_points: torch.Tensor,
    norm_eps: float,
    norm_p: float,
) -> torch.Tensor:
    """Compute adaptive squared-error matching loss.

    Args:
        predicted_points: Predictions with shape `[N, *shape]`.
        target_points: Targets with shape `[N, *shape]`.
        norm_eps: Stability constant.
        norm_p: Adaptive weighting exponent.

    Returns:
        Scalar loss tensor.
    """
    terms = (predicted_points - target_points) ** 2
    terms = terms.reshape(terms.shape[0], -1).sum(dim=1)
    adaptive_weight = (terms.detach() + norm_eps) ** norm_p
    return torch.mean(terms / adaptive_weight)


def _validate_flat_class_labels(
    class_labels: torch.Tensor,
    *,
    expected_batch_size: int,
    min_allowed_label: int,
    max_allowed_label: int,
    label_name: str,
) -> None:
    """Validate flattened class labels.

    Args:
        class_labels: Flat labels with shape `[N]`.
        expected_batch_size: Required batch size `N`.
        min_allowed_label: Minimum allowed label.
        max_allowed_label: Maximum allowed label.
        label_name: Error-label for validation.
    """
    if class_labels.ndim != 1 or class_labels.shape[0] != expected_batch_size:
        raise ValueError(f"{label_name} must have shape [{expected_batch_size}], got {tuple(class_labels.shape)}")
    if class_labels.numel() == 0:
        raise ValueError(f"{label_name} must be non-empty")
    min_label = int(class_labels.min().item())
    max_label = int(class_labels.max().item())
    if min_label < min_allowed_label or max_label > max_allowed_label:
        raise ValueError(
            f"{label_name} must stay in [{min_allowed_label}, {max_allowed_label}], "
            f"got min={min_label}, max={max_label}"
        )


def _sort_by_class_label(batch: TransportBatch) -> TransportBatch:
    """Sort a transport batch by `conditioning.class_labels`.

    Args:
        batch: Flat batch with `x0`, `x1`, and `class_labels` on axis `N`.

    Returns:
        Batch sorted by ascending class label.
    """
    if batch.conditioning is None or batch.conditioning.class_labels is None:
        raise ValueError("Conditioning.class_labels is required for batched drift flow matching")
    class_labels = batch.conditioning.class_labels
    if class_labels.ndim != 1 or class_labels.shape[0] != batch.x0.shape[0] or batch.x1.shape[0] != batch.x0.shape[0]:
        raise ValueError(
            "x0, x1, and conditioning.class_labels must share the same flat batch axis, "
            f"got x0={tuple(batch.x0.shape)}, x1={tuple(batch.x1.shape)}, labels={tuple(class_labels.shape)}"
        )
    if class_labels.shape[0] <= 1 or torch.all(class_labels[1:] >= class_labels[:-1]):
        return batch

    sort_index = torch.argsort(class_labels)
    conditioning = batch.conditioning
    return TransportBatch(
        x0=batch.x0[sort_index],
        x1=batch.x1[sort_index],
        conditioning=Conditioning(
            class_labels=conditioning.class_labels[sort_index],
        ),
    )


def _reshape_flat_groups_to_class_subgroups(
    flat_tensor: torch.Tensor,
    *,
    num_classes: int,
    samples_per_class: int,
    subgroups_per_class: int,
) -> torch.Tensor:
    """Reshape `[N, *shape]` into `[C, S, B, *shape]`.

    Args:
        flat_tensor: Flat tensor with leading batch axis `N`.
        num_classes: Class count `C`.
        samples_per_class: Samples per class.
        subgroups_per_class: Subgroups per class `S`.

    Returns:
        Tensor with shape `[C, S, B, *shape]`.
    """
    subgroup_size = samples_per_class // subgroups_per_class
    return flat_tensor.reshape(
        num_classes,
        samples_per_class,
        *flat_tensor.shape[1:],
    ).reshape(
        num_classes,
        subgroups_per_class,
        subgroup_size,
        *flat_tensor.shape[1:],
    )


def _flatten_class_subgroups_to_groups(class_subgroup_tensor: torch.Tensor) -> torch.Tensor:
    """Flatten class and subgroup axes.

    Args:
        class_subgroup_tensor: Tensor with shape `[C, S, ...]`.

    Returns:
        Tensor with shape `[G, ...]` where `G = C * S`.
    """
    if class_subgroup_tensor.ndim < 3:
        raise ValueError("class_subgroup_tensor must have shape [C, S, ...]")
    return class_subgroup_tensor.reshape(
        class_subgroup_tensor.shape[0] * class_subgroup_tensor.shape[1],
        *class_subgroup_tensor.shape[2:],
    )


def _group_batch_into_class_subgroups(
    batch: TransportBatch,
    subgroups_per_class: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group a balanced batch into explicit class and subgroup axes.

    Args:
        batch: Flat batch with `x0`, `x1`, and class labels on axis `N`.
        subgroups_per_class: Subgroup count `S` per class.

    Returns:
        `(x0, x1, class_labels)` with shapes `[C, S, B, *shape]`, `[C, S, B, *shape]`, and `[C, S, B]`.
    """
    if subgroups_per_class <= 0:
        raise ValueError("groups_per_class must be positive")
    sorted_batch = _sort_by_class_label(batch)
    assert sorted_batch.conditioning is not None and sorted_batch.conditioning.class_labels is not None
    sorted_labels = sorted_batch.conditioning.class_labels
    unique, counts = torch.unique_consecutive(sorted_labels, return_counts=True)
    if unique.numel() == 0:
        raise ValueError("class_labels must be non-empty")
    if counts.min() != counts.max():
        raise ValueError("All classes must contribute the same number of samples for batched drift flow matching")

    samples_per_class = int(counts[0].item())
    if samples_per_class % subgroups_per_class != 0:
        raise ValueError("Per-class sample count must be divisible by groups_per_class")

    num_classes = unique.numel()
    x0_class_subgroups = _reshape_flat_groups_to_class_subgroups(
        sorted_batch.x0,
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        subgroups_per_class=subgroups_per_class,
    )
    x1_class_subgroups = _reshape_flat_groups_to_class_subgroups(
        sorted_batch.x1,
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        subgroups_per_class=subgroups_per_class,
    )
    class_labels_class_subgroups = _reshape_flat_groups_to_class_subgroups(
        sorted_labels,
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        subgroups_per_class=subgroups_per_class,
    )
    return x0_class_subgroups, x1_class_subgroups, class_labels_class_subgroups


def _sample_shared_subgroup_timesteps(
    config: DriftFlowMatchingConfig,
    *,
    num_classes: int,
    subgroups_per_class: int,
    subgroup_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample subgroup-shared drift endpoints and broadcast across classes.

    Args:
        config: Drift Flow Matching config.
        num_classes: Class count `C`.
        subgroups_per_class: Subgroup count `S`.
        subgroup_size: Samples per subgroup `B`.
        device: Output device.
        dtype: Output dtype.

    Returns:
        `(t, r)` with shape `[C, S, B, 1]`.
    """
    t_subgroups, r_subgroups = sample_drift_timestep(config, subgroups_per_class, device)
    target_shape = (num_classes, subgroups_per_class, subgroup_size, 1)
    t_class_subgroups = t_subgroups.view(1, subgroups_per_class, 1, 1).expand(target_shape).to(dtype=dtype)
    r_class_subgroups = r_subgroups.view(1, subgroups_per_class, 1, 1).expand(target_shape).to(dtype=dtype)
    return t_class_subgroups, r_class_subgroups


def _flatten_group_features(group_tensor: torch.Tensor) -> torch.Tensor:
    """Flatten grouped feature axes after `[G, N]`.

    Args:
        group_tensor: Tensor with shape `[G, N, *shape]`.

    Returns:
        Tensor with shape `[G, N, D]`.
    """
    if group_tensor.ndim < 3:
        raise ValueError("group_tensor must have shape [G, N, ...]")
    return group_tensor.reshape(group_tensor.shape[0], group_tensor.shape[1], -1)


def _validate_supported_config(config: DriftFlowMatchingConfig) -> None:
    """Validate the currently supported Sinkhorn drift configuration.

    Args:
        config: Drift Flow Matching config.
    """
    if config.drift_form not in _SUPPORTED_DRIFT_FORMS:
        supported = ", ".join(sorted(_SUPPORTED_DRIFT_FORMS))
        raise ValueError(f"Only drift_form in {{{supported}}} is supported, got {config.drift_form!r}")
    if config.sinkhorn_iters <= 0:
        raise ValueError("sinkhorn_iters must be > 0")


class DriftFlowMatchingMethod(GenerativeMethod):
    def __init__(self, config: DriftFlowMatchingConfig | None = None) -> None:
        self.config = config or DriftFlowMatchingConfig()
        _validate_supported_config(self.config)

    @property
    def uses_ema(self) -> bool:
        return self.config.use_ema

    @property
    def ema_decay(self) -> float:
        return self.config.ema_decay

    def to_config_dict(self) -> dict[str, object]:
        return asdict(self.config)

    def compute_loss(
        self,
        model: VelocityModel,
        batch: TransportBatch,
    ) -> LossOutput:
        """Compute Drift Flow Matching loss on a balanced class-conditional batch.

        Args:
            model: Velocity network mapping `(x_t, t, h, conditioning)` to velocity.
            batch: Transport batch with flat leading axis `[N, *shape]`.

        Returns:
            Scalar loss plus metrics.
        """
        x0_class_subgroups, x1_class_subgroups, class_labels_class_subgroups = _group_batch_into_class_subgroups(
            batch,
            self.config.groups_per_class,
        )
        num_classes, subgroups_per_class, subgroup_size = x0_class_subgroups.shape[:3]

        t_class_subgroups, r_class_subgroups = _sample_shared_subgroup_timesteps(
            self.config,
            num_classes=num_classes,
            subgroups_per_class=subgroups_per_class,
            subgroup_size=subgroup_size,
            device=batch.x0.device,
            dtype=batch.x0.dtype,
        )
        h_class_subgroups = r_class_subgroups - t_class_subgroups
        x_t_class_subgroups = t_class_subgroups * x1_class_subgroups + (1.0 - t_class_subgroups) * x0_class_subgroups
        x_r_class_subgroups = r_class_subgroups * x1_class_subgroups + (1.0 - r_class_subgroups) * x0_class_subgroups
        x_t = _flatten_class_subgroups_to_groups(x_t_class_subgroups).reshape(-1, *batch.x0.shape[1:])
        t = _flatten_class_subgroups_to_groups(t_class_subgroups).reshape(-1, 1)
        h = _flatten_class_subgroups_to_groups(h_class_subgroups).reshape(-1, 1)
        class_labels = class_labels_class_subgroups.reshape(-1)
        _validate_flat_class_labels(
            class_labels,
            expected_batch_size=batch.x0.shape[0],
            min_allowed_label=0,
            max_allowed_label=num_classes - 1,
            label_name="batch.conditioning.class_labels",
        )
        cond = Conditioning(class_labels=class_labels)

        predicted_velocity = model(x_t, t, h, conditioning=cond)
        x_r_pred = x_t + h * predicted_velocity
        x_r_pred_class_subgroups = x_r_pred.reshape_as(x_t_class_subgroups)
        gen = _flatten_group_features(_flatten_class_subgroups_to_groups(x_r_pred_class_subgroups.detach()))
        pos = _flatten_group_features(_flatten_class_subgroups_to_groups(x_r_class_subgroups))
        drift = compute_drift_grouped_sinkhorn(
            gen=gen,
            pos=pos,
            drift_form=self.config.drift_form,
            kernel_temp_pos=self.config.kernel_temp_pos,
            kernel_temp_neg=self.config.kernel_temp_neg,
            num_sinkhorn_iters=self.config.sinkhorn_iters,
        )
        target_x_r = (gen + drift).detach()
        group_shape = _flatten_class_subgroups_to_groups(x_r_pred_class_subgroups).shape
        target_x_r = target_x_r.reshape(group_shape).reshape_as(x_r_pred)

        drift_loss = _adaptive_matching_loss(
            x_r_pred,
            target_x_r,
            self.config.norm_eps,
            self.config.norm_p,
        )
        total_loss = drift_loss

        metrics = {
            "loss": float(total_loss.detach().item()),
            "drift_loss": float(drift_loss.detach().item()),
            "conditional_mass_mean": 1.0,
        }
        return LossOutput(loss=total_loss, metrics=metrics)

    def sample(
        self,
        model: VelocityModel,
        x0: torch.Tensor,
        conditioning: Conditioning | None = None,
        *,
        num_steps: int,
    ) -> torch.Tensor:
        """Integrate the learned velocity field from `x0` to `x1`.

        Args:
            model: Velocity network mapping `(x_t, t, h, conditioning)` to velocity.
            x0: Initial noise or source batch with shape `[N, *shape]`.
            conditioning: Optional conditioning carrying `class_labels`.
            num_steps: Euler integration step count.

        Returns:
            Sampled tensor with shape `[N, *shape]`.
        """
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        was_training = model.training
        model.eval()
        current = x0
        times = torch.linspace(0.0, 1.0, num_steps + 1, device=x0.device, dtype=x0.dtype)
        with torch.no_grad():
            for index in range(num_steps):
                t = torch.full((x0.shape[0], 1), float(times[index].item()), device=x0.device, dtype=x0.dtype)
                r = torch.full((x0.shape[0], 1), float(times[index + 1].item()), device=x0.device, dtype=x0.dtype)
                h = r - t
                velocity = model(current, t, h, conditioning=conditioning)
                current = current + h * velocity
        if was_training:
            model.train()
        return current
