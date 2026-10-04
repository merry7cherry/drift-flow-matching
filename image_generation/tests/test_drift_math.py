from __future__ import annotations


import pytest


import torch


import torch.nn as nn


architectures = pytest.importorskip("driftfm.architectures")


config = pytest.importorskip("driftfm.config")


data = pytest.importorskip("driftfm.data")


dfm = pytest.importorskip("driftfm.methods.drift_flow_matching")


class ZeroVelocityModel(architectures.VelocityModel):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        conditioning: object | None = None,
    ) -> torch.Tensor:
        del t, h, conditioning
        return torch.zeros_like(x_t) + self.bias


class LabelOnlyVelocityModel(architectures.VelocityModel):
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        conditioning: object | None = None,
    ) -> torch.Tensor:
        del t, h
        assert conditioning is not None
        labels = conditioning.class_labels.to(device=x_t.device, dtype=x_t.dtype).view(-1, 1)
        return labels.expand_as(x_t)


def _make_transport_batch(
    torch_device: torch.device,
    labels: list[int],
    *,
    feature_dim: int = 4,
) -> data.TransportBatch:
    batch_size = len(labels)
    x0 = torch.randn(batch_size, feature_dim, device=torch_device)
    x1 = torch.randn(batch_size, feature_dim, device=torch_device)
    class_labels = torch.tensor(labels, dtype=torch.long, device=torch_device)
    return data.TransportBatch(
        x0=x0,
        x1=x1,
        conditioning=data.Conditioning(class_labels=class_labels),
    )


def test_sample_drift_timestep_returns_ordered_times(torch_device):
    cfg = config.DriftFlowMatchingConfig()
    t, r = dfm.sample_drift_timestep(cfg, num_samples=128, device=torch_device)

    assert t.shape == r.shape == (128,)
    assert torch.all(t >= 0)
    assert torch.all(r <= 1)
    assert torch.all(t <= r)


def test_grouped_sinkhorn_is_finite_and_shape_stable(torch_device):
    logits = torch.randn(3, 4, 5, device=torch_device)
    r = torch.full((4,), 1 / 4, device=torch_device)
    c = torch.full((5,), 1 / 5, device=torch_device)

    plan = dfm._sinkhorn_from_logits(logits, row_marginals=r, col_marginals=c, num_iters=8)
    assert plan.shape == logits.shape
    assert torch.isfinite(plan).all()
    assert torch.all(plan >= 0)


def test_sinkhorn_supports_group_specific_column_marginals(torch_device):
    logits = torch.randn(2, 4, 5, device=torch_device)
    r = torch.full((4,), 1 / 4, device=torch_device)
    c = torch.tensor(
        [
            [0.1, 0.1, 0.2, 0.3, 0.3],
            [0.3, 0.3, 0.2, 0.1, 0.1],
        ],
        device=torch_device,
    )

    plan = dfm._sinkhorn_from_logits(logits, row_marginals=r, col_marginals=c, num_iters=32)
    assert plan.shape == logits.shape
    assert torch.isfinite(plan).all()
    assert torch.allclose(plan.sum(dim=-2), c, atol=5e-3, rtol=5e-3)


def test_group_batch_into_class_subgroups_returns_explicit_axes(torch_device):
    batch = _make_transport_batch(torch_device, [0, 0, 0, 0, 1, 1, 1, 1])
    x0_group, x1_group, class_labels = dfm._group_batch_into_class_subgroups(batch, subgroups_per_class=2)

    assert x0_group.shape == (2, 2, 2, 4)
    assert x1_group.shape == (2, 2, 2, 4)
    assert class_labels.shape == (2, 2, 2)
    assert torch.all(class_labels[0] == 0)
    assert torch.all(class_labels[1] == 1)


def test_sort_by_class_label_skips_gpu_gather_for_pre_sorted_batches(torch_device):
    batch = _make_transport_batch(torch_device, [0, 0, 1, 1, 2, 2])
    sorted_batch = dfm._sort_by_class_label(batch)

    assert sorted_batch.x0.data_ptr() == batch.x0.data_ptr()
    assert sorted_batch.x1.data_ptr() == batch.x1.data_ptr()
    assert sorted_batch.conditioning is not None
    assert batch.conditioning is not None
    assert sorted_batch.conditioning.class_labels is not None
    assert batch.conditioning.class_labels is not None
    assert sorted_batch.conditioning.class_labels.data_ptr() == batch.conditioning.class_labels.data_ptr()


def test_flatten_class_subgroups_to_groups_merges_c_and_s_axes(torch_device):
    class_subgroup_tensor = torch.arange(2 * 3 * 4 * 5, device=torch_device, dtype=torch.float32).view(2, 3, 4, 5)
    grouped_tensor = dfm._flatten_class_subgroups_to_groups(class_subgroup_tensor)

    assert grouped_tensor.shape == (6, 4, 5)
    assert torch.equal(grouped_tensor[0], class_subgroup_tensor[0, 0])
    assert torch.equal(grouped_tensor[-1], class_subgroup_tensor[1, 2])


def test_shared_subgroup_timesteps_are_broadcast_across_classes(torch_device):
    cfg = config.DriftFlowMatchingConfig()
    t_group, r_group = dfm._sample_shared_subgroup_timesteps(
        cfg,
        num_classes=3,
        subgroups_per_class=2,
        subgroup_size=4,
        device=torch_device,
        dtype=torch.float32,
    )

    assert t_group.shape == r_group.shape == (3, 2, 4, 1)
    assert torch.all(t_group <= r_group)
    assert torch.allclose(t_group[0], t_group[1])
    assert torch.allclose(t_group[1], t_group[2])


def test_sample_requires_explicit_positive_num_steps(torch_device):
    method = dfm.DriftFlowMatchingMethod(config.DriftFlowMatchingConfig(groups_per_class=1, sinkhorn_iters=2))
    model = ZeroVelocityModel().to(torch_device)
    noise = torch.randn(4, 4, device=torch_device)

    with pytest.raises(TypeError):
        method.sample(model, noise)

    with pytest.raises(ValueError, match="num_steps must be positive"):
        method.sample(model, noise, num_steps=0)


def test_method_rejects_legacy_split_drift_form() -> None:
    with pytest.raises(ValueError, match="Only drift_form in"):
        dfm.DriftFlowMatchingMethod(config.DriftFlowMatchingConfig(drift_form="split"))



def test_one_point_group_drift_is_target_minus_source():
    gen = torch.tensor([[[1.0, 2.0]], [[-2.0, 1.0]]])
    pos = torch.tensor([[[3.0, 1.0]], [[0.0, 4.0]]])
    assert torch.allclose(dfm.compute_drift_grouped_sinkhorn(gen, pos), pos - gen)


def test_mainline_loss_is_finite_and_differentiable(torch_device):
    model = architectures.ConditionalTimeMLP(data_dim=4, hidden_sizes=[16, 16], num_classes=2, class_embedding_dim=4)
    method = dfm.DriftFlowMatchingMethod(config.DriftFlowMatchingConfig(groups_per_class=2, sinkhorn_iters=3))
    output = method.compute_loss(model, _make_transport_batch(torch_device, [0]*8+[1]*8))
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_non_mainline_method_rejected():
    with pytest.raises(ValueError, match='split_v0'):
        dfm.DriftFlowMatchingMethod(config.DriftFlowMatchingConfig(drift_form='split_v1'))
