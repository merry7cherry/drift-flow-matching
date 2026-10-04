from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from ..config import ArchitectureSection
from ..data import TransportDataset
from .conditional_time_mlp import ConditionalTimeMLP
from .mnist_ae import MnistConvAE

ArchitectureFactory = Callable[[dict[str, Any], TransportDataset | None], nn.Module]


def _resolve_context(
    params: dict[str, Any],
    dataset: TransportDataset | None,
) -> tuple[int, int | None]:
    data_dim = int(params.get("data_dim", dataset.data_dim if dataset is not None else 0))
    if data_dim <= 0:
        raise ValueError("data_dim must be provided directly or via dataset context")
    num_classes = params.get("num_classes", dataset.num_classes if dataset is not None else None)
    return data_dim, num_classes


def _build_conditional_time_mlp(
    params: dict[str, Any],
    dataset: TransportDataset | None,
    *,
    default_hidden_sizes: list[int],
    default_class_embedding_dim: int,
) -> nn.Module:
    data_dim, num_classes = _resolve_context(params, dataset)
    hidden_sizes = list(params.get("hidden_sizes", default_hidden_sizes))
    return ConditionalTimeMLP(
        data_dim=data_dim,
        hidden_sizes=hidden_sizes,
        num_classes=None if num_classes is None else int(num_classes),
        conditioning_mode=params.get("conditioning_mode"),
        class_embedding_dim=int(params.get("class_embedding_dim", default_class_embedding_dim)),
        activation=str(params.get("activation", "silu")),
    )


def _build_mnist_conv_ae(
    params: dict[str, Any],
    dataset: TransportDataset | None,
) -> nn.Module:
    del dataset
    return MnistConvAE(latent_dim=int(params.get("latent_dim", 6)))


def _build_mnist_latent_mlp(
    params: dict[str, Any],
    dataset: TransportDataset | None,
) -> nn.Module:
    return _build_conditional_time_mlp(
        params,
        dataset,
        default_hidden_sizes=[256, 256, 256],
        default_class_embedding_dim=32,
    )


def _build_ffhq_latent_mlp(
    params: dict[str, Any],
    dataset: TransportDataset | None,
) -> nn.Module:
    return _build_conditional_time_mlp(
        params,
        dataset,
        default_hidden_sizes=[1024, 1024, 1024],
        default_class_embedding_dim=64,
    )


ARCHITECTURES: dict[str, ArchitectureFactory] = {
    "mnist_conv_ae": _build_mnist_conv_ae,
    "mnist_latent_mlp": _build_mnist_latent_mlp,
    "ffhq_latent_mlp": _build_ffhq_latent_mlp,
}


def create_architecture(
    section: ArchitectureSection,
    dataset: TransportDataset | None = None,
) -> nn.Module:
    try:
        factory = ARCHITECTURES[section.name]
    except KeyError as exc:
        raise KeyError(f"Unknown architecture: {section.name}") from exc
    return factory(dict(section.params), dataset)
