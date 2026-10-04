from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..config import DatasetSection
from .base import TransportDataset
from .ffhq_latent import FFHQLatentDataset
from .mnist_latent import MnistLatentDataset

DatasetFactory = Callable[..., TransportDataset]


def _build_mnist_latent(**params: Any) -> TransportDataset:
    return MnistLatentDataset(**params)


def _build_ffhq_latent(**params: Any) -> TransportDataset:
    return FFHQLatentDataset(**params)


DATASETS: dict[str, DatasetFactory] = {
    "mnist_latent": _build_mnist_latent,
    "ffhq_latent": _build_ffhq_latent,
}


def create_dataset(section: DatasetSection) -> TransportDataset:
    try:
        factory = DATASETS[section.name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset: {section.name}") from exc
    return factory(**dict(section.params))
