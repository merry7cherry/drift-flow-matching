from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Conditioning:
    class_labels: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "Conditioning":
        return Conditioning(
            class_labels=None if self.class_labels is None else self.class_labels.to(device),
        )


@dataclass(slots=True)
class TransportBatch:
    x0: torch.Tensor
    x1: torch.Tensor
    conditioning: Conditioning | None = None

    def to(self, device: torch.device | str) -> "TransportBatch":
        return TransportBatch(
            x0=self.x0.to(device),
            x1=self.x1.to(device),
            conditioning=None if self.conditioning is None else self.conditioning.to(device),
        )


class TransportDataset(ABC):
    def __init__(self, *, data_shape: tuple[int, ...], seed: int = 42) -> None:
        self._data_shape = tuple(data_shape)
        self.seed = seed
        self._generator = torch.Generator(device="cpu").manual_seed(seed)

    @property
    def data_shape(self) -> tuple[int, ...]:
        return self._data_shape

    @property
    def data_dim(self) -> int:
        dim = 1
        for value in self.data_shape:
            dim *= value
        return dim

    @property
    def num_classes(self) -> int | None:
        return None

    @property
    def class_names(self) -> list[str] | None:
        return None

    def make_noise(
        self,
        batch_size: int,
        device: torch.device,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        active_generator = self._generator if generator is None else generator
        sample_device = torch.device(getattr(active_generator, "device", "cpu"))
        noise = torch.randn(
            (batch_size, *self.data_shape),
            device=sample_device,
            generator=active_generator,
        )
        return noise.to(device)

    @abstractmethod
    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
    ) -> TransportBatch:
        raise NotImplementedError

    def sample_generation_conditioning(
        self,
        samples_per_class: int,
        device: torch.device,
    ) -> Conditioning | None:
        return None


class ClassBalancedLatentDataset(TransportDataset):
    def __init__(
        self,
        *,
        latents_by_class: list[torch.Tensor],
        class_names: list[str],
        seed: int = 42,
    ) -> None:
        if not latents_by_class:
            raise ValueError("latents_by_class must be non-empty")
        feature_shapes = {tuple(latent.shape[1:]) for latent in latents_by_class}
        if len(feature_shapes) != 1:
            raise ValueError("All latent tensors must share the same feature shape")
        super().__init__(data_shape=feature_shapes.pop(), seed=seed)
        self._latents_by_class = [latent.detach().cpu().float() for latent in latents_by_class]
        self._class_names = list(class_names)

    @property
    def num_classes(self) -> int:
        return len(self._latents_by_class)

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    def _sample_from_class(
        self,
        class_index: int,
        count: int,
    ) -> torch.Tensor:
        pool = self._latents_by_class[class_index]
        indices = torch.randint(
            low=0,
            high=pool.shape[0],
            size=(count,),
            generator=self._generator,
        )
        return pool[indices]

    def sample_real_per_class(self, count: int) -> dict[int, torch.Tensor]:
        return {class_index: self._sample_from_class(class_index, count) for class_index in range(self.num_classes)}

    def sample_generation_conditioning(
        self,
        samples_per_class: int,
        device: torch.device,
    ) -> Conditioning:
        labels = torch.repeat_interleave(
            torch.arange(self.num_classes, dtype=torch.long),
            repeats=samples_per_class,
        )
        return Conditioning(class_labels=labels.to(device))

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
    ) -> TransportBatch:
        if batch_size % self.num_classes != 0:
            raise ValueError(
                "batch_size must be divisible by num_classes for balanced class sampling, "
                f"but got batch_size={batch_size}, num_classes={self.num_classes}"
            )
        samples_per_class = batch_size // self.num_classes

        x1_parts: list[torch.Tensor] = []
        class_parts: list[torch.Tensor] = []

        for class_index in range(self.num_classes):
            x1_parts.append(self._sample_from_class(class_index, samples_per_class))
            class_parts.append(torch.full((samples_per_class,), class_index, dtype=torch.long))

        x1 = torch.cat(x1_parts, dim=0).to(device)
        class_labels = torch.cat(class_parts, dim=0).to(device)
        x0 = self.make_noise(x1.shape[0], device)
        return TransportBatch(
            x0=x0,
            x1=x1,
            conditioning=Conditioning(class_labels=class_labels),
        )
