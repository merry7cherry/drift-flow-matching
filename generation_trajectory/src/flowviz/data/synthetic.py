from __future__ import annotations

import math

import torch

from .base import PairDataset


def _sample_segments(
    segments: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    lengths = torch.linalg.norm(segments[:, 1] - segments[:, 0], dim=1)
    probs = lengths / lengths.sum()
    segment_idx = torch.multinomial(probs, batch_size, replacement=True, generator=generator)
    selected = segments[segment_idx]
    t = torch.rand(batch_size, 1, generator=generator)
    samples = selected[:, 0] + t * (selected[:, 1] - selected[:, 0])
    return samples.to(device)


def _sample_noise_like_samples(
    batch_size: int,
    dim: int,
    generator: torch.Generator,
    std: float,
    samples: torch.Tensor,
) -> torch.Tensor:
    """Sample Gaussian noise with CPU RNG then move to the sample tensor device."""

    noise = torch.randn(batch_size, dim, generator=generator, dtype=samples.dtype)
    return (noise * std).to(device=samples.device, dtype=samples.dtype)


class CircularUniformToLetterFDataset(PairDataset):
    """2D dataset mapping a uniform circular source to a target letter F."""

    def __init__(
        self,
        source_radius: float = 6.0,
        letter_width: float = 4.0,
        letter_height: float = 5.0,
        middle_bar_ratio: float = 0.6,
        target_std: float = 0.15,
        seed: int = 42,
    ) -> None:
        super().__init__(dim=2, seed=seed)
        self.source_radius = source_radius
        self.target_std = target_std

        half_width = letter_width / 2.0
        half_height = letter_height / 2.0
        middle_width = half_width * middle_bar_ratio
        self.target_segments = torch.tensor(
            [
                ((-half_width, -half_height), (-half_width, half_height)),
                ((-half_width, half_height), (half_width, half_height)),
                ((-half_width, 0.0), (middle_width, 0.0)),
            ],
            dtype=torch.float32,
        )

    def sample_base(self, batch_size: int, device: torch.device) -> torch.Tensor:
        radii = self.source_radius * torch.sqrt(
            torch.rand(batch_size, generator=self._generator)
        )
        angles = 2.0 * math.pi * torch.rand(batch_size, generator=self._generator)
        samples = torch.stack((radii * torch.cos(angles), radii * torch.sin(angles)), dim=1)
        return samples.to(device)

    def sample_target(self, batch_size: int, device: torch.device) -> torch.Tensor:
        samples = _sample_segments(self.target_segments, batch_size, self._generator, device)
        noise = _sample_noise_like_samples(
            batch_size,
            2,
            self._generator,
            self.target_std,
            samples,
        )
        return (samples + noise).to(device)


class CircularUniformToLetterMDataset(PairDataset):
    """2D dataset mapping a uniform circular source to a target letter M."""

    def __init__(
        self,
        source_radius: float = 6.0,
        letter_width: float = 5.0,
        letter_height: float = 5.0,
        target_std: float = 0.15,
        seed: int = 42,
    ) -> None:
        super().__init__(dim=2, seed=seed)
        self.source_radius = source_radius
        self.target_std = target_std

        half_width = letter_width / 2.0
        half_height = letter_height / 2.0
        self.target_segments = torch.tensor(
            [
                ((-half_width, -half_height), (-half_width, half_height)),
                ((-half_width, half_height), (0.0, -half_height)),
                ((0.0, -half_height), (half_width, half_height)),
                ((half_width, half_height), (half_width, -half_height)),
            ],
            dtype=torch.float32,
        )

    def sample_base(self, batch_size: int, device: torch.device) -> torch.Tensor:
        radii = self.source_radius * torch.sqrt(
            torch.rand(batch_size, generator=self._generator)
        )
        angles = 2.0 * math.pi * torch.rand(batch_size, generator=self._generator)
        samples = torch.stack((radii * torch.cos(angles), radii * torch.sin(angles)), dim=1)
        return samples.to(device)

    def sample_target(self, batch_size: int, device: torch.device) -> torch.Tensor:
        samples = _sample_segments(self.target_segments, batch_size, self._generator, device)
        noise = _sample_noise_like_samples(
            batch_size,
            2,
            self._generator,
            self.target_std,
            samples,
        )
        return (samples + noise).to(device)


class CircularUniformToMoonDataset(PairDataset):
    """2D dataset mapping a uniform circular source to a two-moon target."""

    def __init__(
        self,
        source_radius: float = 4.0,
        target_radius: float = 2.5,
        target_horizontal_gap: float | None = None,
        target_vertical_gap: float = 1.5,
        target_std: float = 0.15,
        seed: int = 42,
    ) -> None:
        super().__init__(dim=2, seed=seed)
        self.source_radius = source_radius
        self.target_radius = target_radius
        horizontal_gap = target_radius if target_horizontal_gap is None else target_horizontal_gap
        self.target_horizontal_gap = horizontal_gap
        self.target_vertical_gap = target_vertical_gap
        self.target_std = target_std
        self.upper_center = torch.tensor(
            (-horizontal_gap / 2.0, target_vertical_gap / 2.0), dtype=torch.float32
        )
        self.lower_center = torch.tensor(
            (horizontal_gap / 2.0, -target_vertical_gap / 2.0), dtype=torch.float32
        )

    def sample_base(self, batch_size: int, device: torch.device) -> torch.Tensor:
        radii = self.source_radius * torch.sqrt(
            torch.rand(batch_size, generator=self._generator)
        )
        angles = 2.0 * math.pi * torch.rand(batch_size, generator=self._generator)
        samples = torch.stack((radii * torch.cos(angles), radii * torch.sin(angles)), dim=1)
        return samples.to(device)

    def sample_target(self, batch_size: int, device: torch.device) -> torch.Tensor:
        component_idx = torch.randint(
            low=0,
            high=2,
            size=(batch_size,),
            generator=self._generator,
        )
        theta = torch.rand(batch_size, generator=self._generator) * math.pi

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        arc = torch.stack((self.target_radius * cos_theta, self.target_radius * sin_theta), dim=1)
        upper_center = self.upper_center.to(device=arc.device, dtype=arc.dtype)
        lower_center = self.lower_center.to(device=arc.device, dtype=arc.dtype)
        upper_moon = arc + upper_center
        lower_moon = torch.stack(
            (self.target_radius * cos_theta, -self.target_radius * sin_theta),
            dim=1,
        ) + lower_center

        samples = torch.where(component_idx.unsqueeze(1) == 0, upper_moon, lower_moon)
        noise = _sample_noise_like_samples(
            batch_size,
            2,
            self._generator,
            self.target_std,
            samples,
        )
        samples = samples + noise
        return samples.to(device)


class CircularUniformToCheckerboardGridDataset(PairDataset):
    """2D dataset mapping a uniform circular source to a checkerboard 4x4 grid target."""

    def __init__(
        self,
        source_radius: float = 3.0,
        grid_size: int = 4,
        grid_spacing: float = 2.0,
        target_std: float = 0.2,
        seed: int = 42,
    ) -> None:
        super().__init__(dim=2, seed=seed)
        self.source_radius = source_radius
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing
        self.target_std = target_std

        offset = -((grid_size - 1) / 2.0) * grid_spacing
        centers = []
        for row in range(grid_size):
            for col in range(grid_size):
                if (row + col) % 2 == 1:
                    centers.append(
                        (
                            offset + col * grid_spacing,
                            offset + row * grid_spacing,
                        )
                    )
        self.target_centers = torch.tensor(centers, dtype=torch.float32)

    def sample_base(self, batch_size: int, device: torch.device) -> torch.Tensor:
        radii = self.source_radius * torch.sqrt(
            torch.rand(batch_size, generator=self._generator)
        )
        angles = 2.0 * math.pi * torch.rand(batch_size, generator=self._generator)
        samples = torch.stack((radii * torch.cos(angles), radii * torch.sin(angles)), dim=1)
        return samples.to(device)

    def sample_target(self, batch_size: int, device: torch.device) -> torch.Tensor:
        component_idx = torch.randint(
            low=0,
            high=self.target_centers.shape[0],
            size=(batch_size,),
            generator=self._generator,
        )
        centers = self.target_centers[component_idx]
        noise = _sample_noise_like_samples(
            batch_size,
            2,
            self._generator,
            self.target_std,
            centers,
        )
        samples = centers + noise
        return samples.to(device)


__all__ = ['CircularUniformToCheckerboardGridDataset', 'CircularUniformToLetterFDataset', 'CircularUniformToLetterMDataset', 'CircularUniformToMoonDataset']
