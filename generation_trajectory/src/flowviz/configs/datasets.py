from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Type
from ..data.base import PairDataset
from ..data.synthetic import (
    CircularUniformToMoonDataset, CircularUniformToCheckerboardGridDataset,
    CircularUniformToLetterFDataset, CircularUniformToLetterMDataset,
)

@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for constructing synthetic pair datasets."""

    name: str
    label: str
    dataset_cls: Type[PairDataset]
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def create_dataset(self, seed: int) -> PairDataset:
        params: Dict[str, Any] = dict(self.kwargs)
        params.setdefault("seed", seed)
        return self.dataset_cls(**params)


CIRCULAR_UNIFORM_TO_MOON = DatasetConfig(
    name="2d_circular_uniform_to_moon",
    label="2D Circular Uniform to Moon",
    dataset_cls=CircularUniformToMoonDataset,
    kwargs={
        "source_radius": 6.0,
        "target_radius": 2.5,
        "target_horizontal_gap": 2.5,
        "target_vertical_gap": 1.5,
        "target_std": 0.15,
    },
)


CIRCULAR_UNIFORM_TO_CHECKERBOARD_GRID = DatasetConfig(
    name="2d_circular_uniform_to_checkerboard_grid",
    label="2D Circular Uniform to Checkerboard Grid",
    dataset_cls=CircularUniformToCheckerboardGridDataset,
    kwargs={
        "source_radius": 6.0,
        "grid_size": 4,
        "grid_spacing": 2.0,
        "target_std": 0.2,
    },
)


LETTER_F_DATASET = DatasetConfig(
    name="2d_circular_uniform_to_letter_f",
    label="2D Circular Uniform to Letter F",
    dataset_cls=CircularUniformToLetterFDataset,
    kwargs={
        "source_radius": 6.0,
        "letter_width": 5.0,
        "letter_height": 7.0,
        "middle_bar_ratio": 0.6,
        "target_std": 0.15,
    },
)


LETTER_M_DATASET = DatasetConfig(
    name="2d_circular_uniform_to_letter_m",
    label="2D Circular Uniform to Letter M",
    dataset_cls=CircularUniformToLetterMDataset,
    kwargs={
        "source_radius": 6.0,
        "letter_width": 7.0,
        "letter_height": 7.0,
        "target_std": 0.15,
    },
)


DATASET_CONFIGS = {cfg.name: cfg for cfg in (
    CIRCULAR_UNIFORM_TO_MOON, CIRCULAR_UNIFORM_TO_CHECKERBOARD_GRID,
    LETTER_F_DATASET, LETTER_M_DATASET,
)}
DEFAULT_VISUALIZATION_DATASET_KEYS = (CIRCULAR_UNIFORM_TO_MOON.name,)
