from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .base import ClassBalancedLatentDataset

FFHQ_CLASS_NAMES = [
    "male_children",
    "male_adult",
    "male_old",
    "female_children",
    "female_adult",
    "female_old",
]


class FFHQLatentDataset(ClassBalancedLatentDataset):
    def __init__(
        self,
        *,
        npz_path: str | None = None,
        train_npz_path: str | None = None,
        test_npz_path: str | None = None,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        npz_file = Path(_resolve_npz_path(npz_path=npz_path, train_npz_path=train_npz_path, test_npz_path=test_npz_path, split=split))
        arrays = np.load(npz_file, allow_pickle=False)
        missing = [key for key in FFHQ_CLASS_NAMES if key not in arrays]
        if missing:
            raise ValueError(f"Missing FFHQ class keys in {npz_file}: {', '.join(missing)}")
        latents_by_class = [torch.from_numpy(arrays[key]).float() for key in FFHQ_CLASS_NAMES]
        super().__init__(
            latents_by_class=latents_by_class,
            class_names=FFHQ_CLASS_NAMES,
            seed=seed,
        )
        self.npz_path = str(npz_file)
        self.split = split


def _resolve_npz_path(
    *,
    npz_path: str | None,
    train_npz_path: str | None,
    test_npz_path: str | None,
    split: str,
) -> str:
    if npz_path is not None:
        return npz_path
    if split == "train" and train_npz_path is not None:
        return train_npz_path
    if split == "test" and test_npz_path is not None:
        return test_npz_path
    raise ValueError("Either npz_path or split-specific train/test NPZ paths must be provided")
