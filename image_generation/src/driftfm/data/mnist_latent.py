from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..utils import resolve_latest_alias
from .base import ClassBalancedLatentDataset

MNIST_CLASS_NAMES = [str(index) for index in range(10)]


class MnistLatentDataset(ClassBalancedLatentDataset):
    def __init__(
        self,
        *,
        root_dir: str | None = None,
        split: str = "train",
        train_latents_path: str | None = None,
        train_labels_path: str | None = None,
        test_latents_path: str | None = None,
        test_labels_path: str | None = None,
        latents_path: str | None = None,
        labels_path: str | None = None,
        seed: int = 42,
    ) -> None:
        latents_file, labels_file = _resolve_split_paths(
            root_dir=root_dir,
            split=split,
            train_latents_path=train_latents_path,
            train_labels_path=train_labels_path,
            test_latents_path=test_latents_path,
            test_labels_path=test_labels_path,
            latents_path=latents_path,
            labels_path=labels_path,
        )
        latents = np.load(latents_file)
        labels = np.load(labels_file)
        latents_by_class = []
        for class_index in range(10):
            class_indices = np.where(labels == class_index)[0]
            latents_by_class.append(torch.from_numpy(latents[class_indices]).float())
        super().__init__(
            latents_by_class=latents_by_class,
            class_names=MNIST_CLASS_NAMES,
            seed=seed,
        )
        self.split = split
        self.latents_path = str(latents_file)
        self.labels_path = str(labels_file)


def _resolve_split_paths(
    *,
    root_dir: str | None,
    split: str,
    train_latents_path: str | None,
    train_labels_path: str | None,
    test_latents_path: str | None,
    test_labels_path: str | None,
    latents_path: str | None,
    labels_path: str | None,
) -> tuple[Path, Path]:
    if latents_path is not None and labels_path is not None:
        return Path(latents_path), Path(labels_path)
    if split == "train" and train_latents_path is not None and train_labels_path is not None:
        return Path(train_latents_path), Path(train_labels_path)
    if split == "test" and test_latents_path is not None and test_labels_path is not None:
        return Path(test_latents_path), Path(test_labels_path)
    if root_dir is None:
        raise ValueError("Either root_dir or explicit latents_path/labels_path must be provided")
    base_dir = resolve_latest_alias(root_dir)
    return base_dir / f"{split}_latents.npy", base_dir / f"{split}_labels.npy"
