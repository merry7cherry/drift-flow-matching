from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def rng_seed() -> int:
    return 1234


@pytest.fixture()
def torch_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture()
def synthetic_mnist_latents(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "mnist_latents"
    root.mkdir()
    train_latents = []
    train_labels = []
    test_latents = []
    test_labels = []
    for cls in range(10):
        base = np.full((8, 6), float(cls), dtype=np.float32)
        train_latents.append(base + np.random.default_rng(cls).normal(scale=0.01, size=base.shape))
        train_labels.append(np.full((8,), cls, dtype=np.int64))
        test_latents.append(base[:4] + np.random.default_rng(cls + 100).normal(scale=0.01, size=(4, 6)))
        test_labels.append(np.full((4,), cls, dtype=np.int64))

    np.save(root / "train_latents.npy", np.concatenate(train_latents, axis=0))
    np.save(root / "train_labels.npy", np.concatenate(train_labels, axis=0))
    np.save(root / "test_latents.npy", np.concatenate(test_latents, axis=0))
    np.save(root / "test_labels.npy", np.concatenate(test_labels, axis=0))
    return {
        "root": root,
        "train_latents": root / "train_latents.npy",
        "train_labels": root / "train_labels.npy",
        "test_latents": root / "test_latents.npy",
        "test_labels": root / "test_labels.npy",
    }


@pytest.fixture()
def synthetic_ffhq_npz(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "ffhq_latents"
    root.mkdir()
    keys = [
        "male_children",
        "male_adult",
        "male_old",
        "female_children",
        "female_adult",
        "female_old",
    ]
    payload: dict[str, np.ndarray] = {}
    for idx, key in enumerate(keys):
        arr = np.random.default_rng(idx).normal(loc=float(idx), scale=0.01, size=(12, 512)).astype(np.float32)
        payload[key] = arr

    train_npz = root / "train_latents_by_class.npz"
    test_npz = root / "test_latents_by_class.npz"
    np.savez(train_npz, **payload)
    np.savez(test_npz, **payload)
    return {"root": root, "train_npz": train_npz, "test_npz": test_npz}
