from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

utils = pytest.importorskip("driftfm.utils")


class _RandomizedDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(
            [
                float(index),
                random.random(),
                float(np.random.random()),
                float(torch.rand(()).item()),
            ],
            dtype=torch.float64,
        )


def _collect_loader_rows(seed: int) -> torch.Tensor:
    loader = DataLoader(
        _RandomizedDataset(),
        batch_size=2,
        shuffle=True,
        num_workers=2,
        generator=utils.make_data_loader_generator(seed=seed),
        worker_init_fn=utils.seed_data_loader_worker,
    )
    return torch.cat(list(loader), dim=0)


def test_data_loader_helpers_make_multi_worker_randomness_repeatable() -> None:
    first = _collect_loader_rows(17)
    second = _collect_loader_rows(17)
    third = _collect_loader_rows(23)

    assert torch.equal(first, second)
    assert not torch.equal(first, third)
