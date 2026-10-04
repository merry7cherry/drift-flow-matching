from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

datasets = pytest.importorskip("driftfm.data")


def _maybe_get(obj, name: str):
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _sample_batch(dataset, batch_size: int, device: torch.device):
    for method_name in ("sample_train_batch", "sample_batch", "sample"):
        method = getattr(dataset, method_name, None)
        if callable(method):
            try:
                return method(batch_size=batch_size, device=device)
            except TypeError:
                try:
                    return method(batch_size, device)
                except TypeError:
                    continue
    raise AssertionError("dataset does not expose a recognized sampling method")


def _extract_labels(batch):
    conditioning = _maybe_get(batch, "conditioning")
    if conditioning is not None:
        labels = _maybe_get(conditioning, "class_labels")
        if labels is not None:
            return labels
    labels = _maybe_get(batch, "class_labels")
    if labels is not None:
        return labels
    return None


def test_mnist_latent_loader_emits_balanced_labels(synthetic_mnist_latents, torch_device):
    factory = datasets.DATASETS.get("mnist_latent")
    if factory is None:
        pytest.skip("mnist_latent dataset is not registered yet")

    dataset = factory(
        train_latents_path=synthetic_mnist_latents["train_latents"],
        train_labels_path=synthetic_mnist_latents["train_labels"],
        test_latents_path=synthetic_mnist_latents["test_latents"],
        test_labels_path=synthetic_mnist_latents["test_labels"],
    )
    batch = _sample_batch(dataset, batch_size=10, device=torch_device)
    x0 = _maybe_get(batch, "x0")
    x1 = _maybe_get(batch, "x1")
    labels = _extract_labels(batch)

    assert x0 is not None and x1 is not None
    assert x0.shape == x1.shape
    assert x0.shape[0] == 10
    assert labels is not None
    assert set(torch.as_tensor(labels).tolist()) == set(range(10))
    assert _maybe_get(batch, "group_ids") is None


def test_ffhq_latent_loader_emits_balanced_labels(synthetic_ffhq_npz, torch_device):
    factory = datasets.DATASETS.get("ffhq_latent")
    if factory is None:
        pytest.skip("ffhq_latent dataset is not registered yet")

    dataset = factory(
        train_npz_path=synthetic_ffhq_npz["train_npz"],
        test_npz_path=synthetic_ffhq_npz["test_npz"],
    )
    batch = _sample_batch(dataset, batch_size=6, device=torch_device)
    labels = _extract_labels(batch)
    x0 = _maybe_get(batch, "x0")
    x1 = _maybe_get(batch, "x1")

    assert x0 is not None and x1 is not None
    assert x0.shape == x1.shape
    assert x0.shape[0] == 6
    assert labels is not None
    assert set(torch.as_tensor(labels).tolist()) == set(range(6))
    assert _maybe_get(batch, "group_ids") is None


def test_dataset_labels_remain_class_balanced(synthetic_ffhq_npz, torch_device):
    factory = datasets.DATASETS.get("ffhq_latent")
    if factory is None:
        pytest.skip("ffhq_latent dataset is not registered yet")

    dataset = factory(
        train_npz_path=synthetic_ffhq_npz["train_npz"],
        test_npz_path=synthetic_ffhq_npz["test_npz"],
    )
    batch = _sample_batch(dataset, batch_size=12, device=torch_device)
    labels = _extract_labels(batch)
    assert labels is not None
    counts = defaultdict(int)
    for label in torch.as_tensor(labels).tolist():
        counts[int(label)] += 1
    assert counts
    assert set(counts) == set(range(6))
    assert all(count == 2 for count in counts.values())


def test_make_noise_samples_on_generator_device_before_transferring(monkeypatch):
    class _Dataset(datasets.TransportDataset):
        def __init__(self) -> None:
            super().__init__(data_shape=(2,), seed=7)

        def sample_batch(self, batch_size: int, device: torch.device):
            raise NotImplementedError

    class _FakeTensor:
        def __init__(self) -> None:
            self.to_calls: list[torch.device] = []

        def to(self, device: torch.device | str):
            normalized = torch.device(device)
            self.to_calls.append(normalized)
            return {"moved_to": str(normalized)}

    captured: dict[str, object] = {}
    fake_tensor = _FakeTensor()

    def _fake_randn(shape, *, device=None, generator=None):
        captured["shape"] = shape
        captured["device"] = device
        captured["generator"] = generator
        return fake_tensor

    monkeypatch.setattr(torch, "randn", _fake_randn)

    dataset = _Dataset()
    generator = SimpleNamespace(device=torch.device("cuda"))

    result = dataset.make_noise(3, torch.device("cpu"), generator=generator)

    assert captured["shape"] == (3, 2)
    assert captured["device"] == torch.device("cuda")
    assert captured["generator"] is generator
    assert fake_tensor.to_calls == [torch.device("cpu")]
    assert result == {"moved_to": "cpu"}
