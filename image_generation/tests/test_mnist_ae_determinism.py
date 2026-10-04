from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

task = pytest.importorskip("driftfm.tasks.mnist_ae")
checkpoints = pytest.importorskip("driftfm.training.checkpoints")
utils = pytest.importorskip("driftfm.utils")


class _ToyMNIST(torch.utils.data.Dataset):
    def __init__(self, *, train: bool, transform) -> None:
        self.transform = transform
        count = 8 if train else 4
        self.images = [np.full((28, 28), 255 * (index % 2), dtype=np.uint8) for index in range(count)]
        self.labels = [index % 2 for index in range(count)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.fromarray(self.images[index], mode="L")
        return self.transform(image), self.labels[index]


def _state_dict_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def _assert_determinism_metadata(payload: dict[str, object], *, seed: int, device: str) -> None:
    assert payload["resolved_seed"] == seed
    determinism = payload["determinism"]
    assert isinstance(determinism, dict)
    assert determinism["seed"] == seed
    assert determinism["deterministic"] is True
    assert determinism["device"] == device
    assert determinism["torch_deterministic_algorithms"] is True
    assert determinism["cudnn_benchmark"] is False
    assert determinism["cudnn_deterministic"] is True
    assert determinism["cuda_available"] is False


def _run_mnist_ae(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_key: str,
    timestamp: str,
) -> tuple[dict[str, torch.Tensor], bytes, dict[str, object], Path]:
    mnist_root = tmp_path / run_key / "mnist"
    runs_root = tmp_path / run_key / "runs"

    def _fake_runtime_roots() -> dict[str, str]:
        return {"mnist_root": str(mnist_root), "runs_root": str(runs_root)}

    def _fake_mnist(*, root: str | Path, train: bool, download: bool, transform):
        assert Path(root) == mnist_root
        assert download is False
        return _ToyMNIST(train=train, transform=transform)

    def _fake_datetime():
        class _FakeNow:
            def strftime(self, fmt: str) -> str:
                assert fmt == "%Y%m%d_%H%M%S"
                return timestamp

        return _FakeNow()

    class _FakeDateTime:
        @staticmethod
        def now():
            return _fake_datetime()

    monkeypatch.setattr(task, "default_runtime_roots", _fake_runtime_roots)
    monkeypatch.setattr(task.datasets, "MNIST", _fake_mnist)
    monkeypatch.setattr(task, "datetime", _FakeDateTime)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "driftfm-mnist-train-ae",
            "--latent-dim",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--lr",
            "0.001",
            "--device",
            "cpu",
            "--seed",
            "17",
            "--save-every",
            "1",
        ],
    )

    task.main()

    run_root = Path(runs_root) / "mnist_ae"
    run_dir = utils.resolve_latest_alias(run_root / "latest")
    checkpoint = checkpoints.load_checkpoint(run_dir / "ae_final.pt", map_location="cpu")
    recon_bytes = (run_dir / "recon_epoch1.png").read_bytes()
    metadata = utils.load_json_file(run_dir / "run_metadata.json")
    assert metadata is not None
    return checkpoint["model_state"], recon_bytes, metadata, run_dir


def test_mnist_ae_is_deterministic_and_uses_stable_run_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_state, first_recon, first_metadata, first_run_dir = _run_mnist_ae(
        tmp_path,
        monkeypatch,
        run_key="first",
        timestamp="20260417_010101",
    )
    second_state, second_recon, second_metadata, second_run_dir = _run_mnist_ae(
        tmp_path,
        monkeypatch,
        run_key="second",
        timestamp="20260417_020202",
    )

    assert first_run_dir.name == second_run_dir.name
    assert first_run_dir.name.startswith("seed17_")
    assert _state_dict_equal(first_state, second_state)
    assert first_recon == second_recon
    assert first_metadata == second_metadata
    _assert_determinism_metadata(first_metadata, seed=17, device="cpu")
