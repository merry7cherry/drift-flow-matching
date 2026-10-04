from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


task = pytest.importorskip("driftfm.tasks.mnist_encode")
utils = pytest.importorskip("driftfm.utils")


class _DummyAE:
    def __init__(self, *, latent_dim: int) -> None:
        self.latent_dim = latent_dim
        self.loaded_state: dict[str, object] | None = None

    def to(self, device: torch.device) -> "_DummyAE":
        return self

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.loaded_state = state

    def eval(self) -> "_DummyAE":
        return self

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros((images.shape[0], self.latent_dim), dtype=torch.float32)


def test_mnist_encode_resolves_latest_alias_for_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "runs" / "mnist_ae"
    run_dir = run_root / "20260403_000000_latent6"
    run_dir.mkdir(parents=True)
    ae_ckpt = run_dir / "ae_final.pt"
    ae_ckpt.write_bytes(b"checkpoint placeholder")
    utils.write_latest_run_marker(run_root, run_dir)

    mnist_root = tmp_path / "mnist"

    loaded: dict[str, Path] = {}

    def fake_torch_load(path: str | Path, *args: object, **kwargs: object) -> dict[str, object]:
        loaded["path"] = Path(path)
        return {"latent_dim": 6, "model_state": {}}

    def fake_encode_split(
        model: _DummyAE,
        loader: object,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
        del model, loader, device
        latents = np.zeros((4, 6), dtype=np.float32)
        labels = np.array([0, 1, 2, 3], dtype=np.int64)
        return latents, labels

    monkeypatch.setattr(task.torch, "load", fake_torch_load)
    monkeypatch.setattr(task, "MnistConvAE", _DummyAE)
    monkeypatch.setattr(task, "_encode_split", fake_encode_split)
    def fake_mnist(*, root: str | Path, train: bool, download: bool, transform: object) -> object:
        del train, transform
        assert Path(root) == mnist_root
        assert download is False
        return object()

    monkeypatch.setattr(task.datasets, "MNIST", fake_mnist)
    monkeypatch.setattr(task, "DataLoader", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        task,
        "default_runtime_roots",
        lambda: {
            "project_name": "drift-flow-matching",
            "data_root": str(tmp_path / "data"),
            "runs_root": str(run_root.parent),
            "mnist_root": str(mnist_root),
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mnist_encode",
            "--ae-ckpt",
            str(run_root / "latest" / "ae_final.pt"),
            "--device",
            "cpu",
            "--output-dir",
            str(run_root / "latest"),
        ],
    )

    task.main()

    assert loaded["path"] == ae_ckpt
    assert (run_dir / "train_latents.npy").exists()
    assert (run_dir / "train_labels.npy").exists()
    assert (run_dir / "test_latents.npy").exists()
    assert (run_dir / "test_labels.npy").exists()
    assert not (run_root / "latest" / "train_latents.npy").exists()
