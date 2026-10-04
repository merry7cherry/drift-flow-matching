from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

config = pytest.importorskip("driftfm.config")
mnist_eval = pytest.importorskip("driftfm.evaluation.mnist")
eval_registry = pytest.importorskip("driftfm.evaluation.registry")


class _FastUMAP:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.center: np.ndarray | None = None

    def fit(self, points: np.ndarray) -> "_FastUMAP":
        self.center = np.asarray(points, dtype=np.float32).mean(axis=0)
        return self

    def transform(self, points: np.ndarray) -> np.ndarray:
        assert self.center is not None
        centered = np.asarray(points, dtype=np.float32) - self.center
        if centered.shape[1] == 1:
            return np.concatenate([centered, np.zeros((centered.shape[0], 1), dtype=np.float32)], axis=1)
        return centered[:, :2]


class _FastTSNE:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit_transform(self, points: np.ndarray) -> np.ndarray:
        centered = np.asarray(points, dtype=np.float32) - np.asarray(points, dtype=np.float32).mean(axis=0)
        if centered.shape[1] == 1:
            return np.concatenate([centered, np.zeros((centered.shape[0], 1), dtype=np.float32)], axis=1)
        return centered[:, :2]


@pytest.fixture(autouse=True)
def _fast_projection_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    import sklearn.manifold

    monkeypatch.setattr(sklearn.manifold, "TSNE", _FastTSNE)
    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=_FastUMAP))


def _assert_determinism_payload(payload: dict[str, object], *, seed: int, device: str) -> None:
    determinism = payload["determinism"]
    assert isinstance(determinism, dict)
    assert determinism["seed"] == seed
    assert determinism["deterministic"] is True
    assert determinism["device"] == device
    assert determinism["torch_deterministic_algorithms"] is True
    assert determinism["cudnn_benchmark"] is False
    assert determinism["cudnn_deterministic"] is True
    assert determinism["cuda_available"] is False


def _assert_projection_artifacts(
    method_name: str,
    plot_paths: list[Path],
    projection_paths: list[Path],
    *,
    expected_fit_bases: set[str],
) -> None:
    fit_bases: set[str] = set()
    for projection_path in projection_paths:
        projection = json.loads(projection_path.read_text(encoding="utf-8"))
        assert projection["method"] == method_name
        assert set(projection["stages"]) == {"fake", "real"}
        assert "source" not in projection["stages"]
        fit_bases.add(str(projection["fit_basis"]))
    assert fit_bases == expected_fit_bases
    for plot_path in plot_paths:
        with Image.open(plot_path) as image:
            assert image.width >= 2200
            assert image.height >= 1600


def _assert_projection_payload(payload: dict[str, object], *, method_name: str) -> None:
    plot_path = Path(str(payload[f"{method_name}_plot_path"]))
    real_only_plot_path = Path(str(payload[f"{method_name}_real_only_plot_path"]))
    generated_only_plot_path = Path(str(payload[f"{method_name}_generated_only_plot_path"]))
    projection_path = Path(str(payload[f"{method_name}_projection_path"]))
    assert plot_path.exists()
    assert real_only_plot_path.exists()
    assert generated_only_plot_path.exists()
    assert projection_path.exists()

    if method_name == "tsne":
        assert f"{method_name}_real_fit_plot_path" not in payload
        _assert_projection_artifacts(
            method_name,
            [plot_path, real_only_plot_path, generated_only_plot_path],
            [projection_path],
            expected_fit_bases={"fake_real"},
        )
        return

    real_fit_plot_path = Path(str(payload[f"{method_name}_real_fit_plot_path"]))
    real_fit_real_only_plot_path = Path(str(payload[f"{method_name}_real_fit_real_only_plot_path"]))
    real_fit_generated_only_plot_path = Path(str(payload[f"{method_name}_real_fit_generated_only_plot_path"]))
    real_fit_projection_path = Path(str(payload[f"{method_name}_real_fit_projection_path"]))
    assert real_fit_plot_path.exists()
    assert real_fit_real_only_plot_path.exists()
    assert real_fit_generated_only_plot_path.exists()
    assert real_fit_projection_path.exists()
    _assert_projection_artifacts(
        method_name,
        [
            plot_path,
            real_only_plot_path,
            generated_only_plot_path,
            real_fit_plot_path,
            real_fit_real_only_plot_path,
            real_fit_generated_only_plot_path,
        ],
        [projection_path, real_fit_projection_path],
        expected_fit_bases={"fake_real", "real_only"},
    )


def _assert_sampled_latents_npz(npz_path: Path, *, class_names: list[str]) -> None:
    with np.load(npz_path) as sampled_latents:
        keys = set(sampled_latents.files)
    expected_keys = {f"real_{class_name}" for class_name in class_names} | {
        f"fake_{class_name}" for class_name in class_names
    }
    assert not any(key.startswith("source_") for key in keys)
    assert keys == expected_keys


class _FakeAE:
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        base = latents[:, :1].view(-1, 1, 1, 1)
        return base.expand(-1, 1, 28, 28).clamp(0.0, 1.0)


class _FakeMethod:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float | None]] = []

    def sample(
        self,
        model: object,
        noise: torch.Tensor,
        conditioning: object,
        *,
        num_steps: int,
    ) -> torch.Tensor:
        del model
        self.calls.append((num_steps, None))
        latents = noise.clone()
        labels = conditioning.class_labels.float()
        latents[:, 0] = labels / max(1.0, labels.max().item())
        return latents


def test_mnist_evaluator_writes_class_grid_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_mnist_latents: dict[str, Path],
) -> None:
    test_latents = np.load(synthetic_mnist_latents["test_latents"])
    test_labels = np.load(synthetic_mnist_latents["test_labels"])
    experiment = config.coerce_experiment_config(
        {
            "dataset": {"name": "mnist_latent", "params": {"root_dir": str(synthetic_mnist_latents["root"])}},
            "architecture": {"name": "mnist_latent_mlp", "params": {}},
            "method": {"name": "drift_flow_matching", "params": {}},
            "trainer": {"device": "cpu"},
            "evaluation": {
                "evaluator": "mnist",

                "metrics": ["latent_ot"],
                "sample_count_per_class": 3,
                "params": {
                    "ae_ckpt": str(tmp_path / "ae_final.pt"),
                    "latent_root": str(synthetic_mnist_latents["root"]),
                    "classifier_ckpt": str(tmp_path / "mnist_clf.pt"),
                    "device": "cpu",
                    "n_per_class": 2,
                },
            },
            "output": {"root_dir": str(tmp_path / "runs"), "run_name": "mnist_eval"},
        }
    )
    checkpoint = {
        "dataset_info": {"data_dim": 6, "num_classes": 10},
        "resolved_architecture": {"name": "mnist_latent_mlp", "params": {"conditioning_mode": "class"}},
    }
    output_dir = tmp_path / "evals" / "final"
    fake_method = _FakeMethod()

    monkeypatch.setattr(mnist_eval, "_load_ae", lambda path, device: _FakeAE())
    monkeypatch.setattr(mnist_eval, "load_generation_stack", lambda checkpoint, device: (object(), fake_method))
    monkeypatch.setattr(
        mnist_eval,
        "_load_real_latents_or_encode",
        lambda *args, **kwargs: (test_latents, test_labels),
    )
    monkeypatch.setattr(mnist_eval, "compute_ot_distance", lambda *args, **kwargs: 0.25)

    result = mnist_eval.MnistEvaluator().evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=experiment,
        checkpoint=checkpoint,
        output_dir=output_dir,
        num_sampling_steps=3,
    )

    payload = json.loads((output_dir / "mnist_eval.json").read_text(encoding="utf-8"))
    class_grid = output_dir / "mnist_class_grid.png"
    assert result.output_path == output_dir / "mnist_eval.json"
    assert class_grid.exists()
    assert class_grid.stat().st_size > 0
    assert payload["class_grid_path"] == str(class_grid.resolve())
    assert payload["class_grid_samples_per_class"] == 3
    assert payload["class_grid_rows"] == list(range(10))
    assert payload["num_sampling_steps"] == 3
    assert payload["resolved_seed"] == 42
    assert payload["mean_latent_ot"] == 0.25
    _assert_determinism_payload(payload, seed=42, device="cpu")
    assert set(fake_method.calls) == {(3, None)}


def test_evaluate_checkpoint_writes_multi_step_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_mnist_latents: dict[str, Path],
) -> None:
    test_latents = np.load(synthetic_mnist_latents["test_latents"])
    test_labels = np.load(synthetic_mnist_latents["test_labels"])
    experiment = config.coerce_experiment_config(
        {
            "dataset": {"name": "mnist_latent", "params": {"root_dir": str(synthetic_mnist_latents["root"])}},
            "architecture": {"name": "mnist_latent_mlp", "params": {}},
            "method": {"name": "drift_flow_matching", "params": {}},
            "trainer": {"device": "cpu"},
            "evaluation": {
                "evaluator": "mnist",
                "num_sampling_steps": [2, 5],

                "metrics": ["latent_ot"],
                "sample_count_per_class": 2,
                "params": {
                    "ae_ckpt": str(tmp_path / "ae_final.pt"),
                    "latent_root": str(synthetic_mnist_latents["root"]),
                    "classifier_ckpt": str(tmp_path / "mnist_clf.pt"),
                    "device": "cpu",
                    "n_per_class": 2,
                },
            },
            "output": {"root_dir": str(tmp_path / "runs"), "run_name": "mnist_eval"},
        }
    )
    checkpoint = {
        "dataset_info": {"data_dim": 6, "num_classes": 10},
        "resolved_architecture": {"name": "mnist_latent_mlp", "params": {"conditioning_mode": "class"}},
    }
    output_dir = tmp_path / "evals" / "final"
    fake_method = _FakeMethod()

    monkeypatch.setattr(eval_registry, "load_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr(mnist_eval, "_load_ae", lambda path, device: _FakeAE())
    monkeypatch.setattr(mnist_eval, "load_generation_stack", lambda checkpoint, device: (object(), fake_method))
    monkeypatch.setattr(
        mnist_eval,
        "_load_real_latents_or_encode",
        lambda *args, **kwargs: (test_latents, test_labels),
    )
    monkeypatch.setattr(mnist_eval, "compute_ot_distance", lambda *args, **kwargs: 0.25)

    result = eval_registry.evaluate_checkpoint(
        tmp_path / "checkpoint_final.pt",
        experiment=experiment,
        output_dir=output_dir,
    )

    index_path = output_dir / "evaluation_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert result.output_path == index_path
    assert payload["num_sampling_steps"] == [2, 5]
    assert payload["resolved_seed"] == 42
    _assert_determinism_payload(payload, seed=42, device="cpu")
    assert len(payload["steps"]) == 2
    assert result.summary == {
        "steps_2.mean_latent_ot": 0.25,
        "steps_5.mean_latent_ot": 0.25,
    }
    assert set(fake_method.calls) == {(2, None), (5, None)}

    for entry in payload["steps"]:
        steps = entry["num_sampling_steps"]
        step_dir = output_dir / f"steps_{steps}"
        assert Path(entry["output_dir"]) == step_dir.resolve()
        step_payload = json.loads((step_dir / "mnist_eval.json").read_text(encoding="utf-8"))
        assert Path(entry["output_path"]) == (step_dir / "mnist_eval.json").resolve()
        assert step_payload["num_sampling_steps"] == steps
        assert step_payload["resolved_seed"] == 42
        assert entry["determinism"] == step_payload["determinism"]
        _assert_determinism_payload(step_payload, seed=42, device="cpu")


def test_load_real_latents_or_encode_prefers_cached_latents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_latents = np.arange(24, dtype=np.float32).reshape(4, 6)
    expected_labels = np.array([0, 1, 2, 3], dtype=np.int64)
    latents_path = tmp_path / "test_latents.npy"
    labels_path = tmp_path / "test_labels.npy"
    np.save(latents_path, expected_latents)
    np.save(labels_path, expected_labels)

    monkeypatch.setattr(
        mnist_eval.datasets,
        "MNIST",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cached latents should be used")),
    )

    class _NeverEncodeAE:
        def encode(self, images: torch.Tensor) -> torch.Tensor:
            raise AssertionError("cached latents should avoid AE encoding")

    latents, labels = mnist_eval._load_real_latents_or_encode(
        _NeverEncodeAE(),
        latent_paths=(latents_path, labels_path),
        mnist_root=str(tmp_path),
        device=torch.device("cpu"),
    )

    assert np.array_equal(latents, expected_latents)
    assert np.array_equal(labels, expected_labels)


def test_raw_mnist_evaluation_fallbacks_are_read_only_consumers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mnist_root = tmp_path / "canonical-mnist"
    calls: list[tuple[Path, bool, bool]] = []

    class _ToyMNIST(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.full((1, 28, 28), float(index % 2)), index % 2

    def fake_mnist(*, root: str | Path, train: bool, download: bool, transform: object) -> _ToyMNIST:
        del transform
        calls.append((Path(root), train, download))
        return _ToyMNIST()

    class _ToyAE:
        def encode(self, images: torch.Tensor) -> torch.Tensor:
            return images.view(images.shape[0], -1)[:, :6]

    monkeypatch.setattr(mnist_eval.datasets, "MNIST", fake_mnist)

    latents, latent_labels = mnist_eval._load_real_latents_or_encode(
        _ToyAE(),
        latent_paths=(None, None),
        mnist_root=str(mnist_root),
        device=torch.device("cpu"),
    )
    images, image_labels = mnist_eval._load_real_images(mnist_root=str(mnist_root))

    assert latents.shape == (4, 6)
    assert images.shape == (4, 28 * 28)
    assert np.array_equal(latent_labels, image_labels)
    assert calls == [(mnist_root, False, False), (mnist_root, False, False)]


def test_evaluate_checkpoint_class_mode_writes_class_conditioned_pca(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_mnist_latents: dict[str, Path],
) -> None:
    test_latents = np.load(synthetic_mnist_latents["test_latents"])
    test_labels = np.load(synthetic_mnist_latents["test_labels"])
    experiment = config.coerce_experiment_config(
        {
            "dataset": {"name": "mnist_latent", "params": {"root_dir": str(synthetic_mnist_latents["root"])}},
            "architecture": {"name": "mnist_latent_mlp", "params": {"conditioning_mode": "class"}},
            "method": {"name": "drift_flow_matching", "params": {"drift_form": "split_v0"}},
            "trainer": {"device": "cpu"},
            "evaluation": {
                "evaluator": "mnist",
                "num_sampling_steps": [2],
                "metrics": ["pca", "tsne", "umap", "lda"],
                "sample_count_per_class": 2,
                "params": {
                    "ae_ckpt": str(tmp_path / "ae_final.pt"),
                    "latent_root": str(synthetic_mnist_latents["root"]),
                    "classifier_ckpt": str(tmp_path / "mnist_clf.pt"),
                    "device": "cpu",
                    "n_per_class": 2,
                    "gen_batch": 4,
                },
            },
            "output": {"root_dir": str(tmp_path / "runs"), "run_name": "mnist_eval"},
        }
    )
    checkpoint = {
        "dataset_info": {"data_dim": 6, "num_classes": 10},
        "resolved_architecture": {"name": "mnist_latent_mlp", "params": {"conditioning_mode": "class"}},
    }
    output_dir = tmp_path / "evals" / "final"
    fake_method = _FakeMethod()

    monkeypatch.setattr(eval_registry, "load_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr(mnist_eval, "_load_ae", lambda path, device: _FakeAE())
    monkeypatch.setattr(mnist_eval, "load_generation_stack", lambda checkpoint, device: (object(), fake_method))
    monkeypatch.setattr(
        mnist_eval,
        "_load_real_latents_or_encode",
        lambda *args, **kwargs: (test_latents, test_labels),
    )

    result = eval_registry.evaluate_checkpoint(
        tmp_path / "checkpoint_final.pt",
        experiment=experiment,
        output_dir=output_dir,
    )

    index_payload = json.loads((output_dir / "evaluation_index.json").read_text(encoding="utf-8"))
    step_dir = output_dir / "steps_2"
    step_payload = json.loads((step_dir / "mnist_eval.json").read_text(encoding="utf-8"))

    assert result.output_path == output_dir / "evaluation_index.json"
    assert index_payload["resolved_seed"] == 42
    assert len(index_payload["steps"]) == 1
    assert index_payload["steps"][0]["resolved_seed"] == 42
    assert step_payload["resolved_seed"] == 42
    sampled_latents_path = Path(step_payload["sampled_latents_npz"])
    assert sampled_latents_path.exists()
    _assert_sampled_latents_npz(sampled_latents_path, class_names=mnist_eval.MNIST_CLASS_NAMES)
    for method_name in ("pca", "tsne", "umap", "lda"):
        _assert_projection_payload(step_payload, method_name=method_name)
    assert set(fake_method.calls) == {(2, None)}


def test_mnist_evaluator_sampling_is_reproducible_for_same_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_mnist_latents: dict[str, Path],
) -> None:
    test_latents = np.load(synthetic_mnist_latents["test_latents"])
    test_labels = np.load(synthetic_mnist_latents["test_labels"])
    base_payload = {
        "dataset": {"name": "mnist_latent", "params": {"root_dir": str(synthetic_mnist_latents["root"])}},
        "architecture": {"name": "mnist_latent_mlp", "params": {}},
        "method": {"name": "drift_flow_matching", "params": {}},
        "trainer": {"device": "cpu", "seed": 17},
        "evaluation": {
            "evaluator": "mnist",

            "metrics": ["latent_ot"],
            "sample_count_per_class": 2,
            "params": {
                "ae_ckpt": str(tmp_path / "ae_final.pt"),
                "latent_root": str(synthetic_mnist_latents["root"]),
                "classifier_ckpt": str(tmp_path / "mnist_clf.pt"),
                "device": "cpu",
                "n_per_class": 2,
            },
        },
    }
    same_seed_experiment = config.coerce_experiment_config(base_payload)
    different_seed_payload = json.loads(json.dumps(base_payload))
    different_seed_payload["trainer"]["seed"] = 23
    different_seed_experiment = config.coerce_experiment_config(different_seed_payload)
    checkpoint = {
        "dataset_info": {"data_dim": 6, "num_classes": 10},
        "resolved_architecture": {"name": "mnist_latent_mlp", "params": {"conditioning_mode": "class"}},
    }

    monkeypatch.setattr(mnist_eval, "_load_ae", lambda path, device: _FakeAE())
    monkeypatch.setattr(mnist_eval, "load_generation_stack", lambda checkpoint, device: (object(), _FakeMethod()))
    monkeypatch.setattr(
        mnist_eval,
        "_load_real_latents_or_encode",
        lambda *args, **kwargs: (test_latents, test_labels),
    )
    monkeypatch.setattr(mnist_eval, "compute_ot_distance", lambda *args, **kwargs: 0.25)

    first_dir = tmp_path / "evals" / "first"
    second_dir = tmp_path / "evals" / "second"
    third_dir = tmp_path / "evals" / "third"
    mnist_eval.MnistEvaluator().evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=same_seed_experiment,
        checkpoint=checkpoint,
        output_dir=first_dir,
        num_sampling_steps=3,
    )
    mnist_eval.MnistEvaluator().evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=same_seed_experiment,
        checkpoint=checkpoint,
        output_dir=second_dir,
        num_sampling_steps=3,
    )
    mnist_eval.MnistEvaluator().evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=different_seed_experiment,
        checkpoint=checkpoint,
        output_dir=third_dir,
        num_sampling_steps=3,
    )

    with np.load(first_dir / "sampled_latents_by_class.npz") as first_npz, np.load(
        second_dir / "sampled_latents_by_class.npz"
    ) as second_npz, np.load(third_dir / "sampled_latents_by_class.npz") as third_npz:
        for key in first_npz.files:
            assert np.array_equal(first_npz[key], second_npz[key])
        assert any(not np.array_equal(first_npz[key], third_npz[key]) for key in first_npz.files if key.startswith("fake_"))


def test_load_or_train_classifier_is_deterministic_for_missing_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ToyMNIST(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.images = [torch.full((1, 28, 28), float(index % 2)) for index in range(8)]
            self.labels = [index % 2 for index in range(8)]

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return self.images[index], self.labels[index]

    mnist_root = tmp_path / "mnist"

    def fake_mnist(*, root: str | Path, train: bool, download: bool, transform: object) -> _ToyMNIST:
        del train, transform
        assert Path(root) == mnist_root
        assert download is False
        return _ToyMNIST()

    monkeypatch.setattr(mnist_eval.datasets, "MNIST", fake_mnist)

    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    first_classifier = mnist_eval._load_or_train_classifier(
        first_path,
        mnist_root=str(mnist_root),
        device=torch.device("cpu"),
        seed=123,
    )
    second_classifier = mnist_eval._load_or_train_classifier(
        second_path,
        mnist_root=str(mnist_root),
        device=torch.device("cpu"),
        seed=123,
    )

    first_state = first_classifier.state_dict()
    second_state = second_classifier.state_dict()
    assert first_path.exists()
    assert second_path.exists()
    assert first_state.keys() == second_state.keys()
    for key in first_state:
        assert torch.equal(first_state[key], second_state[key])
