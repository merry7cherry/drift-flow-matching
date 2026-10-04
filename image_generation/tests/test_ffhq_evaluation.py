from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

config = pytest.importorskip("driftfm.config")
ffhq_eval = pytest.importorskip("driftfm.evaluation.ffhq")
evaluation_registry = pytest.importorskip("driftfm.evaluation.registry")
checkpoints = pytest.importorskip("driftfm.training.checkpoints")


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
        samples = noise.clone()
        samples[:, 0] = conditioning.class_labels.float()
        return samples


def _decode_stub(calls: list[Path]):
    def _impl(
        alae_model: object,
        latents,
        output_dir: Path,
        *,
        device: torch.device,
        batch_size: int,
        save_size: int,
        noise: bool | str,
        generator: torch.Generator | None = None,
    ) -> None:
        del alae_model, device, batch_size, save_size, noise, generator
        calls.append(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for index in range(int(latents.shape[0])):
            (output_dir / f"{index:06d}.png").write_bytes(b"png")

    return _impl


def test_ffhq_evaluator_supports_pca_for_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    experiment = config.coerce_experiment_config(
        {
            "dataset": {"name": "ffhq_latent", "params": {"train_npz_path": str(synthetic_ffhq_npz["train_npz"])}},
            "architecture": {"name": "ffhq_latent_mlp", "params": {"conditioning_mode": "class"}},
            "method": {"name": "drift_flow_matching", "params": {"drift_form": "split_v0"}},
            "trainer": {"device": "cpu"},
            "evaluation": {
                "evaluator": "ffhq",
                "metrics": ["latent_ot", "pca", "tsne", "umap", "lda"],
                "params": {
                    "real_npz": str(synthetic_ffhq_npz["test_npz"]),
                    "device": "cpu",
                    "n_per_class": 3,
                    "gen_batch": 4,
                },
            },
        }
    )
    checkpoint = {
        "resolved_architecture": {
            "name": "ffhq_latent_mlp",
            "params": {"data_dim": 512, "conditioning_mode": "class"},
        }
    }
    output_dir = tmp_path / "evals" / "ffhq"
    fake_method = _FakeMethod()

    monkeypatch.setattr(ffhq_eval, "load_generation_stack", lambda checkpoint, device: (object(), fake_method))
    monkeypatch.setattr(ffhq_eval, "compute_ot_distance", lambda *args, **kwargs: 0.5)

    result = ffhq_eval.FFHQEvaluator().evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=experiment,
        checkpoint=checkpoint,
        output_dir=output_dir,
        num_sampling_steps=3,
    )

    payload = json.loads((output_dir / "ffhq_eval.json").read_text(encoding="utf-8"))
    assert result.output_path == output_dir / "ffhq_eval.json"
    assert payload["resolved_seed"] == 42
    _assert_determinism_payload(payload, seed=42, device="cpu")
    assert payload["fid_decode_noise"] is True
    assert payload["mean_latent_ot"] == 0.5
    sampled_latents_path = Path(payload["sampled_latents_npz"])
    assert sampled_latents_path.exists()
    _assert_sampled_latents_npz(sampled_latents_path, class_names=ffhq_eval.FFHQ_CLASS_NAMES)
    for method_name in ("pca", "tsne", "umap", "lda"):
        _assert_projection_payload(payload, method_name=method_name)
    assert set(fake_method.calls) == {(3, None)}


def test_ffhq_evaluator_rejects_accuracy_metric() -> None:
    with pytest.raises(ValueError):
        ffhq_eval.FFHQEvaluator().validate_metrics(["accuracy"])


def test_ffhq_evaluator_requires_alae_runtime_for_fid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    experiment = config.coerce_experiment_config(
        {
            "dataset": {"name": "ffhq_latent", "params": {"train_npz_path": str(synthetic_ffhq_npz["train_npz"])}},
            "architecture": {"name": "ffhq_latent_mlp", "params": {"conditioning_mode": "class"}},
            "method": {"name": "drift_flow_matching", "params": {"drift_form": "split_v0"}},
            "trainer": {"device": "cpu"},
            "evaluation": {
                "evaluator": "ffhq",
                "metrics": ["fid"],
                "params": {
                    "real_npz": str(synthetic_ffhq_npz["test_npz"]),
                    "device": "cpu",
                    "n_per_class": 3,
                    "gen_batch": 4,
                },
            },
        }
    )
    checkpoint = {
        "resolved_architecture": {
            "name": "ffhq_latent_mlp",
            "params": {"data_dim": 512, "conditioning_mode": "class"},
        }
    }
    fake_method = _FakeMethod()

    monkeypatch.setattr(ffhq_eval, "load_generation_stack", lambda checkpoint, device: (object(), fake_method))

    with pytest.raises(ValueError, match="evaluation.params.alae_ckpt"):
        ffhq_eval.FFHQEvaluator().evaluate(
            tmp_path / "checkpoint_final.pt",
            experiment=experiment,
            checkpoint=checkpoint,
            output_dir=tmp_path / "evals" / "ffhq",
            num_sampling_steps=3,
        )


def test_ffhq_evaluator_forwards_fid_num_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    experiment = config.coerce_experiment_config(
        {
            "dataset": {"name": "ffhq_latent", "params": {"train_npz_path": str(synthetic_ffhq_npz["train_npz"])}},
            "architecture": {"name": "ffhq_latent_mlp", "params": {"conditioning_mode": "class"}},
            "method": {"name": "drift_flow_matching", "params": {"drift_form": "split_v0"}},
            "trainer": {"device": "cpu"},
            "evaluation": {
                "evaluator": "ffhq",
                "metrics": ["fid"],
                "params": {
                    "real_npz": str(synthetic_ffhq_npz["test_npz"]),
                    "device": "cpu",
                    "n_per_class": 3,
                    "gen_batch": 4,
                    "decode_batch": 2,
                    "fid_batch": 4,
                    "fid_num_workers": 1,
                    "alae_ckpt": str(tmp_path / "alae_ffhq.pt"),
                },
            },
        }
    )
    checkpoint = {
        "resolved_architecture": {
            "name": "ffhq_latent_mlp",
            "params": {"data_dim": 512, "conditioning_mode": "class"},
        }
    }
    fake_method = _FakeMethod()
    calls: list[int] = []

    monkeypatch.setattr(ffhq_eval, "load_generation_stack", lambda checkpoint, device: (object(), fake_method))
    monkeypatch.setattr(ffhq_eval, "_load_internal_alae_model", lambda device, path: object())
    monkeypatch.setattr(
        ffhq_eval,
        "_decode_and_save",
        lambda *args, **kwargs: args[2].mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(
        ffhq_eval,
        "_compute_fid",
        lambda *args, **kwargs: calls.append(int(kwargs["fid_num_workers"])) or 1.0,
    )

    ffhq_eval.FFHQEvaluator().evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=experiment,
        checkpoint=checkpoint,
        output_dir=tmp_path / "evals" / "ffhq",
        num_sampling_steps=3,
    )

    assert calls == [1] * len(ffhq_eval.FFHQ_CLASS_NAMES)


def test_ffhq_evaluator_resolves_test_split_and_writes_fid_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    experiment = config.coerce_experiment_config(
        {
            "dataset": {
                "name": "ffhq_latent",
                "params": {
                    "train_npz_path": str(synthetic_ffhq_npz["train_npz"]),
                    "test_npz_path": str(synthetic_ffhq_npz["test_npz"]),
                },
            },
            "architecture": {"name": "ffhq_latent_mlp", "params": {"conditioning_mode": "class"}},
            "method": {"name": "drift_flow_matching", "params": {"drift_form": "split_v0"}},
            "trainer": {"device": "cpu"},
            "evaluation": {
                "evaluator": "ffhq",
                "metrics": ["fid"],
                "params": {
                    "real_split": "test",
                    "device": "cpu",
                    "n_per_class": 3,
                    "gen_batch": 4,
                    "decode_batch": 2,
                    "fid_batch": 4,
                    "alae_ckpt": str(tmp_path / "alae_ffhq.pt"),
                },
            },
        }
    )
    checkpoint = {
        "resolved_architecture": {
            "name": "ffhq_latent_mlp",
            "params": {"data_dim": 512, "conditioning_mode": "class"},
        }
    }
    fake_method = _FakeMethod()
    decode_calls: list[Path] = []
    output_dir = tmp_path / "evals" / "ffhq"

    monkeypatch.setattr(ffhq_eval, "load_generation_stack", lambda checkpoint, device: (object(), fake_method))
    monkeypatch.setattr(ffhq_eval, "_load_internal_alae_model", lambda device, path: object())
    monkeypatch.setattr(ffhq_eval, "_decode_and_save", _decode_stub(decode_calls))
    monkeypatch.setattr(ffhq_eval, "_compute_fid", lambda *args, **kwargs: 1.0)

    ffhq_eval.FFHQEvaluator().evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=experiment,
        checkpoint=checkpoint,
        output_dir=output_dir,
        num_sampling_steps=3,
    )

    payload = json.loads((output_dir / "ffhq_eval.json").read_text(encoding="utf-8"))
    fid_artifacts = json.loads((output_dir / "fid_artifacts.json").read_text(encoding="utf-8"))
    assert payload["resolved_real_split"] == "test"
    assert payload["resolved_real_npz"] == str(synthetic_ffhq_npz["test_npz"])
    assert payload["real_input_source"] == "dataset_test"
    assert payload["resolved_seed"] == 42
    assert payload["fid_decode_noise"] is True
    assert payload["real_images_dir"] == str((output_dir / "real_images").resolve())
    assert payload["fake_images_dir"] == str((output_dir / "fake_images").resolve())
    assert fid_artifacts["real"]["resolved_real_split"] == "test"
    assert fid_artifacts["real"]["resolved_real_npz"] == str(synthetic_ffhq_npz["test_npz"])
    assert fid_artifacts["real"]["fid_decode_noise"] is True
    assert fid_artifacts["real"]["resolved_seed"] == 42
    assert fid_artifacts["fake"]["root_dir"] == str((output_dir / "fake_images").resolve())
    assert fid_artifacts["fake"]["alae_source"]["type"] == "project_checkpoint"
    assert fid_artifacts["fake"]["fid_decode_noise"] is True
    assert fid_artifacts["fake"]["resolved_seed"] == 42
    assert len(decode_calls) == 2 * len(ffhq_eval.FFHQ_CLASS_NAMES)


def test_ffhq_evaluator_real_npz_override_beats_real_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    experiment = config.coerce_experiment_config(
        {
            "dataset": {
                "name": "ffhq_latent",
                "params": {
                    "train_npz_path": str(synthetic_ffhq_npz["train_npz"]),
                    "test_npz_path": str(synthetic_ffhq_npz["test_npz"]),
                },
            },
            "architecture": {"name": "ffhq_latent_mlp", "params": {"conditioning_mode": "class"}},
            "method": {"name": "drift_flow_matching", "params": {"drift_form": "split_v0"}},
            "trainer": {"device": "cpu"},
            "evaluation": {
                "evaluator": "ffhq",
                "metrics": ["latent_ot"],
                "params": {
                    "real_split": "test",
                    "real_npz": str(synthetic_ffhq_npz["train_npz"]),
                    "device": "cpu",
                    "n_per_class": 3,
                    "gen_batch": 4,
                },
            },
        }
    )
    checkpoint = {
        "resolved_architecture": {
            "name": "ffhq_latent_mlp",
            "params": {"data_dim": 512, "conditioning_mode": "class"},
        }
    }

    monkeypatch.setattr(ffhq_eval, "load_generation_stack", lambda checkpoint, device: (object(), _FakeMethod()))
    monkeypatch.setattr(ffhq_eval, "compute_ot_distance", lambda *args, **kwargs: 0.5)

    ffhq_eval.FFHQEvaluator().evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=experiment,
        checkpoint=checkpoint,
        output_dir=tmp_path / "evals" / "ffhq",
        num_sampling_steps=3,
    )

    payload = json.loads((tmp_path / "evals" / "ffhq" / "ffhq_eval.json").read_text(encoding="utf-8"))
    assert payload["resolved_real_split"] is None
    assert payload["resolved_real_npz"] == str(synthetic_ffhq_npz["train_npz"])
    assert payload["real_input_source"] == "explicit_npz"


def test_ffhq_evaluator_missing_target_split_npz_raises_without_train_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    experiment = config.coerce_experiment_config(
        {
            "dataset": {
                "name": "ffhq_latent",
                "params": {
                    "train_npz_path": str(synthetic_ffhq_npz["train_npz"]),
                    "test_npz_path": str(tmp_path / "missing_test_latents.npz"),
                },
            },
            "architecture": {"name": "ffhq_latent_mlp", "params": {"conditioning_mode": "class"}},
            "method": {"name": "drift_flow_matching", "params": {"drift_form": "split_v0"}},
            "trainer": {"device": "cpu"},
            "evaluation": {
                "evaluator": "ffhq",
                "metrics": ["latent_ot"],
                "params": {
                    "real_split": "test",
                    "device": "cpu",
                    "n_per_class": 3,
                    "gen_batch": 4,
                },
            },
        }
    )
    checkpoint = {
        "resolved_architecture": {
            "name": "ffhq_latent_mlp",
            "params": {"data_dim": 512, "conditioning_mode": "class"},
        }
    }

    monkeypatch.setattr(ffhq_eval, "load_generation_stack", lambda checkpoint, device: (object(), _FakeMethod()))

    with pytest.raises(FileNotFoundError, match="real_split=test"):
        ffhq_eval.FFHQEvaluator().evaluate(
            tmp_path / "checkpoint_final.pt",
            experiment=experiment,
            checkpoint=checkpoint,
            output_dir=tmp_path / "evals" / "ffhq",
            num_sampling_steps=3,
        )


def test_ffhq_evaluator_reuses_only_real_images_when_metadata_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    experiment = config.coerce_experiment_config(
        {
            "dataset": {
                "name": "ffhq_latent",
                "params": {
                    "train_npz_path": str(synthetic_ffhq_npz["train_npz"]),
                    "test_npz_path": str(synthetic_ffhq_npz["test_npz"]),
                },
            },
            "architecture": {"name": "ffhq_latent_mlp", "params": {"conditioning_mode": "class"}},
            "method": {"name": "drift_flow_matching", "params": {"drift_form": "split_v0"}},
            "trainer": {"device": "cpu"},
            "evaluation": {
                "evaluator": "ffhq",
                "metrics": ["fid"],
                "params": {
                    "real_split": "test",
                    "device": "cpu",
                    "n_per_class": 3,
                    "gen_batch": 4,
                    "decode_batch": 2,
                    "fid_batch": 4,
                    "reuse_images": True,
                    "alae_ckpt": str(tmp_path / "alae_ffhq.pt"),
                },
            },
        }
    )
    checkpoint = {
        "resolved_architecture": {
            "name": "ffhq_latent_mlp",
            "params": {"data_dim": 512, "conditioning_mode": "class"},
        }
    }
    decode_calls: list[Path] = []
    output_dir = tmp_path / "evals" / "ffhq"

    monkeypatch.setattr(ffhq_eval, "load_generation_stack", lambda checkpoint, device: (object(), _FakeMethod()))
    monkeypatch.setattr(ffhq_eval, "_load_internal_alae_model", lambda device, path: object())
    monkeypatch.setattr(ffhq_eval, "_decode_and_save", _decode_stub(decode_calls))
    monkeypatch.setattr(ffhq_eval, "_compute_fid", lambda *args, **kwargs: 1.0)

    evaluator = ffhq_eval.FFHQEvaluator()
    evaluator.evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=experiment,
        checkpoint=checkpoint,
        output_dir=output_dir,
        num_sampling_steps=3,
    )
    evaluator.evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=experiment,
        checkpoint=checkpoint,
        output_dir=output_dir,
        num_sampling_steps=3,
    )

    real_calls = [path for path in decode_calls if "real_images" in path.parts]
    fake_calls = [path for path in decode_calls if "fake_images" in path.parts]
    assert len(real_calls) == len(ffhq_eval.FFHQ_CLASS_NAMES)
    assert len(fake_calls) == 2 * len(ffhq_eval.FFHQ_CLASS_NAMES)


def test_ffhq_evaluator_rebuilds_real_images_when_metadata_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    base_payload = {
        "dataset": {
            "name": "ffhq_latent",
            "params": {
                "train_npz_path": str(synthetic_ffhq_npz["train_npz"]),
                "test_npz_path": str(synthetic_ffhq_npz["test_npz"]),
            },
        },
        "architecture": {"name": "ffhq_latent_mlp", "params": {"conditioning_mode": "class"}},
        "method": {"name": "drift_flow_matching", "params": {"drift_form": "split_v0"}},
        "trainer": {"device": "cpu"},
        "evaluation": {
            "evaluator": "ffhq",
            "metrics": ["fid"],
            "params": {
                "real_split": "test",
                "device": "cpu",
                "n_per_class": 3,
                "gen_batch": 4,
                "decode_batch": 2,
                "fid_batch": 4,
                "reuse_images": True,
                "alae_ckpt": str(tmp_path / "alae_ffhq.pt"),
            },
        },
    }
    first_experiment = config.coerce_experiment_config(base_payload)
    second_payload = json.loads(json.dumps(base_payload))
    second_payload["evaluation"]["params"]["save_size"] = 512
    second_experiment = config.coerce_experiment_config(second_payload)
    checkpoint = {
        "resolved_architecture": {
            "name": "ffhq_latent_mlp",
            "params": {"data_dim": 512, "conditioning_mode": "class"},
        }
    }
    decode_calls: list[Path] = []
    output_dir = tmp_path / "evals" / "ffhq"

    monkeypatch.setattr(ffhq_eval, "load_generation_stack", lambda checkpoint, device: (object(), _FakeMethod()))
    monkeypatch.setattr(ffhq_eval, "_load_internal_alae_model", lambda device, path: object())
    monkeypatch.setattr(ffhq_eval, "_decode_and_save", _decode_stub(decode_calls))
    monkeypatch.setattr(ffhq_eval, "_compute_fid", lambda *args, **kwargs: 1.0)

    evaluator = ffhq_eval.FFHQEvaluator()
    evaluator.evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=first_experiment,
        checkpoint=checkpoint,
        output_dir=output_dir,
        num_sampling_steps=3,
    )
    evaluator.evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=second_experiment,
        checkpoint=checkpoint,
        output_dir=output_dir,
        num_sampling_steps=3,
    )

    real_calls = [path for path in decode_calls if "real_images" in path.parts]
    fake_calls = [path for path in decode_calls if "fake_images" in path.parts]
    assert len(real_calls) == 2 * len(ffhq_eval.FFHQ_CLASS_NAMES)
    assert len(fake_calls) == 2 * len(ffhq_eval.FFHQ_CLASS_NAMES)


def test_ffhq_registry_evaluation_index_records_resolved_real_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    experiment = config.coerce_experiment_config(
        {
            "dataset": {
                "name": "ffhq_latent",
                "params": {
                    "train_npz_path": str(synthetic_ffhq_npz["train_npz"]),
                    "test_npz_path": str(synthetic_ffhq_npz["test_npz"]),
                },
            },
            "architecture": {"name": "ffhq_latent_mlp", "params": {"conditioning_mode": "class"}},
            "method": {"name": "drift_flow_matching", "params": {"drift_form": "split_v0"}},
            "trainer": {"device": "cpu"},
            "evaluation": {
                "evaluator": "ffhq",
                "metrics": ["latent_ot"],
                "num_sampling_steps": [3],
                "params": {
                    "real_split": "test",
                    "device": "cpu",
                    "n_per_class": 3,
                    "gen_batch": 4,
                },
            },
        }
    )
    checkpoint_path = tmp_path / "checkpoint_final.pt"
    checkpoints.save_checkpoint(
        checkpoint_path,
        {
            "experiment": experiment.to_dict(),
            "resolved_architecture": {
                "name": "ffhq_latent_mlp",
                "params": {"data_dim": 512, "conditioning_mode": "class"},
            },
            "model_state": {},
            "ema_state": None,
        },
    )

    monkeypatch.setattr(ffhq_eval, "load_generation_stack", lambda checkpoint, device: (object(), _FakeMethod()))
    monkeypatch.setattr(ffhq_eval, "compute_ot_distance", lambda *args, **kwargs: 0.5)

    output_path = evaluation_registry.evaluate_checkpoint(
        checkpoint_path,
        experiment=experiment,
        output_dir=tmp_path / "evals",
    ).output_path

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["resolved_seed"] == 42
    _assert_determinism_payload(payload, seed=42, device="cpu")
    assert payload["resolved_real_split"] == "test"
    assert payload["resolved_real_npz"] == str(synthetic_ffhq_npz["test_npz"])
    assert payload["steps"][0]["resolved_seed"] == 42
    assert payload["steps"][0]["determinism"] == payload["determinism"]
    assert payload["steps"][0]["resolved_real_split"] == "test"
    assert payload["steps"][0]["resolved_real_npz"] == str(synthetic_ffhq_npz["test_npz"])


def test_ffhq_fid_decode_defaults_to_noisy_and_can_disable_noise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    def _build_experiment(*, fid_decode_noise: bool | None) -> object:
        params: dict[str, object] = {
            "real_npz": str(synthetic_ffhq_npz["test_npz"]),
            "device": "cpu",
            "n_per_class": 3,
            "gen_batch": 4,
            "decode_batch": 2,
            "fid_batch": 4,
            "alae_ckpt": str(tmp_path / "alae_ffhq.pt"),
        }
        if fid_decode_noise is not None:
            params["fid_decode_noise"] = fid_decode_noise
        return config.coerce_experiment_config(
            {
                "dataset": {"name": "ffhq_latent", "params": {"train_npz_path": str(synthetic_ffhq_npz["train_npz"])}},
                "architecture": {"name": "ffhq_latent_mlp", "params": {"conditioning_mode": "class"}},
                "method": {"name": "drift_flow_matching", "params": {"drift_form": "split_v0"}},
                "trainer": {"device": "cpu", "seed": 77},
                "evaluation": {
                    "evaluator": "ffhq",
                    "metrics": ["fid"],
                    "params": params,
                },
            }
        )

    checkpoint = {
        "resolved_architecture": {
            "name": "ffhq_latent_mlp",
            "params": {"data_dim": 512, "conditioning_mode": "class"},
        }
    }
    fake_method = _FakeMethod()
    noise_calls: list[bool] = []

    def _record_decode(*args, **kwargs) -> None:
        noise_calls.append(bool(kwargs["noise"]))
        args[2].mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ffhq_eval, "load_generation_stack", lambda checkpoint, device: (object(), fake_method))
    monkeypatch.setattr(ffhq_eval, "_load_internal_alae_model", lambda device, path: object())
    monkeypatch.setattr(ffhq_eval, "_decode_and_save", _record_decode)
    monkeypatch.setattr(ffhq_eval, "_compute_fid", lambda *args, **kwargs: 1.0)

    default_output = tmp_path / "evals" / "default"
    deterministic_output = tmp_path / "evals" / "deterministic"
    ffhq_eval.FFHQEvaluator().evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=_build_experiment(fid_decode_noise=None),
        checkpoint=checkpoint,
        output_dir=default_output,
        num_sampling_steps=3,
    )
    ffhq_eval.FFHQEvaluator().evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=_build_experiment(fid_decode_noise=False),
        checkpoint=checkpoint,
        output_dir=deterministic_output,
        num_sampling_steps=3,
    )

    split = len(ffhq_eval.FFHQ_CLASS_NAMES) * 2
    assert noise_calls[:split] == [True] * split
    assert noise_calls[split:] == [False] * split
    default_payload = json.loads((default_output / "ffhq_eval.json").read_text(encoding="utf-8"))
    deterministic_payload = json.loads((deterministic_output / "ffhq_eval.json").read_text(encoding="utf-8"))
    assert default_payload["fid_decode_noise"] is True
    assert deterministic_payload["fid_decode_noise"] is False
    assert deterministic_payload["resolved_seed"] == 77


def test_ffhq_evaluator_sampling_is_reproducible_for_same_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    base_payload = {
        "dataset": {"name": "ffhq_latent", "params": {"train_npz_path": str(synthetic_ffhq_npz["train_npz"])}},
        "architecture": {"name": "ffhq_latent_mlp", "params": {"conditioning_mode": "class"}},
        "method": {"name": "drift_flow_matching", "params": {"drift_form": "split_v0"}},
        "trainer": {"device": "cpu", "seed": 17},
        "evaluation": {
            "evaluator": "ffhq",
            "metrics": ["latent_ot"],
            "params": {
                "real_npz": str(synthetic_ffhq_npz["test_npz"]),
                "device": "cpu",
                "n_per_class": 3,
                "gen_batch": 4,
            },
        },
    }
    same_seed_experiment = config.coerce_experiment_config(base_payload)
    different_seed_payload = json.loads(json.dumps(base_payload))
    different_seed_payload["trainer"]["seed"] = 23
    different_seed_experiment = config.coerce_experiment_config(different_seed_payload)
    checkpoint = {
        "resolved_architecture": {
            "name": "ffhq_latent_mlp",
            "params": {"data_dim": 512, "conditioning_mode": "class"},
        }
    }

    monkeypatch.setattr(ffhq_eval, "load_generation_stack", lambda checkpoint, device: (object(), _FakeMethod()))
    monkeypatch.setattr(ffhq_eval, "compute_ot_distance", lambda *args, **kwargs: 0.5)

    first_dir = tmp_path / "evals" / "first"
    second_dir = tmp_path / "evals" / "second"
    third_dir = tmp_path / "evals" / "third"
    ffhq_eval.FFHQEvaluator().evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=same_seed_experiment,
        checkpoint=checkpoint,
        output_dir=first_dir,
        num_sampling_steps=3,
    )
    ffhq_eval.FFHQEvaluator().evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=same_seed_experiment,
        checkpoint=checkpoint,
        output_dir=second_dir,
        num_sampling_steps=3,
    )
    ffhq_eval.FFHQEvaluator().evaluate(
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


def test_ffhq_real_cache_tracks_actual_image_counts_when_class_is_short(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    with np.load(synthetic_ffhq_npz["test_npz"]) as arrays:
        truncated_payload = {
            class_name: (
                np.asarray(arrays[class_name][:1], dtype=np.float32)
                if class_name == ffhq_eval.FFHQ_CLASS_NAMES[0]
                else np.asarray(arrays[class_name][:3], dtype=np.float32)
            )
            for class_name in ffhq_eval.FFHQ_CLASS_NAMES
        }
    truncated_npz = tmp_path / "truncated_test.npz"
    np.savez_compressed(truncated_npz, **truncated_payload)

    experiment = config.coerce_experiment_config(
        {
            "dataset": {"name": "ffhq_latent", "params": {"train_npz_path": str(synthetic_ffhq_npz["train_npz"])}},
            "architecture": {"name": "ffhq_latent_mlp", "params": {"conditioning_mode": "class"}},
            "method": {"name": "drift_flow_matching", "params": {"drift_form": "split_v0"}},
            "trainer": {"device": "cpu", "seed": 33},
            "evaluation": {
                "evaluator": "ffhq",
                "metrics": ["fid"],
                "params": {
                    "real_npz": str(truncated_npz),
                    "device": "cpu",
                    "n_per_class": 3,
                    "gen_batch": 4,
                    "decode_batch": 2,
                    "fid_batch": 4,
                    "reuse_images": True,
                    "alae_ckpt": str(tmp_path / "alae_ffhq.pt"),
                },
            },
        }
    )
    checkpoint = {
        "resolved_architecture": {
            "name": "ffhq_latent_mlp",
            "params": {"data_dim": 512, "conditioning_mode": "class"},
        }
    }
    decode_calls: list[Path] = []
    output_dir = tmp_path / "evals" / "ffhq"

    monkeypatch.setattr(ffhq_eval, "load_generation_stack", lambda checkpoint, device: (object(), _FakeMethod()))
    monkeypatch.setattr(ffhq_eval, "_load_internal_alae_model", lambda device, path: object())
    monkeypatch.setattr(ffhq_eval, "_decode_and_save", _decode_stub(decode_calls))
    monkeypatch.setattr(ffhq_eval, "_compute_fid", lambda *args, **kwargs: 1.0)

    evaluator = ffhq_eval.FFHQEvaluator()
    evaluator.evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=experiment,
        checkpoint=checkpoint,
        output_dir=output_dir,
        num_sampling_steps=3,
    )
    evaluator.evaluate(
        tmp_path / "checkpoint_final.pt",
        experiment=experiment,
        checkpoint=checkpoint,
        output_dir=output_dir,
        num_sampling_steps=3,
    )

    fid_artifacts = json.loads((output_dir / "fid_artifacts.json").read_text(encoding="utf-8"))
    assert fid_artifacts["real"]["per_class_image_counts"][ffhq_eval.FFHQ_CLASS_NAMES[0]] == 1
    assert fid_artifacts["real"]["per_class_image_counts"][ffhq_eval.FFHQ_CLASS_NAMES[1]] == 3
    real_calls = [path for path in decode_calls if "real_images" in path.parts]
    fake_calls = [path for path in decode_calls if "fake_images" in path.parts]
    assert len(real_calls) == len(ffhq_eval.FFHQ_CLASS_NAMES)
    assert len(fake_calls) == 2 * len(ffhq_eval.FFHQ_CLASS_NAMES)
