from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

config = pytest.importorskip("driftfm.config")
checkpoints = pytest.importorskip("driftfm.training.checkpoints")
trainer = pytest.importorskip("driftfm.training.trainer")
EvaluationResult = pytest.importorskip("driftfm.evaluation.base").EvaluationResult


def _ffhq_training_config(
    tmp_path: Path,
    train_npz: Path,
    *,
    epochs: int,
    test_npz: Path | None = None,
    run_name: str = "resume_case",
    sample_every: int = 0,
    sample_count_per_class: int = 8,
    num_sampling_steps: list[int] | None = None,
    metrics_every: int = 0,
    run_final: bool = False,
    metrics: list[str] | None = None,
    evaluation_params: dict[str, Any] | None = None,
) -> object:
    dataset_params: dict[str, object] = {
        "train_npz_path": str(train_npz),
        "split": "train",
    }
    if test_npz is not None:
        dataset_params["test_npz_path"] = str(test_npz)
    return config.coerce_experiment_config(
        {
            "dataset": {
                "name": "ffhq_latent",
                "params": dataset_params,
            },
            "architecture": {
                "name": "ffhq_latent_mlp",
                "params": {
                    "hidden_sizes": [16, 16],
                    "class_embedding_dim": 8,
                },
            },
            "method": {
                "name": "drift_flow_matching",
                "params": {
                    "groups_per_class": 1,
                    "drift_form": "split_v0",
                    "use_ema": True,
                    "ema_decay": 0.9,
                    "sinkhorn_iters": 2,
                },
            },
            "trainer": {
                "epochs": epochs,
                "batch_size": 6,
                "steps_per_epoch": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
                "device": "cpu",
                "seed": 123,
                "log_every": 1,
                "checkpoint_every": 1,
            },
            "evaluation": {
                "sample_every": sample_every,
                "sample_count_per_class": sample_count_per_class,
                "num_sampling_steps": [1, 2, 5, 10, 20, 50] if num_sampling_steps is None else list(num_sampling_steps),
                "metrics_every": metrics_every,
                "run_final": run_final,
                "metrics": [] if metrics is None else list(metrics),
                "params": {} if evaluation_params is None else dict(evaluation_params),
            },
            "output": {
                "root_dir": str(tmp_path / "runs"),
                "run_name": run_name,
            },
        }
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as arrays:
        return {name: np.array(arrays[name]) for name in arrays.files}


def _numpy_state_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    if left[0] != right[0]:
        return False
    if not np.array_equal(left[1], right[1]):
        return False
    return left[2:] == right[2:]


def _state_dict_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def _python_state_equal(left: object, right: object) -> bool:
    return left == right


def _make_eval_stub(
    tmp_path: Path,
    *,
    fail_on_calls: set[int] | None = None,
    calls: list[dict[str, Any]] | None = None,
):
    fail_on_calls = set() if fail_on_calls is None else set(fail_on_calls)

    def _fake_evaluate(checkpoint_path: str | Path, *, experiment: object, output_dir: str | Path) -> EvaluationResult:
        call_index = len(calls or []) + 1
        random.random()
        torch.rand(4)
        np.random.rand(4)
        payload = checkpoints.load_checkpoint(checkpoint_path, map_location="cpu")
        if calls is not None:
            calls.append(
                {
                    "call_index": call_index,
                    "epoch": int(payload["epoch"]),
                    "checkpoint": str(Path(checkpoint_path)),
                    "output_dir": str(Path(output_dir)),
                }
            )
        if call_index in fail_on_calls:
            raise RuntimeError(f"synthetic eval failure #{call_index}")
        output_path = Path(output_dir) / "ffhq_eval.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")
        return EvaluationResult(output_path=output_path, summary={"score": float(call_index)})

    return _fake_evaluate


def test_save_checkpoint_uses_atomic_replace(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint_final.pt"
    checkpoints.save_checkpoint(checkpoint_path, {"epoch": 1, "value": torch.tensor([1.0])})

    assert checkpoint_path.exists()
    assert checkpoints.load_checkpoint(checkpoint_path, map_location="cpu")["epoch"] == 1
    assert list(tmp_path.glob("checkpoint_final.pt.tmp-*")) == []


def test_restore_rng_state_normalizes_tensor_states_before_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTensor:
        def __init__(self, *, name: str, device_type: str, dtype: torch.dtype) -> None:
            self.name = name
            self.device = type("Device", (), {"type": device_type})()
            self.dtype = dtype

        def detach(self) -> "_FakeTensor":
            return self

        def cpu(self) -> "_FakeTensor":
            return _FakeTensor(name=self.name, device_type="cpu", dtype=self.dtype)

        def to(self, *, dtype: torch.dtype) -> "_FakeTensor":
            return _FakeTensor(name=self.name, device_type=self.device.type, dtype=dtype)

    class _FakeGenerator:
        def __init__(self) -> None:
            self.state: object | None = None

        def set_state(self, state: object) -> None:
            self.state = state

    dataset = type("Dataset", (), {"_generator": _FakeGenerator()})()
    calls: dict[str, object] = {}

    monkeypatch.setattr(torch, "set_rng_state", lambda state: calls.setdefault("torch", state))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda state: calls.setdefault("cuda", state))
    monkeypatch.setattr(random, "setstate", lambda state: calls.setdefault("python", state))

    checkpoint = {
        "python_rng_state": ("python", "state"),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": _FakeTensor(name="torch", device_type="cuda", dtype=torch.int64),
        "cuda_rng_state_all": [
            _FakeTensor(name="cuda0", device_type="cuda", dtype=torch.int64),
            _FakeTensor(name="cuda1", device_type="cuda", dtype=torch.float32),
        ],
        "dataset_generator_state": _FakeTensor(name="dataset", device_type="cuda", dtype=torch.int16),
    }

    trainer._restore_rng_state(checkpoint, dataset=dataset)

    restored_torch = calls["torch"]
    assert getattr(getattr(restored_torch, "device", None), "type", None) == "cpu"
    assert getattr(restored_torch, "dtype", None) == torch.uint8

    assert calls["python"] == checkpoint["python_rng_state"]

    restored_cuda = calls["cuda"]
    assert isinstance(restored_cuda, list)
    assert len(restored_cuda) == 2
    assert all(getattr(getattr(state, "device", None), "type", None) == "cpu" for state in restored_cuda)
    assert all(getattr(state, "dtype", None) == torch.uint8 for state in restored_cuda)

    restored_dataset = dataset._generator.state
    assert getattr(getattr(restored_dataset, "device", None), "type", None) == "cpu"
    assert getattr(restored_dataset, "dtype", None) == torch.uint8


def test_train_resume_continues_from_next_epoch(
    tmp_path: Path,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    first_config = _ffhq_training_config(tmp_path, synthetic_ffhq_npz["train_npz"], epochs=1)
    run_dir = trainer.train_from_config(first_config)
    first_checkpoint = checkpoints.load_checkpoint(run_dir / "checkpoint_final.pt", map_location="cpu")

    second_config = _ffhq_training_config(tmp_path, synthetic_ffhq_npz["train_npz"], epochs=2)
    resumed_run_dir = trainer.train_from_config(second_config, resume_from=run_dir / "checkpoint_final.pt")

    assert resumed_run_dir == run_dir
    final_checkpoint = checkpoints.load_checkpoint(run_dir / "checkpoint_final.pt", map_location="cpu")
    log_rows = _read_jsonl(run_dir / "logs.jsonl")

    assert [row["epoch"] for row in log_rows] == [1, 2]
    assert final_checkpoint["epoch"] == 2
    assert final_checkpoint["step"] > first_checkpoint["step"]
    assert final_checkpoint["optimizer_state"]["state"]
    assert final_checkpoint["ema_state"] is not None


def test_train_from_config_uses_stable_seed_hash_run_name_when_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    def _fake_datetime(token: str):
        class _FakeNow:
            def strftime(self, fmt: str) -> str:
                assert fmt == "%Y%m%d_%H%M%S"
                return token

        class _FakeDateTime:
            @staticmethod
            def now() -> _FakeNow:
                return _FakeNow()

        return _FakeDateTime

    first_config = _ffhq_training_config(tmp_path, synthetic_ffhq_npz["train_npz"], epochs=0, run_name=None)
    first_config.output.run_name = None
    monkeypatch.setattr(trainer, "datetime", _fake_datetime("20260417_010101"))
    first_run_dir = trainer.train_from_config(first_config)

    second_config = _ffhq_training_config(tmp_path, synthetic_ffhq_npz["train_npz"], epochs=0, run_name=None)
    second_config.output.run_name = None
    monkeypatch.setattr(trainer, "datetime", _fake_datetime("20260417_020202"))
    second_run_dir = trainer.train_from_config(second_config)

    assert first_run_dir == second_run_dir
    assert first_run_dir.name.startswith("seed123_")


def test_train_from_config_injects_dataset_seed_into_resolved_config(
    tmp_path: Path,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    training_config = _ffhq_training_config(tmp_path, synthetic_ffhq_npz["train_npz"], epochs=0, run_name="seeded")

    assert "seed" not in training_config.dataset.params

    run_dir = trainer.train_from_config(training_config)
    resolved_config = config.load_experiment_config(run_dir / "resolved_config.yaml")
    metadata = _read_json(run_dir / "run_metadata.json")

    assert resolved_config.dataset.params["seed"] == training_config.trainer.seed
    assert metadata["resolved_seed"] == training_config.trainer.seed
    assert metadata["dataset_seed"] == training_config.trainer.seed
    assert metadata["determinism"]["seed"] == training_config.trainer.seed
    assert metadata["determinism"]["device"] == "cpu"


def test_short_training_run_repeats_training_preview_and_evaluation_outputs_for_same_seed(
    tmp_path: Path,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    common_kwargs = {
        "epochs": 1,
        "test_npz": synthetic_ffhq_npz["test_npz"],
        "sample_every": 1,
        "sample_count_per_class": 1,
        "num_sampling_steps": [1],
        "metrics_every": 1,
        "metrics": ["latent_ot"],
        "run_final": False,
        "evaluation_params": {
            "real_npz": str(synthetic_ffhq_npz["test_npz"]),
            "device": "cpu",
            "n_per_class": 2,
            "gen_batch": 2,
        },
    }
    first_config = _ffhq_training_config(
        tmp_path / "first",
        synthetic_ffhq_npz["train_npz"],
        run_name="seed_short_first",
        **common_kwargs,
    )
    second_config = _ffhq_training_config(
        tmp_path / "second",
        synthetic_ffhq_npz["train_npz"],
        run_name="seed_short_second",
        **common_kwargs,
    )
    third_config = _ffhq_training_config(
        tmp_path / "third",
        synthetic_ffhq_npz["train_npz"],
        run_name="seed_short_third",
        **common_kwargs,
    )
    third_config.trainer.seed = 321

    first_run_dir = trainer.train_from_config(first_config)
    second_run_dir = trainer.train_from_config(second_config)
    third_run_dir = trainer.train_from_config(third_config)

    first_checkpoint = checkpoints.load_checkpoint(first_run_dir / "checkpoint_final.pt", map_location="cpu")
    second_checkpoint = checkpoints.load_checkpoint(second_run_dir / "checkpoint_final.pt", map_location="cpu")
    third_checkpoint = checkpoints.load_checkpoint(third_run_dir / "checkpoint_final.pt", map_location="cpu")
    first_preview = _load_npz_payload(first_run_dir / "samples" / "epoch001" / "steps_1" / "samples.npz")
    second_preview = _load_npz_payload(second_run_dir / "samples" / "epoch001" / "steps_1" / "samples.npz")
    third_preview = _load_npz_payload(third_run_dir / "samples" / "epoch001" / "steps_1" / "samples.npz")
    first_eval_latents = _load_npz_payload(
        first_run_dir / "evals" / "epoch001" / "steps_1" / "sampled_latents_by_class.npz"
    )
    second_eval_latents = _load_npz_payload(
        second_run_dir / "evals" / "epoch001" / "steps_1" / "sampled_latents_by_class.npz"
    )
    third_eval_latents = _load_npz_payload(
        third_run_dir / "evals" / "epoch001" / "steps_1" / "sampled_latents_by_class.npz"
    )
    first_metadata = _read_json(first_run_dir / "run_metadata.json")
    second_metadata = _read_json(second_run_dir / "run_metadata.json")
    third_metadata = _read_json(third_run_dir / "run_metadata.json")
    first_eval_index = _read_json(first_run_dir / "evals" / "epoch001" / "evaluation_index.json")
    second_eval_index = _read_json(second_run_dir / "evals" / "epoch001" / "evaluation_index.json")
    third_eval_index = _read_json(third_run_dir / "evals" / "epoch001" / "evaluation_index.json")
    first_eval_leaf = _read_json(first_run_dir / "evals" / "epoch001" / "steps_1" / "ffhq_eval.json")
    second_eval_leaf = _read_json(second_run_dir / "evals" / "epoch001" / "steps_1" / "ffhq_eval.json")
    third_eval_leaf = _read_json(third_run_dir / "evals" / "epoch001" / "steps_1" / "ffhq_eval.json")

    assert _state_dict_equal(first_checkpoint["model_state"], second_checkpoint["model_state"])
    assert any(
        not torch.equal(first_checkpoint["model_state"][key], third_checkpoint["model_state"][key])
        for key in first_checkpoint["model_state"]
    )
    assert first_metadata == second_metadata
    assert first_metadata["resolved_seed"] == 123
    assert first_metadata["dataset_seed"] == 123
    assert third_metadata["resolved_seed"] == 321
    assert first_eval_index["determinism"] == second_eval_index["determinism"]
    assert third_eval_index["determinism"]["seed"] == 321
    assert first_eval_leaf["determinism"] == second_eval_leaf["determinism"]
    assert first_eval_leaf["resolved_seed"] == 123
    assert second_eval_leaf["resolved_seed"] == 123
    assert third_eval_leaf["resolved_seed"] == 321
    for key in first_preview:
        assert np.array_equal(first_preview[key], second_preview[key])
    for key in first_eval_latents:
        assert np.array_equal(first_eval_latents[key], second_eval_latents[key])
    assert any(not np.array_equal(first_preview[key], third_preview[key]) for key in first_preview)
    assert any(not np.array_equal(first_eval_latents[key], third_eval_latents[key]) for key in first_eval_latents)


def test_training_is_reproducible_for_same_seed_and_changes_for_different_seed(
    tmp_path: Path,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    first_config = _ffhq_training_config(tmp_path, synthetic_ffhq_npz["train_npz"], epochs=1, run_name="seed_first")
    second_config = _ffhq_training_config(tmp_path, synthetic_ffhq_npz["train_npz"], epochs=1, run_name="seed_second")
    third_config = _ffhq_training_config(tmp_path, synthetic_ffhq_npz["train_npz"], epochs=1, run_name="seed_third")
    third_config.trainer.seed = 321

    first_run_dir = trainer.train_from_config(first_config)
    second_run_dir = trainer.train_from_config(second_config)
    third_run_dir = trainer.train_from_config(third_config)

    first_checkpoint = checkpoints.load_checkpoint(first_run_dir / "checkpoint_final.pt", map_location="cpu")
    second_checkpoint = checkpoints.load_checkpoint(second_run_dir / "checkpoint_final.pt", map_location="cpu")
    third_checkpoint = checkpoints.load_checkpoint(third_run_dir / "checkpoint_final.pt", map_location="cpu")

    assert _state_dict_equal(first_checkpoint["model_state"], second_checkpoint["model_state"])
    assert any(
        not torch.equal(first_checkpoint["model_state"][key], third_checkpoint["model_state"][key])
        for key in first_checkpoint["model_state"]
    )


def test_sampling_previews_do_not_perturb_training_rng(
    tmp_path: Path,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    baseline_config = _ffhq_training_config(tmp_path / "baseline", synthetic_ffhq_npz["train_npz"], epochs=2, run_name="baseline")
    preview_config = _ffhq_training_config(tmp_path / "preview", synthetic_ffhq_npz["train_npz"], epochs=2, run_name="preview")
    preview_config.evaluation.sample_every = 1
    preview_config.evaluation.sample_count_per_class = 1
    preview_config.evaluation.num_sampling_steps = [1]

    baseline_run_dir = trainer.train_from_config(baseline_config)
    preview_run_dir = trainer.train_from_config(preview_config)

    baseline_checkpoint = checkpoints.load_checkpoint(baseline_run_dir / "checkpoint_final.pt", map_location="cpu")
    preview_checkpoint = checkpoints.load_checkpoint(preview_run_dir / "checkpoint_final.pt", map_location="cpu")

    assert _state_dict_equal(baseline_checkpoint["model_state"], preview_checkpoint["model_state"])


def test_inline_evaluation_does_not_perturb_training_rng_or_final_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    baseline_config = _ffhq_training_config(
        tmp_path / "baseline",
        synthetic_ffhq_npz["train_npz"],
        epochs=2,
        run_name="baseline_inline_rng",
    )
    inline_eval_config = _ffhq_training_config(
        tmp_path / "inline_eval",
        synthetic_ffhq_npz["train_npz"],
        epochs=2,
        run_name="inline_eval_rng",
        metrics_every=1,
        metrics=["latent_ot"],
    )

    monkeypatch.setattr(trainer, "evaluate_checkpoint", _make_eval_stub(tmp_path / "inline_eval"))
    baseline_run_dir = trainer.train_from_config(baseline_config)
    inline_eval_run_dir = trainer.train_from_config(inline_eval_config)

    baseline_checkpoint = checkpoints.load_checkpoint(baseline_run_dir / "checkpoint_final.pt", map_location="cpu")
    inline_eval_checkpoint = checkpoints.load_checkpoint(inline_eval_run_dir / "checkpoint_final.pt", map_location="cpu")

    assert _state_dict_equal(baseline_checkpoint["model_state"], inline_eval_checkpoint["model_state"])
    assert _python_state_equal(baseline_checkpoint["python_rng_state"], inline_eval_checkpoint["python_rng_state"])
    assert _numpy_state_equal(baseline_checkpoint["numpy_rng_state"], inline_eval_checkpoint["numpy_rng_state"])
    assert torch.equal(baseline_checkpoint["torch_rng_state"], inline_eval_checkpoint["torch_rng_state"])
    assert torch.equal(
        baseline_checkpoint["dataset_generator_state"],
        inline_eval_checkpoint["dataset_generator_state"],
    )


def test_find_latest_checkpoint_skips_corrupt_final_and_epoch_files(
    tmp_path: Path,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    first_config = _ffhq_training_config(tmp_path, synthetic_ffhq_npz["train_npz"], epochs=1)
    run_dir = trainer.train_from_config(first_config)
    final_checkpoint_path = run_dir / "checkpoint_final.pt"
    payload = checkpoints.load_checkpoint(final_checkpoint_path, map_location="cpu")

    corrupt_epoch_path = run_dir / "checkpoint_epoch2.pt"
    corrupt_epoch_path.write_bytes(b"not a checkpoint")
    final_checkpoint_path.write_bytes(b"also not a checkpoint")
    checkpoints.save_checkpoint(run_dir / "checkpoint_epoch1.pt", payload)

    assert checkpoints.find_latest_checkpoint(run_dir) == run_dir / "checkpoint_epoch1.pt"


def test_train_resume_auto_ignores_corrupt_final_checkpoint(
    tmp_path: Path,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    first_config = _ffhq_training_config(tmp_path, synthetic_ffhq_npz["train_npz"], epochs=1)
    run_dir = trainer.train_from_config(first_config)
    final_checkpoint_path = run_dir / "checkpoint_final.pt"
    payload = checkpoints.load_checkpoint(final_checkpoint_path, map_location="cpu")
    checkpoints.save_checkpoint(run_dir / "checkpoint_epoch1.pt", payload)
    final_checkpoint_path.write_bytes(b"corrupt")

    second_config = _ffhq_training_config(tmp_path, synthetic_ffhq_npz["train_npz"], epochs=2)
    trainer.train_from_config(second_config, resume_from="auto")

    log_rows = _read_jsonl(run_dir / "logs.jsonl")
    assert [row["epoch"] for row in log_rows] == [1, 2]
    assert checkpoints.find_latest_checkpoint(run_dir) == run_dir / "checkpoint_final.pt"


def test_train_resume_replays_pending_inline_evaluation_before_next_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    config_with_eval = _ffhq_training_config(
        tmp_path,
        synthetic_ffhq_npz["train_npz"],
        epochs=2,
        run_name="resume_pending_epoch",
        metrics_every=1,
        metrics=["latent_ot"],
    )
    failing_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        trainer,
        "evaluate_checkpoint",
        _make_eval_stub(tmp_path, fail_on_calls={1}, calls=failing_calls),
    )

    with pytest.raises(RuntimeError, match="synthetic eval failure #1"):
        trainer.train_from_config(config_with_eval)

    run_dir = Path(config_with_eval.output.root_dir) / str(config_with_eval.output.run_name)
    checkpoint_epoch1 = checkpoints.load_checkpoint(run_dir / "checkpoint_epoch1.pt", map_location="cpu")
    assert checkpoint_epoch1["pending_inline_evaluation"] is True
    assert checkpoint_epoch1["pending_inline_evaluation_epoch"] == 1
    assert Path(checkpoint_epoch1["pending_inline_evaluation_output_dir"]).name == "epoch001"

    resumed_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        trainer,
        "evaluate_checkpoint",
        _make_eval_stub(tmp_path, calls=resumed_calls),
    )

    trainer.train_from_config(config_with_eval, resume_from="auto")

    log_rows = _read_jsonl(run_dir / "logs.jsonl")
    eval_rows = _read_jsonl(run_dir / "evals.jsonl")
    final_checkpoint = checkpoints.load_checkpoint(run_dir / "checkpoint_final.pt", map_location="cpu")

    assert [row["epoch"] for row in log_rows] == [1, 2]
    assert [row["epoch"] for row in eval_rows] == [1, 2]
    assert [call["epoch"] for call in resumed_calls] == [1, 2]
    assert final_checkpoint["epoch"] == 2
    assert final_checkpoint["pending_inline_evaluation"] is False
    assert final_checkpoint["pending_inline_evaluation_epoch"] is None
    assert final_checkpoint["pending_inline_evaluation_output_dir"] is None


def test_train_resume_final_pending_eval_only_replays_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    config_with_eval = _ffhq_training_config(
        tmp_path,
        synthetic_ffhq_npz["train_npz"],
        epochs=1,
        run_name="resume_pending_final",
        metrics_every=1,
        metrics=["latent_ot"],
    )
    monkeypatch.setattr(
        trainer,
        "evaluate_checkpoint",
        _make_eval_stub(tmp_path, fail_on_calls={1}),
    )

    with pytest.raises(RuntimeError, match="synthetic eval failure #1"):
        trainer.train_from_config(config_with_eval)

    run_dir = Path(config_with_eval.output.root_dir) / str(config_with_eval.output.run_name)
    checkpoint_before = checkpoints.load_checkpoint(run_dir / "checkpoint_final.pt", map_location="cpu")
    assert checkpoint_before["pending_inline_evaluation"] is True

    resumed_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        trainer,
        "evaluate_checkpoint",
        _make_eval_stub(tmp_path, calls=resumed_calls),
    )

    resumed_run_dir = trainer.train_from_config(config_with_eval, resume_from=run_dir / "checkpoint_final.pt")
    checkpoint_after = checkpoints.load_checkpoint(run_dir / "checkpoint_final.pt", map_location="cpu")

    assert resumed_run_dir == run_dir
    assert checkpoint_after["epoch"] == 1
    assert checkpoint_after["step"] == checkpoint_before["step"]
    assert checkpoint_after["pending_inline_evaluation"] is False
    assert [row["epoch"] for row in _read_jsonl(run_dir / "logs.jsonl")] == [1]
    assert [row["epoch"] for row in _read_jsonl(run_dir / "evals.jsonl")] == [1]
    assert [call["epoch"] for call in resumed_calls] == [1]


def test_resume_after_failed_inline_eval_matches_uninterrupted_rng_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_ffhq_npz: dict[str, Path],
) -> None:
    full_config = _ffhq_training_config(
        tmp_path / "full",
        synthetic_ffhq_npz["train_npz"],
        epochs=2,
        run_name="rng_full",
        metrics_every=1,
        metrics=["latent_ot"],
    )
    interrupted_config = _ffhq_training_config(
        tmp_path / "interrupted",
        synthetic_ffhq_npz["train_npz"],
        epochs=2,
        run_name="rng_interrupted",
        metrics_every=1,
        metrics=["latent_ot"],
    )

    monkeypatch.setattr(trainer, "evaluate_checkpoint", _make_eval_stub(tmp_path / "full"))
    full_run_dir = trainer.train_from_config(full_config)
    full_checkpoint = checkpoints.load_checkpoint(full_run_dir / "checkpoint_final.pt", map_location="cpu")

    monkeypatch.setattr(
        trainer,
        "evaluate_checkpoint",
        _make_eval_stub(tmp_path / "interrupted", fail_on_calls={1}),
    )
    with pytest.raises(RuntimeError, match="synthetic eval failure #1"):
        trainer.train_from_config(interrupted_config)

    monkeypatch.setattr(trainer, "evaluate_checkpoint", _make_eval_stub(tmp_path / "interrupted_resume"))
    resumed_run_dir = trainer.train_from_config(interrupted_config, resume_from="auto")
    resumed_checkpoint = checkpoints.load_checkpoint(resumed_run_dir / "checkpoint_final.pt", map_location="cpu")

    assert resumed_checkpoint["epoch"] == full_checkpoint["epoch"]
    assert resumed_checkpoint["step"] == full_checkpoint["step"]
    assert _state_dict_equal(resumed_checkpoint["model_state"], full_checkpoint["model_state"])
    assert _python_state_equal(resumed_checkpoint["python_rng_state"], full_checkpoint["python_rng_state"])
    assert torch.equal(resumed_checkpoint["torch_rng_state"], full_checkpoint["torch_rng_state"])
    assert _numpy_state_equal(resumed_checkpoint["numpy_rng_state"], full_checkpoint["numpy_rng_state"])
    assert torch.equal(resumed_checkpoint["dataset_generator_state"], full_checkpoint["dataset_generator_state"])
