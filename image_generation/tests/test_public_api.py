from __future__ import annotations

import pytest

driftfm = pytest.importorskip("driftfm")
config = pytest.importorskip("driftfm.config")
architectures = pytest.importorskip("driftfm.architectures")
datasets = pytest.importorskip("driftfm.data")
methods = pytest.importorskip("driftfm.methods")
evaluation = pytest.importorskip("driftfm.evaluation")


def test_registries_expose_expected_keys() -> None:
    assert set(datasets.DATASETS) == {"mnist_latent", "ffhq_latent"}
    assert set(architectures.ARCHITECTURES) == {
        "mnist_conv_ae",
        "mnist_latent_mlp",
        "ffhq_latent_mlp",
    }
    assert set(methods.METHODS) == {"drift_flow_matching"}


def test_config_modules_export_method_fields() -> None:
    drift_cfg = config.DriftFlowMatchingConfig()
    assert drift_cfg.sinkhorn_iters >= 1
    assert drift_cfg.groups_per_class == 4
    assert drift_cfg.drift_form == "split_v0"
    assert not hasattr(drift_cfg, "coupling")
    assert not hasattr(drift_cfg, "sinkhorn_marginal")
    assert not hasattr(drift_cfg, "unconditional_negatives_per_group")
    assert not hasattr(drift_cfg, "use_flow_matching_loss")
    assert not hasattr(drift_cfg, "flow_matching_loss_weight")
    assert not hasattr(drift_cfg, "guidance_scale")
    assert not hasattr(drift_cfg, "num_sampling_steps")
    assert not hasattr(drift_cfg, "velocity_hidden_sizes")
    assert not hasattr(drift_cfg, "batch_groups")

    eval_cfg = config.EvaluationConfig()
    assert eval_cfg.num_sampling_steps == [1, 2, 5, 10, 20, 50]
    assert eval_cfg.metrics_every >= 0
    assert eval_cfg.run_final is True
    assert isinstance(eval_cfg.params, dict)

    trainer_cfg = config.TrainerConfig()
    assert trainer_cfg.deterministic is True

    runtime_cfg = config.RuntimeConfig()
    assert runtime_cfg.project_name is None
    assert not hasattr(runtime_cfg, "project_root")


def test_legacy_method_config_field_is_not_accepted() -> None:
    with pytest.raises(TypeError):
        config.DriftFlowMatchingConfig(unconditional_negatives_per_group=32)

    with pytest.raises(TypeError):
        config.DriftFlowMatchingConfig(coupling="sinkhorn")

    with pytest.raises(TypeError):
        config.DriftFlowMatchingConfig(sinkhorn_marginal="weighted_cols")


def test_evaluator_registry_exposes_expected_keys() -> None:
    assert set(evaluation.EVALUATORS) == {"mnist", "ffhq"}


def test_package_exports_are_present() -> None:
    assert hasattr(driftfm, "__dict__")
    assert hasattr(config, "load_evaluation_config")
    assert hasattr(architectures, "FFHQALAE")
