from __future__ import annotations


import pytest


config = pytest.importorskip("driftfm.config")


evaluation = pytest.importorskip("driftfm.evaluation")


latent_eval = pytest.importorskip("driftfm.evaluation.latent")


def test_resolve_evaluator_name_defaults_from_dataset() -> None:
    experiment = config.coerce_experiment_config(
        {
            "dataset": {"name": "mnist_latent", "params": {}},
            "architecture": {"name": "mnist_latent_mlp", "params": {}},
            "method": {"name": "drift_flow_matching", "params": {}},
        }
    )
    assert evaluation.resolve_evaluator_name(experiment) == "mnist"


def test_get_evaluator_returns_registered_instance() -> None:
    evaluator = evaluation.get_evaluator("ffhq")
    assert evaluator.name == "ffhq"
    assert "latent_ot" in evaluator.supported_metrics


def test_metric_validation_rejects_unsupported_metrics() -> None:
    evaluator = evaluation.get_evaluator("mnist")
    with pytest.raises(ValueError):
        evaluator.validate_metrics(["fid"])


def test_resolve_evaluation_seed_prefers_params_and_falls_back_to_trainer_seed() -> None:
    fallback_experiment = config.coerce_experiment_config(
        {
            "dataset": {"name": "mnist_latent", "params": {}},
            "architecture": {"name": "mnist_latent_mlp", "params": {}},
            "method": {"name": "drift_flow_matching", "params": {}},
            "trainer": {"seed": 123},
        }
    )
    override_experiment = config.coerce_experiment_config(
        {
            "dataset": {"name": "mnist_latent", "params": {}},
            "architecture": {"name": "mnist_latent_mlp", "params": {}},
            "method": {"name": "drift_flow_matching", "params": {}},
            "trainer": {"seed": 123},
            "evaluation": {"params": {"seed": 456}},
        }
    )

    assert latent_eval.resolve_evaluation_seed(fallback_experiment) == 123
    assert latent_eval.resolve_evaluation_seed(override_experiment) == 456
