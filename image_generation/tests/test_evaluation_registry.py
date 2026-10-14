from __future__ import annotations

import json
from pathlib import Path

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


@pytest.mark.parametrize("dataset", ["mnist", "ffhq"])
@pytest.mark.parametrize("requested", [[], ["latent_ot", "pca", "umap"]])
def test_generic_evaluation_uses_numeric_defaults_and_keeps_explicit_projections(
    dataset, requested, tmp_path, monkeypatch,
):
    from driftfm.evaluation import registry

    experiment = config.coerce_experiment_config({
        "dataset": {"name": f"{dataset}_latent"},
        "architecture": {"name": f"{dataset}_latent_mlp"},
        "method": {"name": "drift_flow_matching"},
        "evaluation": {"metrics": requested, "num_sampling_steps": [1]},
    })
    config_path = Path(__file__).parents[1] / "configs" / "experiments" / f"{dataset}_eval.yaml"
    expected = requested or config.load_evaluation_config(config_path).metrics
    evaluator = registry.get_evaluator(dataset)
    observed = []

    def evaluate(*args, experiment, output_dir, **kwargs):
        observed.append(experiment.evaluation.metrics)
        output_dir.mkdir(parents=True)
        path = output_dir / "metrics.json"
        path.write_text("{}")
        return evaluation.EvaluationResult(output_path=path, summary={})

    monkeypatch.setattr(registry, "load_checkpoint", lambda *args: {})
    monkeypatch.setattr(evaluator, "evaluate", evaluate)
    result = registry.evaluate_checkpoint(
        tmp_path / "generator.pt", experiment=experiment, output_dir=tmp_path / "evaluation",
    )
    assert observed == [expected]
    assert json.loads(result.output_path.read_text())["metrics"] == expected


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
