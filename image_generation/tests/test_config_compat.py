"""The old public config/checkpoint format stays readable at the input boundary."""
import sys
from pathlib import Path

import pytest
import torch

from driftfm.architectures import ConditionalTimeMLP
from driftfm.config import coerce_experiment_config, load_evaluation_config
from driftfm.data import Conditioning
from driftfm.evaluation.latent import load_generation_stack


def _legacy_experiment():
    return {
        "dataset": {"name": "mnist_latent"},
        "architecture": {"name": "mnist_latent_mlp", "params": {
            "input_dim": 4, "num_classes": 2, "hidden_sizes": [8],
            "class_emb_dim": 3, "conditioning_mode": "class",
        }},
        "method": {"name": "drift_flow_matching", "params": {"drift_form": "split_v0"}},
    }


def test_legacy_config_normalizes_to_current_names():
    config = coerce_experiment_config(_legacy_experiment())
    assert config.method.params == {}
    assert config.architecture.params == {
        "data_dim": 4, "num_classes": 2, "hidden_sizes": [8], "class_embedding_dim": 3,
    }


def test_legacy_checkpoint_reloads_unchanged_weights_and_predictions():
    model = ConditionalTimeMLP(data_dim=4, num_classes=2, hidden_sizes=[8], class_embedding_dim=3)
    experiment = _legacy_experiment()
    loaded, method = load_generation_stack({
        "experiment": experiment,
        "resolved_architecture": experiment["architecture"],
        "model_state": model.state_dict(),
    }, torch.device("cpu"))
    noise = torch.randn(4, 4)
    labels = Conditioning(torch.tensor([0, 0, 1, 1]))
    assert model.state_dict().keys() == loaded.state_dict().keys()
    for nfe in (1, 2, 5):
        assert torch.equal(method.sample(model, noise, labels, num_steps=nfe),
                           method.sample(loaded, noise, labels, num_steps=nfe))


def test_legacy_non_mainline_recipe_is_rejected_at_input():
    experiment = _legacy_experiment()
    experiment["method"]["params"]["drift_form"] = "split_v1"
    with pytest.raises(ValueError, match="Unsupported legacy drift_form"):
        coerce_experiment_config(experiment)


@pytest.mark.parametrize("dataset", ["mnist", "ffhq"])
def test_evaluation_cli_defaults_match_supplied_config(dataset, monkeypatch):
    from driftfm.evaluation import ffhq, mnist

    argv = ["evaluate", "--checkpoint", "generator.pt"]
    if dataset == "mnist":
        argv += ["--ae-ckpt", "autoencoder.pt"]
    monkeypatch.setattr(sys, "argv", argv)
    args = (mnist if dataset == "mnist" else ffhq)._parse_args()
    yaml_path = Path(__file__).parents[1] / "configs" / "experiments" / f"{dataset}_eval.yaml"
    assert args.metrics == load_evaluation_config(yaml_path).metrics
    assert "pca" not in args.metrics and "umap" not in args.metrics


@pytest.mark.parametrize("flags, expected", [
    ([], True), (["--fid-decode-noise"], True), (["--no-fid-decode-noise"], False),
])
def test_ffhq_cli_decoder_noise_default_and_explicit_overrides(flags, expected, monkeypatch):
    from driftfm.evaluation import ffhq

    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", "generator.pt", *flags])
    assert ffhq._parse_args().fid_decode_noise is expected
    if not flags:
        yaml_path = Path(__file__).parents[1] / "configs" / "experiments" / "ffhq_eval.yaml"
        assert expected == load_evaluation_config(yaml_path).params["fid_decode_noise"]
