"""Portable inference checkpoints: tensor state plus plain metadata, no pickled models."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import torch
from ..models.mlp import DriftFlowVelocityMLP, MeanVelocityMLP, VelocityMLP


MODEL_TYPES = {
    "drift_flow_matching": DriftFlowVelocityMLP,
    "mean_flow": MeanVelocityMLP,
    "flow_matching": VelocityMLP,
}


def save_checkpoint(path, model, *, method, dataset_key, hidden_sizes, training_config,
                    method_config, seed, losses):
    payload = {
        "format_version": 1, "method": method, "dataset": dataset_key, "dim": 2,
        "hidden_sizes": list(hidden_sizes), "seed": seed,
        "training": asdict(training_config), "method_config": dict(method_config),
        "losses": list(losses),
        "weights": "ema" if method_config.get("use_ema", False) else "online",
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    torch.save(payload, Path(path))


def load_checkpoint(path, device, *, expected_method, expected_dataset):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported trajectory checkpoint format")
    if payload["method"] != expected_method or payload["dataset"] != expected_dataset:
        raise ValueError("Checkpoint method/dataset does not match the selected experiment")
    model = MODEL_TYPES[payload["method"]](payload["dim"], payload["hidden_sizes"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), {key: value for key, value in payload.items() if key != "state_dict"}
