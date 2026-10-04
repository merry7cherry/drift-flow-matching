"""Run a tiny synthetic CPU train/checkpoint/sample/evaluation check, without downloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ..data.ffhq_latent import FFHQ_CLASS_NAMES
from ..evaluation.registry import evaluate_checkpoint
from ..training.trainer import train_from_config_path
from ..utils import normalize_path
from .sample import sample_checkpoint


def run_smoke(output_dir: str | Path) -> Path:
    output = normalize_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    data = output / "latents"
    data.mkdir(exist_ok=True)
    for split, seed in [("train", 1), ("test", 2)]:
        rng = np.random.default_rng(seed)
        np.savez(data / f"{split}.npz", **{
            name: rng.normal(loc=index / 5, scale=0.1, size=(8, 4)).astype(np.float32)
            for index, name in enumerate(FFHQ_CLASS_NAMES)
        })
    raw = {
        "runtime": {"data_root": "latents", "runs_root": "runs"},
        "dataset": {"name": "ffhq_latent", "params": {
            "train_npz_path": "{data_root}/train.npz", "test_npz_path": "{data_root}/test.npz"}},
        "architecture": {"name": "ffhq_latent_mlp", "params": {"hidden_sizes": [16, 16], "class_embedding_dim": 4}},
        "method": {"name": "drift_flow_matching", "params": {"drift_form": "split_v0", "groups_per_class": 1, "sinkhorn_iters": 2}},
        "trainer": {"epochs": 1, "steps_per_epoch": 2, "batch_size": 12, "device": "cpu", "checkpoint_every": 1},
        "evaluation": {"evaluator": "ffhq", "num_sampling_steps": [1, 2], "sample_every": 0,
            "run_final": False, "metrics": ["latent_ot"], "params": {"n_per_class": 4, "device": "cpu"}},
        "output": {"root_dir": "{runs_root}", "run_name": "synthetic"},
    }
    config = output / "smoke.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False))
    run = train_from_config_path(config)
    checkpoint = run / "checkpoint_final.pt"
    for nfe in [1, 2]:
        sample_checkpoint(checkpoint, output_dir=output / f"samples_{nfe}", num_steps=nfe, samples_per_class=2)
    result = evaluate_checkpoint(checkpoint, output_dir=output / "evaluation")
    report = output / "smoke_result.json"
    report.write_text(json.dumps({
        "scope": "synthetic six-class four-dimensional latents; software check, not FFHQ image quality",
        "checkpoint": str(checkpoint), "evaluation": str(result.output_path), "metrics": result.summary,
    }, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(run_smoke(args.output_dir))


if __name__ == "__main__":
    main()
