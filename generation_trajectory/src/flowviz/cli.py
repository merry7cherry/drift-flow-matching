from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
from .configs import (
    DATASET_CONFIGS, DEFAULT_VISUALIZATION_DATASET_KEYS, DriftFlowMatchingConfig,
    IntegratorConfig, MeanFlowConfig, TrainingConfig, VisualizationEvalConfig,
)
from .runners import ALL_METHODS, VisualizationRunConfig, run_visualization_experiments


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Drift Flow Matching: synthetic generation trajectories")
    parser.add_argument("--config", type=Path, help="JSON configuration (CLI flags override it)")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=["drift_flow_matching"])
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_CONFIGS), default=list(DEFAULT_VISUALIZATION_DATASET_KEYS))
    parser.add_argument("--device", default="cpu", help="PyTorch device: cpu or cuda[:index]")
    parser.add_argument("--epochs", type=positive_int, default=TrainingConfig.epochs)
    parser.add_argument("--steps-per-epoch", type=positive_int, default=TrainingConfig.steps_per_epoch)
    parser.add_argument("--batch-size", type=positive_int, default=TrainingConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--nfe", nargs="+", type=positive_int, default=[1, 20], help="DFM and MeanFlow evaluation NFE")
    parser.add_argument("--fm-nfe", type=positive_int, default=50, help="Euler steps for FM baseline")
    parser.add_argument("--eval-samples", type=positive_int, default=512)
    parser.add_argument("--max-display", type=positive_int, default=128)
    parser.add_argument("--threads", type=positive_int, default=1, help="CPU intra-op threads")
    parser.add_argument("--load-dir", type=Path, help="Load saved checkpoints instead of training")
    partial, _ = parser.parse_known_args(argv)
    if partial.config:
        try:
            config = json.loads(partial.config.read_text())
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if not isinstance(config, dict):
            parser.error("Config must be a JSON object")
        allowed = {a.dest for a in parser._actions} - {"help", "config"}
        unknown = set(config) - allowed
        if unknown:
            parser.error(f"Unknown config keys: {sorted(unknown)}")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    for key in ("epochs", "steps_per_epoch", "batch_size", "fm_nfe", "eval_samples", "max_display", "threads"):
        if not isinstance(getattr(args, key), int) or getattr(args, key) < 1:
            parser.error(f"{key} must be a positive integer")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be positive")
    if not args.nfe or any(not isinstance(n, int) or n < 1 for n in args.nfe):
        parser.error("nfe must contain positive integers")
    if not args.methods or any(m not in ALL_METHODS for m in args.methods):
        parser.error("Unknown or empty methods")
    if not args.datasets or any(d not in DATASET_CONFIGS for d in args.datasets):
        parser.error("Unknown or empty datasets")
    if "drift_flow_matching" in args.methods and not args.load_dir and (args.batch_size % 4 or args.batch_size < 8):
        parser.error("DFM batch size must be divisible by 4, with at least 2 samples per group")
    try:
        device = torch.device(args.device)
    except RuntimeError as error:
        parser.error(str(error))
    if device.type not in {"cpu", "cuda"}:
        parser.error("Supported devices are cpu and cuda[:index]")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable in this PyTorch environment")
    args.output = Path(args.output)
    args.load_dir = Path(args.load_dir) if args.load_dir else None
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    run = VisualizationRunConfig(
        output_dir=args.output, seed=args.seed,
        training_config=TrainingConfig(args.epochs, args.batch_size, args.steps_per_epoch, args.lr, args.device),
        integrator_config=IntegratorConfig(args.fm_nfe),
        drift_flow_config=DriftFlowMatchingConfig(), mean_flow_config=MeanFlowConfig(),
        eval_config=VisualizationEvalConfig(tuple(dict.fromkeys(args.nfe)), args.eval_samples, args.max_display),
        methods=tuple(dict.fromkeys(args.methods)), load_dir=args.load_dir,
    )
    run_visualization_experiments([DATASET_CONFIGS[key] for key in args.datasets], run)


if __name__ == "__main__":
    main()
