from __future__ import annotations

import argparse

from ..evaluation.registry import evaluate_checkpoint
from ..utils import normalize_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint through the registered evaluator.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    result = evaluate_checkpoint(
        normalize_path(args.checkpoint, resolve_latest=False),
        config_path=None if args.config is None else normalize_path(args.config, resolve_latest=False),
    )
    print(result.output_path)


__all__ = ["main"]


if __name__ == "__main__":
    main()
