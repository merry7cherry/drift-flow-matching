from __future__ import annotations

import argparse

from ..training.trainer import train_from_config_path
from ..utils import normalize_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a generative model from a YAML experiment config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    run_dir = train_from_config_path(
        normalize_path(args.config, resolve_latest=False),
        resume_from=None if args.resume_from is None else args.resume_from,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
