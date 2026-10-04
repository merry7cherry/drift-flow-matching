from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..architectures import FFHQALAE, FFHQALAEConfig
from ..architectures.ffhq_alae import load_official_alae_checkpoint
from ..utils import (
    default_runtime_roots,
    normalize_path,
    resolve_latest_alias,
    write_latest_run_marker,
)


def _default_run_name() -> str:
    return datetime.now().strftime("import_%Y%m%d_%H%M%S")


def _resolve_official_checkpoint_path(artifacts_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = artifacts_dir / candidate
    if candidate.exists():
        return candidate.resolve()
    fallback = (artifacts_dir / candidate.name).resolve()
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Unable to resolve official ALAE checkpoint: {raw_path}")


def resolve_official_alae_paths(
    *,
    alae_root: str | Path | None = None,
    official_config: str | Path | None = None,
    official_checkpoint: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    root_path = None if alae_root in (None, "") else resolve_latest_alias(normalize_path(alae_root, resolve_latest=False))
    config_path = (
        None
        if official_config in (None, "")
        else resolve_latest_alias(normalize_path(official_config, resolve_latest=False))
    )
    checkpoint_path = (
        None
        if official_checkpoint in (None, "")
        else resolve_latest_alias(normalize_path(official_checkpoint, resolve_latest=False))
    )
    if root_path is not None:
        if not root_path.exists():
            raise FileNotFoundError(f"Missing official ALAE root: {root_path}")
        artifacts_dir = (root_path / "training_artifacts" / "ffhq").resolve()
        if config_path is None:
            config_path = (root_path / "configs" / "ffhq.yaml").resolve()
    else:
        artifacts_dir = None
    if config_path is None:
        raise ValueError("Provide --alae-root or --official-config")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing official ALAE config: {config_path}")
    if artifacts_dir is None:
        artifacts_dir = (
            checkpoint_path.parent.resolve()
            if checkpoint_path is not None
            else config_path.resolve().parents[1] / "training_artifacts" / "ffhq"
        )
    if checkpoint_path is None:
        last_checkpoint = artifacts_dir / "last_checkpoint"
        if not last_checkpoint.exists():
            raise FileNotFoundError(
                "Missing official ALAE checkpoint. Provide --official-checkpoint or populate training_artifacts/ffhq/last_checkpoint"
            )
        checkpoint_path = _resolve_official_checkpoint_path(artifacts_dir, last_checkpoint.read_text(encoding="utf-8"))
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Missing official ALAE artifacts dir: {artifacts_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing official ALAE checkpoint: {checkpoint_path}")
    return config_path.resolve(), checkpoint_path.resolve(), artifacts_dir.resolve()


def _infer_official_module_root(
    *,
    alae_root: str | Path | None,
    config_path: Path,
    artifacts_dir: Path,
) -> Path | None:
    if alae_root not in (None, ""):
        return resolve_latest_alias(normalize_path(alae_root, resolve_latest=False))
    config_root = config_path.resolve().parent.parent
    artifacts_root = artifacts_dir.resolve().parent.parent
    if config_root == artifacts_root:
        return config_root
    return None


def import_official_alae(
    *,
    alae_root: str | Path | None = None,
    official_config: str | Path | None = None,
    official_checkpoint: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
) -> Path:
    runtime = default_runtime_roots()
    config_path, checkpoint_path, artifacts_dir = resolve_official_alae_paths(
        alae_root=alae_root,
        official_config=official_config,
        official_checkpoint=official_checkpoint,
    )
    with config_path.open("r", encoding="utf-8") as handle:
        official_yaml = yaml.safe_load(handle) or {}
    model_config = FFHQALAEConfig.from_official_yaml_payload(official_yaml)
    official_payload = load_official_alae_checkpoint(
        checkpoint_path,
        module_root=_infer_official_module_root(
            alae_root=alae_root,
            config_path=config_path,
            artifacts_dir=artifacts_dir,
        ),
    )
    model = FFHQALAE.from_official_checkpoint_payload(official_payload, config=model_config)
    resolved_output_dir = (
        normalize_path(output_dir, runtime_roots=runtime, resolve_latest=False)
        if output_dir is not None
        else None
    )
    run_root_path = (
        resolved_output_dir.parent
        if resolved_output_dir is not None
        else normalize_path("{runs_root}/ffhq_alae", runtime_roots=runtime, resolve_latest=False)
    )
    if resolved_output_dir is None:
        resolved_output_dir = run_root_path / (run_name or _default_run_name())
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out = resolved_output_dir / "alae_ffhq.pt"
    source = {
        "official_root": None if alae_root in (None, "") else str(resolve_latest_alias(normalize_path(alae_root))),
        "official_config": str(config_path),
        "official_checkpoint": str(checkpoint_path),
        "official_artifacts_dir": str(artifacts_dir),
    }
    model.save_project_checkpoint(checkpoint_out, source=source)
    resolved_payload = {
        "model": model_config.to_dict(),
        "source": source,
        "output_checkpoint": str(checkpoint_out),
    }
    (resolved_output_dir / "metadata.json").write_text(json.dumps(resolved_payload, indent=2), encoding="utf-8")
    with (resolved_output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_payload, handle, sort_keys=False)
    write_latest_run_marker(run_root_path, resolved_output_dir)
    return resolved_output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import an official FFHQ ALAE checkpoint into the project format.")
    parser.add_argument("--alae-root", default=None)
    parser.add_argument("--official-config", default=None)
    parser.add_argument("--official-checkpoint", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = import_official_alae(
        alae_root=args.alae_root,
        official_config=args.official_config,
        official_checkpoint=args.official_checkpoint,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
