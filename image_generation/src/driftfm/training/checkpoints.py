from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any

import torch

from ..utils import normalize_path, resolve_latest_alias


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    checkpoint_path = normalize_path(path, resolve_latest=False)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, checkpoint_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = resolve_latest_alias(normalize_path(path, resolve_latest=False))
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


_EPOCH_CHECKPOINT_RE = re.compile(r"^checkpoint_epoch(\d+)\.pt$")


def checkpoint_epoch(path: str | Path) -> int | None:
    candidate = Path(path)
    if candidate.name == "checkpoint_final.pt":
        try:
            payload = load_checkpoint(candidate, map_location="cpu")
        except Exception:
            return None
        epoch_value = payload.get("epoch")
        if epoch_value is None:
            return None
        try:
            return int(epoch_value)
        except (TypeError, ValueError):
            return None

    match = _EPOCH_CHECKPOINT_RE.match(candidate.name)
    if match is None:
        return None
    try:
        payload = load_checkpoint(candidate, map_location="cpu")
    except Exception:
        return None
    epoch_value = payload.get("epoch")
    if epoch_value is None:
        return int(match.group(1))
    try:
        return int(epoch_value)
    except (TypeError, ValueError):
        return None


def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    root = normalize_path(run_dir, resolve_latest=False)
    if not root.exists():
        return None

    best_path: Path | None = None
    best_key = (-1, -1)
    for candidate in root.iterdir():
        if not candidate.is_file() or candidate.suffix != ".pt":
            continue
        epoch_value = checkpoint_epoch(candidate)
        if epoch_value is None:
            continue
        candidate_key = (epoch_value, 1 if candidate.name == "checkpoint_final.pt" else 0)
        if candidate_key < best_key:
            continue
        best_key = candidate_key
        best_path = candidate
    return best_path
