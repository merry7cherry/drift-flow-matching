from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

_DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_SUPPORTED_CUBLAS_WORKSPACE_CONFIGS = {_DEFAULT_CUBLAS_WORKSPACE_CONFIG, ":16:8"}


def _normalize_device_type(device: str | torch.device | None) -> str | None:
    if device is None:
        return None
    return torch.device(device).type


def _ensure_cublas_workspace_config(device: str | torch.device | None) -> None:
    device_type = _normalize_device_type(device)
    if device_type not in (None, "cuda"):
        return

    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current in _SUPPORTED_CUBLAS_WORKSPACE_CONFIGS:
        return
    if current not in (None, ""):
        raise RuntimeError(
            "Deterministic CUDA execution requires CUBLAS_WORKSPACE_CONFIG to be one of "
            f"{sorted(_SUPPORTED_CUBLAS_WORKSPACE_CONFIGS)!r}, got {current!r}."
        )
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "Deterministic CUDA execution must set CUBLAS_WORKSPACE_CONFIG before CUDA initialization."
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _DEFAULT_CUBLAS_WORKSPACE_CONFIG


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(
    seed: int,
    *,
    deterministic: bool,
    device: str | torch.device | None = None,
) -> None:
    if deterministic:
        _ensure_cublas_workspace_config(device)

    seed_everything(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = not deterministic
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def make_torch_generator(device: torch.device, *, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def make_data_loader_generator(*, seed: int) -> torch.Generator:
    return make_torch_generator(torch.device("cpu"), seed=seed)


def seed_data_loader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def collect_determinism_metadata(
    seed: int,
    *,
    deterministic: bool,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    cudnn_allow_tf32 = getattr(torch.backends.cudnn, "allow_tf32", None)
    cuda_matmul_allow_tf32 = None
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        cuda_matmul_allow_tf32 = getattr(torch.backends.cuda.matmul, "allow_tf32", None)
    torch_deterministic_algorithms = None
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        torch_deterministic_algorithms = bool(torch.are_deterministic_algorithms_enabled())

    normalized_device = None if device is None else str(torch.device(device))
    return {
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "device": normalized_device,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms": torch_deterministic_algorithms,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": None if cudnn_allow_tf32 is None else bool(cudnn_allow_tf32),
        "cuda_matmul_allow_tf32": None if cuda_matmul_allow_tf32 is None else bool(cuda_matmul_allow_tf32),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def stable_hash(value: Any, *, length: int = 10) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def build_seed_hash_run_name(
    *,
    seed: int,
    payload: Any,
    prefix: str | None = None,
) -> str:
    parts = [f"seed{int(seed)}"]
    if prefix not in (None, ""):
        parts.append(str(prefix))
    parts.append(stable_hash(payload))
    return "_".join(parts)


def load_json_file(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    with candidate.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_file(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
