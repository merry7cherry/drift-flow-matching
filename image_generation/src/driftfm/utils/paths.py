from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

DEFAULT_PROJECT_NAME = "drift-flow-matching"
LATEST_RUN_FILENAME = "LATEST_RUN"
RUNTIME_ENV_VARS = {
    "project_name": "DRIFTFM_PROJECT_NAME",
    "data_root": "DRIFTFM_DATA_ROOT",
    "runs_root": "DRIFTFM_RUNS_ROOT",
    "mnist_root": "DRIFTFM_MNIST_ROOT",
}
RUNTIME_ROOT_NAMES = tuple(RUNTIME_ENV_VARS)
_PATH_SUFFIXES = ("_path", "_dir", "_root", "_ckpt", "_npz")
_EXPLICIT_PATH_KEYS = {
    "ae_ckpt",
    "checkpoint",
    "classifier_ckpt",
    "config",
    "output_dir",
    "real_npz",
    "root_dir",
}
_RUNTIME_PLACEHOLDER = re.compile(
    r"\{(project_name|data_root|runs_root|mnist_root)\}"
)


def _expand_runtime_value(value: str | Path) -> str:
    return os.path.expanduser(os.path.expandvars(str(value)))


def _coerce_runtime_mapping(runtime: Any | None) -> dict[str, str | None]:
    if runtime is None:
        return {}
    if isinstance(runtime, dict):
        return {key: runtime.get(key) for key in RUNTIME_ROOT_NAMES}
    return {key: getattr(runtime, key, None) for key in RUNTIME_ROOT_NAMES}


def _default_base_dir(source_path: str | Path | None = None) -> Path:
    if source_path is not None:
        path = Path(source_path).resolve()
        return path if path.is_dir() else path.parent
    return Path.cwd().resolve()


def _is_path_key(key: str) -> bool:
    return key in _EXPLICIT_PATH_KEYS or key.endswith(_PATH_SUFFIXES)


def _interpolate_runtime_placeholders(text: str, runtime_roots: dict[str, str | Path | None]) -> str:
    def replace(match: re.Match[str]) -> str:
        value = runtime_roots.get(match.group(1))
        if value in (None, ""):
            raise KeyError(f"Unknown runtime placeholder: {match.group(0)}")
        return str(value)

    return _RUNTIME_PLACEHOLDER.sub(replace, text)


def normalize_path(
    value: str | Path,
    *,
    runtime_roots: Any | None = None,
    base_dir: str | Path | None = None,
    resolve_latest: bool = False,
) -> Path:
    raw = str(value)
    if raw.strip() == "":
        return Path(raw)

    expanded = _expand_runtime_value(raw)
    runtime_mapping = _coerce_runtime_mapping(runtime_roots)
    if "{" in expanded and "}" in expanded:
        expanded = _interpolate_runtime_placeholders(expanded, runtime_mapping)

    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = _default_base_dir(base_dir) / candidate
        candidate = candidate.resolve()
    else:
        candidate = Path(os.path.normpath(str(candidate)))
    return resolve_latest_alias(candidate) if resolve_latest else candidate


def default_runtime_roots(
    source_path: str | Path | None = None,
    *,
    project_name: str | None = None,
) -> dict[str, str]:
    return resolve_runtime_config({"project_name": project_name}, source_path=source_path)


def resolve_runtime_config(
    runtime: Any | None,
    *,
    source_path: str | Path | None = None,
) -> dict[str, str]:
    """Resolve explicit config roots, then environment overrides, then local defaults.

    Relative YAML roots are anchored to the YAML file, while CLI paths and
    defaults without a source file are anchored to the current directory.
    Absolute roots saved in a checkpoint therefore stay portable and explicit.
    """
    if isinstance(runtime, dict):
        unknown = set(runtime) - set(RUNTIME_ROOT_NAMES)
        if unknown:
            raise ValueError(f"Unknown runtime fields: {sorted(unknown)}")
    raw = _coerce_runtime_mapping(runtime)
    base_dir = _default_base_dir(source_path)
    resolved = {"project_name": str(raw.get("project_name") or DEFAULT_PROJECT_NAME)}
    defaults = {"data_root": "data", "runs_root": "runs", "mnist_root": "{data_root}/mnist"}
    for name, default in defaults.items():
        value = raw.get(name) or os.getenv(RUNTIME_ENV_VARS[name]) or default
        resolved[name] = str(normalize_path(value, runtime_roots=resolved, base_dir=base_dir))
    return resolved


def normalize_path_fields(
    payload: Any,
    *,
    runtime_roots: Any | None = None,
    base_dir: str | Path | None = None,
) -> Any:
    if isinstance(payload, dict):
        normalized: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "runtime":
                normalized[key] = value
                continue
            if isinstance(value, (str, Path)) and _is_path_key(key):
                if isinstance(value, str) and value.strip() == "":
                    normalized[key] = value
                else:
                    normalized[key] = str(
                        normalize_path(
                            value,
                            runtime_roots=runtime_roots,
                            base_dir=base_dir,
                            resolve_latest=True,
                        )
                    )
                continue
            normalized[key] = normalize_path_fields(
                value,
                runtime_roots=runtime_roots,
                base_dir=base_dir,
            )
        return normalized
    if isinstance(payload, list):
        return [normalize_path_fields(item, runtime_roots=runtime_roots, base_dir=base_dir) for item in payload]
    return payload


def write_latest_run_marker(run_root: str | Path, run_dir: str | Path) -> None:
    root_path = Path(run_root)
    directory = Path(run_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    marker_path = root_path / LATEST_RUN_FILENAME
    marker_path.write_text(f"{directory.name}\n", encoding="utf-8")


def resolve_latest_alias(path: str | Path) -> Path:
    raw = str(path)
    if raw.strip() == "":
        return Path(raw)
    candidate = Path(_expand_runtime_value(raw))
    if candidate.exists():
        return candidate

    parts = list(candidate.parts)
    for index, part in enumerate(parts):
        if part != "latest":
            continue
        root = Path(*parts[:index]) if index > 0 else Path(".")
        marker_path = root / LATEST_RUN_FILENAME
        if not marker_path.exists():
            continue
        latest_name = marker_path.read_text(encoding="utf-8").strip()
        if not latest_name:
            continue
        return Path(*parts[:index], latest_name, *parts[index + 1 :])
    return candidate
