"""Read v0.1 class-conditional configs/checkpoints without changing weights."""
from __future__ import annotations

from typing import Any


def normalize_method_params(params: dict[str, Any]) -> dict[str, Any]:
    params = dict(params)
    legacy_form = params.pop("drift_form", "split_v0")
    if legacy_form != "split_v0":
        raise ValueError(f"Unsupported legacy drift_form: {legacy_form!r}; expected split_v0")
    return params


def normalize_architecture_params(params: dict[str, Any]) -> dict[str, Any]:
    params = dict(params)
    legacy_mode = params.pop("conditioning_mode", "class")
    if legacy_mode not in (None, "class"):
        raise ValueError(f"Unsupported legacy conditioning_mode: {legacy_mode!r}")
    for old, current in (("input_dim", "data_dim"), ("class_emb_dim", "class_embedding_dim")):
        if old in params:
            value = params.pop(old)
            if current in params and params[current] != value:
                raise ValueError(f"Conflicting {old} and {current} values")
            params[current] = value
    return params
