from __future__ import annotations

from typing import Any

from ..config import DriftFlowMatchingConfig, MethodSection
from .base import GenerativeMethod
from .drift_flow_matching import DriftFlowMatchingMethod


def _build_drift_flow_matching(params: dict[str, Any]) -> GenerativeMethod:
    config = DriftFlowMatchingConfig(**params)
    return DriftFlowMatchingMethod(config=config)


METHODS: dict[str, callable] = {
    "drift_flow_matching": _build_drift_flow_matching,
}


def create_method(section: MethodSection) -> GenerativeMethod:
    try:
        factory = METHODS[section.name]
    except KeyError as exc:
        raise KeyError(f"Unknown method: {section.name}") from exc
    return factory(dict(section.params))
