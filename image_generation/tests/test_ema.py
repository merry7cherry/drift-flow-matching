from __future__ import annotations

import pytest
import torch

ema_mod = pytest.importorskip("driftfm.training.ema")


def _ema_class():
    ema_cls = getattr(ema_mod, "EMA", None) or getattr(ema_mod, "ExponentialMovingAverage", None)
    if ema_cls is None:
        pytest.skip("EMA helper is not implemented yet")
    return ema_cls


def test_ema_update_and_copy_to_round_trip():
    EMA = _ema_class()

    model = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)

    shadow = getattr(ema, "shadow", None)
    if shadow is not None:
        shadow_weight = next(iter(shadow.values())) if isinstance(shadow, dict) else None
        if shadow_weight is not None:
            assert torch.allclose(shadow_weight, torch.full_like(shadow_weight, 2.0))

    with torch.no_grad():
        model.weight.fill_(0.0)
    ema.copy_to(model)
    assert torch.allclose(model.weight, torch.full_like(model.weight, 2.0))
