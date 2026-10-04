from .checkpoints import load_checkpoint, save_checkpoint
from .ema import EMA
from ..utils import resolve_latest_alias, write_latest_run_marker

__all__ = [
    "EMA",
    "load_checkpoint",
    "resolve_latest_alias",
    "save_checkpoint",
    "train_from_config",
    "write_latest_run_marker",
]


def __getattr__(name: str):
    if name == "train_from_config":
        from .trainer import train_from_config as _train_from_config

        return _train_from_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
