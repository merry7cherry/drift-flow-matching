from __future__ import annotations


def main() -> None:
    from ..evaluation.ffhq import main as _main

    _main()

__all__ = ["main"]


if __name__ == "__main__":
    main()
