from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_ROOT) if not existing else f"{SRC_ROOT}{os.pathsep}{existing}"
    return env


@pytest.mark.parametrize(
    ("label", "snippet"),
    [
        (
            "training_then_evaluation",
            "\n".join(
                [
                    "import driftfm.training.trainer",
                    "import driftfm.evaluation",
                    "from driftfm.evaluation import evaluate_checkpoint",
                    "print('ok')",
                ]
            ),
        ),
        (
            "evaluation_then_training",
            "\n".join(
                [
                    "import driftfm.evaluation",
                    "from driftfm.training import train_from_config",
                    "print('ok')",
                ]
            ),
        ),
    ],
)
def test_training_evaluation_import_boundary_is_stable(label: str, snippet: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=REPO_ROOT,
        env=_python_env(),
        capture_output=True,
        text=True,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, f"{label} failed:\n{combined}"
    assert "ok" in result.stdout
