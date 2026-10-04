from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TASK_LAUNCHERS = [
    ("driftfm.tasks.mnist_ae", "driftfm-mnist-train-ae", "driftfm.tasks.mnist_ae:main"),
    ("driftfm.tasks.mnist_encode", "driftfm-mnist-encode-latents", "driftfm.tasks.mnist_encode:main"),
    ("driftfm.tasks.ffhq_download_aging", "driftfm-ffhq-download-aging", "driftfm.tasks.ffhq_download_aging:main"),
    ("driftfm.tasks.ffhq_split", "driftfm-ffhq-build-split", "driftfm.tasks.ffhq_split:main"),
    ("driftfm.tasks.ffhq_import_alae", "driftfm-ffhq-import-alae", "driftfm.tasks.ffhq_import_alae:main"),
    ("driftfm.tasks.ffhq_encode", "driftfm-ffhq-encode-latents", "driftfm.tasks.ffhq_encode:main"),
    ("driftfm.tasks.train", "driftfm-train", "driftfm.tasks.train:main"),
    ("driftfm.tasks.evaluate", "driftfm-evaluate", "driftfm.tasks.evaluate:main"),
    ("driftfm.tasks.evaluate_mnist", "driftfm-eval-mnist", "driftfm.tasks.evaluate_mnist:main"),
    ("driftfm.tasks.evaluate_ffhq", "driftfm-eval-ffhq", "driftfm.tasks.evaluate_ffhq:main"),
]


def _task_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_ROOT) if not existing else f"{SRC_ROOT}{os.pathsep}{existing}"
    return env


def _run_console_entrypoint(target: str, script_name: str) -> subprocess.CompletedProcess[str]:
    module_name, function_name = target.split(":")
    code = (
        f"import sys; from {module_name} import {function_name} as _entry; "
        "sys.argv = [sys.argv[1], *sys.argv[2:]]; _entry()"
    )
    return subprocess.run(
        [sys.executable, "-c", code, script_name, "--help"],
        cwd=REPO_ROOT,
        env=_task_env(),
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("module_name", "script_name", "target", "expected_fragments"),
    [
        (*TASK_LAUNCHERS[0], ("--latent-dim", "--save-every", "--no-deterministic")),
        (*TASK_LAUNCHERS[1], ("--ae-ckpt", "--output-dir")),
        (*TASK_LAUNCHERS[2], ("--repo-id", "--work-dir")),
        (*TASK_LAUNCHERS[3], ("--manifest", "--materialize-mode")),
        (*TASK_LAUNCHERS[4], ("--alae-root", "--official-checkpoint")),
        (*TASK_LAUNCHERS[5], ("--alae-ckpt", "--image-root")),
        (*TASK_LAUNCHERS[6], ("--config", "--resume-from")),
        (*TASK_LAUNCHERS[7], ("--checkpoint", "--config")),
        (*TASK_LAUNCHERS[8], ("--ae-ckpt", "--metrics", "--seed", "--no-deterministic")),
        (
            *TASK_LAUNCHERS[9],
            ("--real-npz", "--real-split", "--alae-ckpt", "--metrics", "--seed", "--fid-decode-noise", "--no-deterministic"),
        ),
    ],
)
def test_task_modules_expose_executable_help(
    module_name: str,
    script_name: str,
    target: str,
    expected_fragments: tuple[str, ...],
) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=REPO_ROOT,
        env=_task_env(),
        capture_output=True,
        text=True,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "usage:" in combined.lower(), combined
    for fragment in expected_fragments:
        assert fragment in combined, combined


@pytest.mark.parametrize(("module_name", "script_name", "target"), TASK_LAUNCHERS)
def test_console_script_equivalents_expose_executable_help(
    module_name: str,
    script_name: str,
    target: str,
) -> None:
    result = _run_console_entrypoint(target, script_name)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "usage:" in combined.lower(), combined


def test_ffhq_eval_help_removes_legacy_alae_root() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "driftfm.tasks.evaluate_ffhq", "--help"],
        cwd=REPO_ROOT,
        env=_task_env(),
        capture_output=True,
        text=True,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "--alae-ckpt" in combined, combined
    assert "--alae-root" not in combined, combined
