from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from driftfm.architectures import FFHQALAE, FFHQALAEConfig
from driftfm.architectures.ffhq_alae import load_official_alae_checkpoint
from driftfm.data.ffhq_latent import FFHQ_CLASS_NAMES
from driftfm.tasks.ffhq_encode import encode_ffhq_image_root
from driftfm.tasks.ffhq_import_alae import import_official_alae
from driftfm.utils import write_latest_run_marker


def _tiny_alae_config() -> FFHQALAEConfig:
    return FFHQALAEConfig(
        start_channel_count=2,
        max_channel_count=16,
        layer_count=2,
        latent_size=8,
        mapping_layers=2,
        dlatent_avg_beta=0.995,
        channels=3,
    )


def _write_tiny_official_alae_root(root: Path) -> tuple[Path, FFHQALAE]:
    config = _tiny_alae_config()
    model = FFHQALAE(config=config)
    official_root = root / "alae_official"
    config_dir = official_root / "configs"
    artifacts_dir = official_root / "training_artifacts" / "ffhq"
    config_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    with (config_dir / "ffhq.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "NAME": "ffhq",
                "MODEL": {
                    "START_CHANNEL_COUNT": config.start_channel_count,
                    "MAX_CHANNEL_COUNT": config.max_channel_count,
                    "LAYER_COUNT": config.layer_count,
                    "LATENT_SPACE_SIZE": config.latent_size,
                    "DLATENT_AVG_BETA": config.dlatent_avg_beta,
                    "MAPPING_LAYERS": config.mapping_layers,
                    "CHANNELS": config.channels,
                },
                "OUTPUT_DIR": "training_artifacts/ffhq",
            },
            handle,
            sort_keys=False,
        )
    checkpoint_path = artifacts_dir / "official_alae.pth"
    torch.save(
        {
            "models": {
                "generator_s": model.decoder.state_dict(),
                "discriminator_s": model.encoder.state_dict(),
                "mapping_tl_s": model.mapping_d.state_dict(),
                "mapping_fl_s": model.mapping_f.state_dict(),
                "dlatent_avg": model.dlatent_avg.state_dict(),
            }
        },
        checkpoint_path,
    )
    (artifacts_dir / "last_checkpoint").write_text("official_alae.pth\n", encoding="utf-8")
    return official_root, model


def _write_ffhq_image_tree(root: Path, image_size: int) -> Path:
    image_root = root / "ffhq_images"
    for split_idx, split in enumerate(("train", "test")):
        for class_idx, class_name in enumerate(FFHQ_CLASS_NAMES):
            class_dir = image_root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            image[..., 0] = (class_idx * 20 + split_idx * 10) % 255
            image[..., 1] = (class_idx * 30 + 5) % 255
            image[..., 2] = (class_idx * 40 + 10) % 255
            Image.fromarray(image, mode="RGB").save(class_dir / "sample.png")
    return image_root


def test_official_alae_import_generates_project_checkpoint(tmp_path: Path) -> None:
    official_root, _ = _write_tiny_official_alae_root(tmp_path)

    output_dir = import_official_alae(alae_root=official_root, output_dir=tmp_path / "imported")
    checkpoint_path = output_dir / "alae_ffhq.pt"
    model = FFHQALAE.load_project_checkpoint(checkpoint_path)

    sample = torch.randn(2, model.config.channels, model.image_size, model.image_size)
    encoded = model.encode(sample)
    decoded = model.decode(encoded)

    assert checkpoint_path.exists()
    assert encoded.shape == (2, model.config.latent_size)
    assert decoded.shape == (2, model.config.channels, model.image_size, model.image_size)
    assert (output_dir / "resolved_config.yaml").exists()
    assert (output_dir / "metadata.json").exists()


def test_load_project_checkpoint_resolves_latest_alias(tmp_path: Path) -> None:
    run_root = tmp_path / "ffhq_alae"
    run_dir = run_root / "import_20260412_172406"
    checkpoint_path = run_dir / "alae_ffhq.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    FFHQALAE(config=_tiny_alae_config()).save_project_checkpoint(checkpoint_path)
    write_latest_run_marker(run_root, run_dir)

    model = FFHQALAE.load_project_checkpoint(run_root / "latest" / "alae_ffhq.pt")

    assert model.config.latent_size == _tiny_alae_config().latent_size


def test_load_official_alae_checkpoint_supports_external_module_root(tmp_path: Path) -> None:
    module_root = tmp_path / "alae_official"
    module_root.mkdir()
    (module_root / "tracker.py").write_text(
        "class Marker:\n"
        "    def __init__(self, value):\n"
        "        self.value = value\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(module_root))
    try:
        importlib.invalidate_caches()
        tracker = importlib.import_module("tracker")
        checkpoint_path = module_root / "official_alae.pth"
        torch.save({"marker": tracker.Marker(7)}, checkpoint_path)
    finally:
        sys.path.remove(str(module_root))
        sys.modules.pop("tracker", None)
        importlib.invalidate_caches()

    with pytest.raises(ModuleNotFoundError):
        load_official_alae_checkpoint(checkpoint_path)

    payload = load_official_alae_checkpoint(checkpoint_path, module_root=module_root)

    assert payload["marker"].value == 7
    assert payload["marker"].__class__.__module__ == "tracker"


def test_ffhq_encode_latents_writes_train_and_test_npz(tmp_path: Path) -> None:
    project_ckpt = tmp_path / "alae_ffhq.pt"
    model = FFHQALAE(config=_tiny_alae_config())
    model.save_project_checkpoint(project_ckpt)
    image_root = _write_ffhq_image_tree(tmp_path, model.image_size)

    output_dir = encode_ffhq_image_root(
        alae_ckpt=project_ckpt,
        image_root=image_root,
        device="cpu",
        batch_size=2,
        output_dir=tmp_path / "latents",
    )

    train_npz = np.load(output_dir / "train_latents_by_class.npz")
    test_npz = np.load(output_dir / "test_latents_by_class.npz")
    assert set(train_npz.files) == set(FFHQ_CLASS_NAMES)
    assert set(test_npz.files) == set(FFHQ_CLASS_NAMES)
    for class_name in FFHQ_CLASS_NAMES:
        assert train_npz[class_name].shape == (1, model.config.latent_size)
        assert test_npz[class_name].shape == (1, model.config.latent_size)


def test_ffhq_encode_latents_rejects_missing_class_dirs(tmp_path: Path) -> None:
    project_ckpt = tmp_path / "alae_ffhq.pt"
    model = FFHQALAE(config=_tiny_alae_config())
    model.save_project_checkpoint(project_ckpt)
    image_root = tmp_path / "ffhq_images"
    for split in ("train", "test"):
        for class_name in FFHQ_CLASS_NAMES[:-1]:
            class_dir = image_root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            image = np.zeros((model.image_size, model.image_size, 3), dtype=np.uint8)
            Image.fromarray(image, mode="RGB").save(class_dir / "sample.png")

    with pytest.raises(ValueError, match="Invalid FFHQ class directory layout"):
        encode_ffhq_image_root(
            alae_ckpt=project_ckpt,
            image_root=image_root,
            device="cpu",
            output_dir=tmp_path / "latents",
        )
