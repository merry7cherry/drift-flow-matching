from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

from driftfm.tasks.ffhq_download_aging import (
    DEFAULT_FFHQ_AGING_HF_REPO_ID,
    HF_FFHQ_AGING_LABELS_PATH,
    build_ffhq_aging_manifest,
    extract_ffhq_aging_archives,
    ffhq_aging_image_dir_name,
    ffhq_aging_relative_image_path,
    prepare_ffhq_aging_public_source,
)
from driftfm.tasks.ffhq_split import build_ffhq_six_class_split


def _write_labels_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_number", "age_group", "age_group_confidence", "gender", "gender_confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _touch_image(image_root: Path, image_number: int, content: str = "image") -> Path:
    relative = ffhq_aging_relative_image_path(image_number)
    path = image_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_archive(path: Path, members: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as handle:
        for name, content in members.items():
            handle.writestr(name, content)
    return path


def _write_mock_hf_snapshot(root: Path) -> Path:
    _write_labels_csv(
        root / HF_FFHQ_AGING_LABELS_PATH,
        [
            {"image_number": 0, "age_group": "0-2", "age_group_confidence": 1, "gender": "male", "gender_confidence": 1},
            {"image_number": 1001, "age_group": "30-39", "age_group_confidence": 1, "gender": "female", "gender_confidence": 1},
        ],
    )
    _write_archive(
        root / "images_zip" / "images1024x1024" / "images_part_00000.zip",
        {
            "00000.png": "zero",
            "01001.png": "one",
        },
    )
    return root


def test_ffhq_aging_relative_image_path_matches_official_layout() -> None:
    assert ffhq_aging_relative_image_path(0) == "00000/00000.png"
    assert ffhq_aging_relative_image_path(17) == "00000/00017.png"
    assert ffhq_aging_relative_image_path(1000) == "01000/01000.png"
    assert ffhq_aging_image_dir_name(1024) == "ffhq_aging1024x1024"


def test_default_repo_id_is_pinned_to_hf_mirror() -> None:
    assert DEFAULT_FFHQ_AGING_HF_REPO_ID == "NUS-SRI-2025/FFHQ-Aging-Dataset"


def test_extract_ffhq_aging_archives_unzips_pngs_and_skips_existing(tmp_path: Path) -> None:
    archives_dir = tmp_path / "images_zip" / "images1024x1024"
    image_root = tmp_path / ffhq_aging_image_dir_name(1024)
    _write_archive(
        archives_dir / "images_part_00000.zip",
        {
            "images1024x1024/00000/00000.png": "zero",
            "images1024x1024/01000/01001.png": "one",
        },
    )

    first = extract_ffhq_aging_archives(archives_dir=archives_dir, image_root=image_root)
    second = extract_ffhq_aging_archives(archives_dir=archives_dir, image_root=image_root)

    assert first["archive_count"] == 1
    assert first["expected_files"] == 2
    assert first["extracted_files"] == 2
    assert first["skipped_files"] == 0
    assert second["extracted_files"] == 0
    assert second["skipped_files"] == 2
    assert (image_root / "00000/00000.png").read_text(encoding="utf-8") == "zero"
    assert (image_root / "01000/01001.png").read_text(encoding="utf-8") == "one"


def test_extract_ffhq_aging_archives_accepts_flat_png_layout(tmp_path: Path) -> None:
    archives_dir = tmp_path / "images_zip" / "images1024x1024"
    image_root = tmp_path / ffhq_aging_image_dir_name(1024)
    _write_archive(
        archives_dir / "images_part_00000.zip",
        {
            "00000.png": "zero",
            "01001.png": "one",
        },
    )

    stats = extract_ffhq_aging_archives(archives_dir=archives_dir, image_root=image_root)

    assert stats["expected_files"] == 2
    assert stats["extracted_files"] == 2
    assert (image_root / "00000/00000.png").read_text(encoding="utf-8") == "zero"
    assert (image_root / "01000/01001.png").read_text(encoding="utf-8") == "one"


def test_extract_ffhq_aging_archives_rejects_invalid_zip(tmp_path: Path) -> None:
    archives_dir = tmp_path / "images_zip" / "images1024x1024"
    archives_dir.mkdir(parents=True, exist_ok=True)
    bad_archive = archives_dir / "images_part_00000.zip"
    bad_archive.write_text("not a zip", encoding="utf-8")

    with pytest.raises(zipfile.BadZipFile, match="Invalid FFHQ-Aging archive"):
        extract_ffhq_aging_archives(archives_dir=archives_dir, image_root=tmp_path / "images")


def test_build_ffhq_aging_manifest_writes_repo_native_columns(tmp_path: Path) -> None:
    image_root = tmp_path / ffhq_aging_image_dir_name(1024)
    _touch_image(image_root, 0)
    _touch_image(image_root, 1001)
    labels_csv = _write_labels_csv(
        tmp_path / "ffhq_aging_labels.csv",
        [
            {"image_number": 0, "age_group": "0-2", "age_group_confidence": 1, "gender": "male", "gender_confidence": 1},
            {"image_number": 1001, "age_group": "30-39", "age_group_confidence": 1, "gender": "female", "gender_confidence": 1},
        ],
    )

    manifest_path = build_ffhq_aging_manifest(
        labels_csv=labels_csv,
        image_root=image_root,
        manifest_out=tmp_path / "ffhq_aging" / "ffhq_aging_manifest.csv",
    )

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((manifest_path.parent / "summary.json").read_text(encoding="utf-8"))

    assert rows == [
        {"path": "00000/00000.png", "gender": "male", "age_group": "0-2"},
        {"path": "01000/01001.png", "gender": "female", "age_group": "30-39"},
    ]
    assert summary["kept_rows"] == 2
    assert summary["missing_rows"] == 0


def test_build_ffhq_aging_manifest_rejects_missing_images(tmp_path: Path) -> None:
    image_root = tmp_path / ffhq_aging_image_dir_name(1024)
    _touch_image(image_root, 0)
    labels_csv = _write_labels_csv(
        tmp_path / "ffhq_aging_labels.csv",
        [
            {"image_number": 0, "age_group": "0-2", "age_group_confidence": 1, "gender": "male", "gender_confidence": 1},
            {"image_number": 1, "age_group": "20-29", "age_group_confidence": 1, "gender": "female", "gender_confidence": 1},
        ],
    )

    with pytest.raises(FileNotFoundError, match="Expected FFHQ-Aging image is missing"):
        build_ffhq_aging_manifest(
            labels_csv=labels_csv,
            image_root=image_root,
            manifest_out=tmp_path / "ffhq_aging" / "ffhq_aging_manifest.csv",
        )


def test_ffhq_aging_manifest_integrates_with_generic_ffhq_split_defaults(tmp_path: Path) -> None:
    image_root = tmp_path / ffhq_aging_image_dir_name(1024)
    specs = [
        (0, "male", "0-2"),
        (1, "male", "20-29"),
        (2, "male", "50-69"),
        (3, "female", "3-9"),
        (4, "female", "30-39"),
        (5, "female", "70+"),
    ]
    rows: list[dict[str, object]] = []
    for image_number, gender, age_group in specs:
        for offset in (0, 1000):
            current_number = image_number + offset
            _touch_image(image_root, current_number)
            rows.append(
                {
                    "image_number": current_number,
                    "age_group": age_group,
                    "age_group_confidence": 1,
                    "gender": gender,
                    "gender_confidence": 1,
                }
            )
    labels_csv = _write_labels_csv(tmp_path / "ffhq_aging_labels.csv", rows)
    manifest_path = build_ffhq_aging_manifest(
        labels_csv=labels_csv,
        image_root=image_root,
        manifest_out=tmp_path / "ffhq_aging" / "ffhq_aging_manifest.csv",
    )

    output_dir = build_ffhq_six_class_split(
        manifest=manifest_path,
        image_root=image_root,
        output_dir=tmp_path / "ffhq_6class",
        seed=7,
    )

    assert (output_dir / "train" / "male_children").is_dir()
    assert (output_dir / "test" / "female_old").is_dir()


def test_prepare_ffhq_aging_public_source_cleans_archives_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot_root = _write_mock_hf_snapshot(tmp_path / "snapshot")

    def fake_download(*, repo_id: str, work_dir: str | Path) -> Path:
        assert repo_id == DEFAULT_FFHQ_AGING_HF_REPO_ID
        assert Path(work_dir) == tmp_path / "work"
        copied = tmp_path / "work" / "NUS-SRI-2025__FFHQ-Aging-Dataset"
        if copied.exists():
            return copied
        shutil.copytree(snapshot_root, copied)
        return copied

    import shutil

    monkeypatch.setattr("driftfm.tasks.ffhq_download_aging.download_ffhq_aging_hf_snapshot", fake_download)

    manifest_path = prepare_ffhq_aging_public_source(
        image_dir=tmp_path / "images",
        manifest_out=tmp_path / "manifest" / "ffhq_aging_manifest.csv",
        work_dir=tmp_path / "work",
    )

    summary = json.loads((manifest_path.parent / "summary.json").read_text(encoding="utf-8"))
    assert manifest_path.exists()
    assert summary["repo_id"] == DEFAULT_FFHQ_AGING_HF_REPO_ID
    assert summary["archives_cleaned"] is True
    assert not (tmp_path / "work" / "NUS-SRI-2025__FFHQ-Aging-Dataset").exists()


def test_prepare_ffhq_aging_public_source_keeps_archives_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_root = _write_mock_hf_snapshot(tmp_path / "snapshot")

    def fake_download(*, repo_id: str, work_dir: str | Path) -> Path:
        copied = tmp_path / "work" / "custom"
        if copied.exists():
            return copied
        shutil.copytree(snapshot_root, copied)
        return copied

    import shutil

    monkeypatch.setattr("driftfm.tasks.ffhq_download_aging.download_ffhq_aging_hf_snapshot", fake_download)

    manifest_path = prepare_ffhq_aging_public_source(
        repo_id="custom/repo",
        image_dir=tmp_path / "images",
        manifest_out=tmp_path / "manifest" / "ffhq_aging_manifest.csv",
        work_dir=tmp_path / "work",
        keep_archives=True,
    )

    summary = json.loads((manifest_path.parent / "summary.json").read_text(encoding="utf-8"))
    assert summary["repo_id"] == "custom/repo"
    assert summary["archives_cleaned"] is False
    assert (tmp_path / "work" / "custom" / HF_FFHQ_AGING_LABELS_PATH).exists()
