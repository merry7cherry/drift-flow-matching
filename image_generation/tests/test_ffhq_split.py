from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from driftfm.data.ffhq_latent import FFHQ_CLASS_NAMES
from driftfm.tasks.ffhq_split import build_ffhq_six_class_split


def _write_file(path: Path, content: str = "ffhq") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_csv_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_jsonl_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _read_split_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _direct_class_rows(image_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_name in FFHQ_CLASS_NAMES:
        rows.append(
            {
                "path": str(_write_file(image_root / "explicit" / class_name / f"{class_name}_train.png")),
                "split": "train",
                "class_name": class_name,
            }
        )
        rows.append(
            {
                "path": str(_write_file(image_root / "explicit" / class_name / f"{class_name}_val.png")),
                "split": "val",
                "class_name": class_name,
            }
        )
    return rows


def _numeric_age_rows(image_root: Path) -> list[dict[str, object]]:
    specs = [
        ("male", 10, "male_children"),
        ("male", 25, "male_adult"),
        ("male", 70, "male_old"),
        ("female", 8, "female_children"),
        ("female", 35, "female_adult"),
        ("female", 65, "female_old"),
    ]
    rows: list[dict[str, object]] = []
    for gender, age, class_name in specs:
        rows.append(
            {
                "path": str(_write_file(image_root / "numeric" / class_name / f"{class_name}_train.png")),
                "split": "train",
                "gender": gender,
                "age": age,
            }
        )
        rows.append(
            {
                "path": str(_write_file(image_root / "numeric" / class_name / f"{class_name}_test.png")),
                "split": "test",
                "gender": gender,
                "age": age,
            }
        )
    return rows


def _age_group_rows(image_root: Path) -> list[dict[str, object]]:
    specs = [
        ("boy", "10-19", "male_children"),
        ("man", "20-29", "male_adult"),
        ("male", "more than 70", "male_old"),
        ("girl", "3-9", "female_children"),
        ("woman", "40-49", "female_adult"),
        ("female", "60-69", "female_old"),
    ]
    rows: list[dict[str, object]] = []
    for gender, age_group, class_name in specs:
        rows.append(
            {
                "path": str(_write_file(image_root / "age_group" / class_name / f"{class_name}_train.png")),
                "split": "train",
                "gender": gender,
                "age_group": age_group,
            }
        )
        rows.append(
            {
                "path": str(_write_file(image_root / "age_group" / class_name / f"{class_name}_test.png")),
                "split": "test",
                "gender": gender,
                "age_group": age_group,
            }
        )
    return rows


def _rows_without_split(image_root: Path, *, samples_per_class: int = 4) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specs = [
        ("male", 10, "male_children"),
        ("male", 25, "male_adult"),
        ("male", 65, "male_old"),
        ("female", 12, "female_children"),
        ("female", 38, "female_adult"),
        ("female", 72, "female_old"),
    ]
    for gender, age, class_name in specs:
        for index in range(samples_per_class):
            rows.append(
                {
                    "path": str(_write_file(image_root / "nosplit" / class_name / f"{class_name}_{index}.png")),
                    "gender": gender,
                    "age": age,
                }
            )
    return rows


def test_ffhq_split_builds_six_class_tree_from_direct_class_manifest(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    manifest_path = _write_csv_manifest(tmp_path / "manifest.csv", _direct_class_rows(image_root))

    output_dir = build_ffhq_six_class_split(
        manifest=manifest_path,
        output_dir=tmp_path / "ffhq_6class",
        materialize_mode="symlink",
    )

    manifest_rows = _read_split_manifest(output_dir / "split_manifest.csv")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["split_strategy"] == "explicit"
    assert summary["counts"]["test"]["male_children"] == 1
    assert len(manifest_rows) == 12
    for class_name in FFHQ_CLASS_NAMES:
        train_dir = output_dir / "train" / class_name
        test_dir = output_dir / "test" / class_name
        assert train_dir.is_dir()
        assert test_dir.is_dir()
        assert len(list(train_dir.iterdir())) == 1
        assert len(list(test_dir.iterdir())) == 1
    sample_target = output_dir / "train" / "male_children" / next(iter((output_dir / "train" / "male_children").iterdir())).name
    assert sample_target.is_symlink()


def test_ffhq_split_derives_classes_from_numeric_age_and_can_copy_files(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    manifest_path = _write_csv_manifest(tmp_path / "manifest.csv", _numeric_age_rows(image_root))

    output_dir = build_ffhq_six_class_split(
        manifest=manifest_path,
        output_dir=tmp_path / "ffhq_6class",
        materialize_mode="copy",
    )

    target = next((output_dir / "train" / "female_adult").iterdir())
    assert target.is_file()
    assert not target.is_symlink()
    assert target.read_text(encoding="utf-8") == "ffhq"


def test_ffhq_split_derives_classes_from_age_group_jsonl(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    manifest_path = _write_jsonl_manifest(tmp_path / "manifest.jsonl", _age_group_rows(image_root))

    output_dir = build_ffhq_six_class_split(
        manifest=manifest_path,
        output_dir=tmp_path / "ffhq_6class",
    )
    manifest_rows = _read_split_manifest(output_dir / "split_manifest.csv")

    classes = {row["class_name"] for row in manifest_rows}
    assert classes == set(FFHQ_CLASS_NAMES)
    assert any(row["age_group"] == "old" and row["class_name"] == "male_old" for row in manifest_rows)


def test_ffhq_split_stratifies_deterministically_when_split_is_missing(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    manifest_path = _write_csv_manifest(tmp_path / "manifest.csv", _rows_without_split(image_root))

    first = build_ffhq_six_class_split(
        manifest=manifest_path,
        output_dir=tmp_path / "first",
        seed=7,
    )
    second = build_ffhq_six_class_split(
        manifest=manifest_path,
        output_dir=tmp_path / "second",
        seed=7,
    )

    first_rows = {
        (row["source_path"], row["class_name"]): row["split"]
        for row in _read_split_manifest(first / "split_manifest.csv")
    }
    second_rows = {
        (row["source_path"], row["class_name"]): row["split"]
        for row in _read_split_manifest(second / "split_manifest.csv")
    }

    assert first_rows == second_rows
    for class_name in FFHQ_CLASS_NAMES:
        assert (first / "train" / class_name).is_dir()
        assert (first / "test" / class_name).is_dir()


def test_ffhq_split_rejects_classes_missing_in_one_split(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    rows = _numeric_age_rows(image_root)
    rows = [row for row in rows if not (row["gender"] == "male" and row["age"] == 10 and row["split"] == "test")]
    rows.append(
        {
            "path": str(_write_file(image_root / "numeric" / "male_children" / "male_children_train_extra.png")),
            "split": "train",
            "gender": "male",
            "age": 10,
        }
    )
    manifest_path = _write_csv_manifest(tmp_path / "manifest.csv", rows)

    with pytest.raises(ValueError, match="must be non-empty in both train and test splits"):
        build_ffhq_six_class_split(manifest=manifest_path, output_dir=tmp_path / "ffhq_6class")


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ({"path": "missing.png", "class_name": "unknown_class", "split": "train"}, "unsupported 'class_name'"),
        ({"path": "missing.png", "gender": "robot", "age": 22, "split": "train"}, "unsupported 'gender'"),
        ({"path": "missing.png", "gender": "female", "age_group": "ancient", "split": "train"}, "unsupported 'age_group'"),
    ],
)
def test_ffhq_split_rejects_invalid_metadata(tmp_path: Path, row: dict[str, object], match: str) -> None:
    bad_row = dict(row)
    bad_row["path"] = str(_write_file(tmp_path / "images" / "invalid" / "bad.png"))
    rows = [bad_row]
    rows.extend(
        {
            "path": str(_write_file(tmp_path / "images" / class_name / f"{class_name}_{split}.png")),
            "class_name": class_name,
            "split": split,
        }
        for class_name in FFHQ_CLASS_NAMES
        for split in ("train", "test")
    )
    manifest_path = _write_csv_manifest(tmp_path / "manifest.csv", rows)

    with pytest.raises(ValueError, match=match):
        build_ffhq_six_class_split(manifest=manifest_path, output_dir=tmp_path / "ffhq_6class")


def test_ffhq_split_rejects_missing_source_files(tmp_path: Path) -> None:
    rows = [
        {"path": "does_not_exist.png", "class_name": "male_children", "split": "train"},
        {"path": "does_not_exist_2.png", "class_name": "male_children", "split": "test"},
    ]
    rows.extend(
        {
            "path": str(_write_file(tmp_path / "images" / class_name / f"{class_name}_{split}.png")),
            "class_name": class_name,
            "split": split,
        }
        for class_name in FFHQ_CLASS_NAMES[1:]
        for split in ("train", "test")
    )
    manifest_path = _write_csv_manifest(tmp_path / "manifest.csv", rows)

    with pytest.raises(FileNotFoundError, match="missing source file"):
        build_ffhq_six_class_split(manifest=manifest_path, output_dir=tmp_path / "ffhq_6class")


def test_ffhq_split_disambiguates_duplicate_basenames_with_hash_suffix(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    rows = _direct_class_rows(image_root)
    rows.append(
        {
            "path": str(_write_file(image_root / "alt" / "same_name.png")),
            "split": "train",
            "class_name": "male_children",
        }
    )
    rows.append(
        {
            "path": str(_write_file(image_root / "alt2" / "same_name.png")),
            "split": "train",
            "class_name": "male_children",
        }
    )
    manifest_path = _write_csv_manifest(tmp_path / "manifest.csv", rows)

    output_dir = build_ffhq_six_class_split(manifest=manifest_path, output_dir=tmp_path / "ffhq_6class")
    filenames = sorted(path.name for path in (output_dir / "train" / "male_children").iterdir())

    assert len(filenames) == 3
    assert len(set(filenames)) == 3
