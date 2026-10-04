from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..data.ffhq_latent import FFHQ_CLASS_NAMES
from ..utils import default_runtime_roots, normalize_path, resolve_latest_alias

_VALID_MATERIALIZE_MODES = ("symlink", "hardlink", "copy")
_VALID_SPLITS = {"train", "test"}
_SPLIT_ALIASES = {
    "train": "train",
    "test": "test",
    "val": "test",
    "valid": "test",
    "validation": "test",
}
_CLASS_ALIASES = {
    **{name: name for name in FFHQ_CLASS_NAMES},
    "male_child": "male_children",
    "female_child": "female_children",
}
_GENDER_ALIASES = {
    "m": "male",
    "male": "male",
    "man": "male",
    "boy": "male",
    "f": "female",
    "female": "female",
    "woman": "female",
    "girl": "female",
}
_AGE_GROUP_ALIASES = {
    "child": "children",
    "children": "children",
    "kid": "children",
    "kids": "children",
    "teen": "children",
    "teens": "children",
    "teenager": "children",
    "teenagers": "children",
    "adolescent": "children",
    "adolescents": "children",
    "adult": "adult",
    "adults": "adult",
    "old": "old",
    "older": "old",
    "elderly": "old",
    "senior": "old",
    "seniors": "old",
}


@dataclass(slots=True)
class _PreparedRecord:
    source_path: Path
    class_name: str
    split: str | None
    gender: str
    age_value: float | None
    age_group: str


def _normalized_token(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _read_manifest_rows(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Manifest line {line_no} must be a JSON object")
                rows.append(payload)
        return rows
    raise ValueError(f"Unsupported manifest format: {path.suffix!r}. Expected .csv or .jsonl")


def _require_field(row: dict[str, object], field: str, row_idx: int) -> object:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Manifest row {row_idx} is missing required field {field!r}")
    return value


def _resolve_source_path(
    raw_value: object,
    *,
    image_root: Path | None,
    manifest_path: Path,
    row_idx: int,
) -> Path:
    raw = str(raw_value).strip()
    candidate = Path(os.path.expanduser(os.path.expandvars(raw)))
    if not candidate.is_absolute():
        base = image_root if image_root is not None else manifest_path.parent
        candidate = (base / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Manifest row {row_idx} references missing source file: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"Manifest row {row_idx} path must reference a file: {candidate}")
    return candidate


def _normalize_class_name(value: object, *, row_idx: int, field_name: str) -> str:
    token = _normalized_token(value)
    class_name = _CLASS_ALIASES.get(token)
    if class_name is None:
        raise ValueError(f"Manifest row {row_idx} has unsupported {field_name!r}: {value!r}")
    return class_name


def _normalize_gender(value: object, *, row_idx: int, field_name: str) -> str:
    token = _normalized_token(value)
    gender = _GENDER_ALIASES.get(token)
    if gender is None:
        raise ValueError(f"Manifest row {row_idx} has unsupported {field_name!r}: {value!r}")
    return gender


def _parse_age_value(value: object, *, row_idx: int, field_name: str) -> float:
    try:
        return float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Manifest row {row_idx} has non-numeric {field_name!r}: {value!r}") from exc


def _age_group_from_numeric(age_value: float, *, child_max_age: int, old_min_age: int) -> str:
    if age_value <= child_max_age:
        return "children"
    if age_value >= old_min_age:
        return "old"
    return "adult"


def _normalize_age_group(
    value: object,
    *,
    row_idx: int,
    field_name: str,
    child_max_age: int,
    old_min_age: int,
) -> str:
    token = _normalized_token(value)
    if token in _AGE_GROUP_ALIASES:
        return _AGE_GROUP_ALIASES[token]

    numbers = [float(part) for part in re.findall(r"\d+(?:\.\d+)?", str(value))]
    if numbers:
        if len(numbers) == 1:
            return _age_group_from_numeric(numbers[0], child_max_age=child_max_age, old_min_age=old_min_age)
        lower = min(numbers)
        upper = max(numbers)
        if upper <= child_max_age:
            return "children"
        if lower >= old_min_age:
            return "old"
        return "adult"
    raise ValueError(f"Manifest row {row_idx} has unsupported {field_name!r}: {value!r}")


def _derive_class_name(
    row: dict[str, object],
    *,
    row_idx: int,
    class_column: str,
    gender_column: str,
    age_column: str,
    age_group_column: str,
    child_max_age: int,
    old_min_age: int,
) -> tuple[str, str, float | None, str]:
    class_value = row.get(class_column)
    if class_value is not None and str(class_value).strip() != "":
        class_name = _normalize_class_name(class_value, row_idx=row_idx, field_name=class_column)
        gender = "male" if class_name.startswith("male_") else "female"
        age_group = class_name.split("_", maxsplit=1)[1]
        age_value = None
        return class_name, gender, age_value, age_group

    gender = _normalize_gender(_require_field(row, gender_column, row_idx), row_idx=row_idx, field_name=gender_column)
    age_group_value = row.get(age_group_column)
    age_value_raw = row.get(age_column)
    if age_group_value is not None and str(age_group_value).strip() != "":
        age_group = _normalize_age_group(
            age_group_value,
            row_idx=row_idx,
            field_name=age_group_column,
            child_max_age=child_max_age,
            old_min_age=old_min_age,
        )
        age_value = None
    elif age_value_raw is not None and str(age_value_raw).strip() != "":
        age_value = _parse_age_value(age_value_raw, row_idx=row_idx, field_name=age_column)
        age_group = _age_group_from_numeric(age_value, child_max_age=child_max_age, old_min_age=old_min_age)
    else:
        raise ValueError(
            f"Manifest row {row_idx} requires either {class_column!r} or "
            f"{gender_column!r} with {age_group_column!r}/{age_column!r}"
        )

    return f"{gender}_{age_group}", gender, age_value, age_group


def _normalize_split_value(value: object, *, row_idx: int, field_name: str) -> str:
    token = _normalized_token(value)
    normalized = _SPLIT_ALIASES.get(token)
    if normalized is None:
        raise ValueError(f"Manifest row {row_idx} has unsupported {field_name!r}: {value!r}")
    return normalized


def _prepare_records(
    rows: list[dict[str, object]],
    *,
    manifest_path: Path,
    image_root: Path | None,
    path_column: str,
    split_column: str,
    class_column: str,
    gender_column: str,
    age_column: str,
    age_group_column: str,
    child_max_age: int,
    old_min_age: int,
) -> list[_PreparedRecord]:
    prepared: list[_PreparedRecord] = []
    seen_sources: set[Path] = set()
    for row_idx, row in enumerate(rows, start=2):
        source_path = _resolve_source_path(
            _require_field(row, path_column, row_idx),
            image_root=image_root,
            manifest_path=manifest_path,
            row_idx=row_idx,
        )
        if source_path in seen_sources:
            raise ValueError(f"Manifest row {row_idx} references duplicate source file: {source_path}")
        seen_sources.add(source_path)
        class_name, gender, age_value, age_group = _derive_class_name(
            row,
            row_idx=row_idx,
            class_column=class_column,
            gender_column=gender_column,
            age_column=age_column,
            age_group_column=age_group_column,
            child_max_age=child_max_age,
            old_min_age=old_min_age,
        )
        split_value = row.get(split_column)
        split = None if split_value is None or str(split_value).strip() == "" else _normalize_split_value(
            split_value,
            row_idx=row_idx,
            field_name=split_column,
        )
        prepared.append(
            _PreparedRecord(
                source_path=source_path,
                class_name=class_name,
                split=split,
                gender=gender,
                age_value=age_value,
                age_group=age_group,
            )
        )
    return prepared


def _assign_splits(
    records: list[_PreparedRecord],
    *,
    train_ratio: float,
    seed: int,
) -> str:
    split_values = {record.split for record in records}
    if split_values == {None}:
        records_by_class: dict[str, list[_PreparedRecord]] = defaultdict(list)
        for record in records:
            records_by_class[record.class_name].append(record)
        for class_index, class_name in enumerate(FFHQ_CLASS_NAMES):
            class_records = records_by_class[class_name]
            if len(class_records) < 2:
                raise ValueError(
                    f"Class {class_name!r} needs at least 2 samples for deterministic train/test splitting"
                )
            ordered = sorted(class_records, key=lambda item: str(item.source_path))
            rng = random.Random(seed + class_index)
            rng.shuffle(ordered)
            test_count = max(1, int(round(len(ordered) * (1.0 - train_ratio))))
            test_count = min(test_count, len(ordered) - 1)
            for record in ordered[:test_count]:
                record.split = "test"
            for record in ordered[test_count:]:
                record.split = "train"
        return "stratified"
    if None in split_values:
        raise ValueError("Manifest must either provide split for every row or for no rows")
    return "explicit"


def _validate_split_balance(records: list[_PreparedRecord]) -> None:
    counts = {
        split: {class_name: 0 for class_name in FFHQ_CLASS_NAMES}
        for split in _VALID_SPLITS
    }
    for record in records:
        assert record.split is not None
        counts[record.split][record.class_name] += 1
    for class_name in FFHQ_CLASS_NAMES:
        if counts["train"][class_name] <= 0 or counts["test"][class_name] <= 0:
            raise ValueError(f"Class {class_name!r} must be non-empty in both train and test splits")


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _prepare_output_root(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("train", "test", "split_manifest.csv", "summary.json"):
        _remove_path(output_root / name)


def _target_name(source_path: Path) -> str:
    suffix = source_path.suffix
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    return f"{source_path.stem}_{digest}{suffix}"


def _materialize_file(source_path: Path, target_path: Path, *, mode: str) -> None:
    if mode == "symlink":
        target_path.symlink_to(source_path)
        return
    if mode == "hardlink":
        os.link(source_path, target_path)
        return
    if mode == "copy":
        shutil.copy2(source_path, target_path)
        return
    raise ValueError(f"Unsupported materialize mode: {mode!r}")


def _write_outputs(
    records: list[_PreparedRecord],
    *,
    output_root: Path,
    manifest_path: Path,
    image_root: Path | None,
    materialize_mode: str,
    split_strategy: str,
    child_max_age: int,
    old_min_age: int,
) -> None:
    _prepare_output_root(output_root)
    manifest_rows: list[dict[str, object]] = []
    counts = {split: {class_name: 0 for class_name in FFHQ_CLASS_NAMES} for split in _VALID_SPLITS}
    allocated_targets: set[Path] = set()

    for record in sorted(records, key=lambda item: (item.split or "", item.class_name, str(item.source_path))):
        assert record.split is not None
        class_dir = output_root / record.split / record.class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        target_path = class_dir / _target_name(record.source_path)
        if target_path in allocated_targets:
            raise ValueError(f"Duplicate target path allocation for {target_path}")
        allocated_targets.add(target_path)
        _materialize_file(record.source_path, target_path, mode=materialize_mode)
        counts[record.split][record.class_name] += 1
        manifest_rows.append(
            {
                "source_path": str(record.source_path),
                "split": record.split,
                "class_name": record.class_name,
                "gender": record.gender,
                "age": "" if record.age_value is None else record.age_value,
                "age_group": record.age_group,
                "target_path": str(target_path),
            }
        )

    with (output_root / "split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "split", "class_name", "gender", "age", "age_group", "target_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "manifest": str(manifest_path),
        "image_root": None if image_root is None else str(image_root),
        "output_dir": str(output_root),
        "materialize_mode": materialize_mode,
        "split_strategy": split_strategy,
        "total_files": len(records),
        "age_thresholds": {
            "child_max_age": child_max_age,
            "old_min_age": old_min_age,
        },
        "counts": counts,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_ffhq_six_class_split(
    *,
    manifest: str | Path,
    image_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    path_column: str = "path",
    split_column: str = "split",
    class_column: str = "class_name",
    gender_column: str = "gender",
    age_column: str = "age",
    age_group_column: str = "age_group",
    train_ratio: float = 0.9,
    seed: int = 42,
    child_max_age: int = 19,
    old_min_age: int = 50,
    materialize_mode: str = "symlink",
) -> Path:
    if materialize_mode not in _VALID_MATERIALIZE_MODES:
        raise ValueError(f"materialize_mode must be one of {_VALID_MATERIALIZE_MODES}")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    if child_max_age >= old_min_age:
        raise ValueError("child_max_age must be smaller than old_min_age")

    runtime = default_runtime_roots()
    manifest_path = resolve_latest_alias(normalize_path(manifest, runtime_roots=runtime, resolve_latest=False))
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing FFHQ manifest: {manifest_path}")
    resolved_image_root = (
        None
        if image_root in (None, "")
        else resolve_latest_alias(normalize_path(image_root, runtime_roots=runtime, resolve_latest=False))
    )
    output_root = (
        normalize_path(output_dir, runtime_roots=runtime, resolve_latest=False)
        if output_dir is not None
        else normalize_path("{data_root}/ffhq_6class", runtime_roots=runtime, resolve_latest=False)
    )
    rows = _read_manifest_rows(manifest_path)
    if not rows:
        raise ValueError(f"FFHQ manifest is empty: {manifest_path}")
    records = _prepare_records(
        rows,
        manifest_path=manifest_path,
        image_root=resolved_image_root,
        path_column=path_column,
        split_column=split_column,
        class_column=class_column,
        gender_column=gender_column,
        age_column=age_column,
        age_group_column=age_group_column,
        child_max_age=child_max_age,
        old_min_age=old_min_age,
    )
    split_strategy = _assign_splits(records, train_ratio=train_ratio, seed=seed)
    _validate_split_balance(records)
    _write_outputs(
        records,
        output_root=output_root,
        manifest_path=manifest_path,
        image_root=resolved_image_root,
        materialize_mode=materialize_mode,
        split_strategy=split_strategy,
        child_max_age=child_max_age,
        old_min_age=old_min_age,
    )
    return output_root


def main() -> None:
    runtime_roots = default_runtime_roots()
    parser = argparse.ArgumentParser(
        description="Build a six-class FFHQ train/test directory tree from a metadata manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default=f"{runtime_roots['data_root']}/ffhq_6class")
    parser.add_argument("--path-column", default="path")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--class-column", default="class_name")
    parser.add_argument("--gender-column", default="gender")
    parser.add_argument("--age-column", default="age")
    parser.add_argument("--age-group-column", default="age_group")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--child-max-age", type=int, default=19)
    parser.add_argument("--old-min-age", type=int, default=50)
    parser.add_argument("--materialize-mode", choices=_VALID_MATERIALIZE_MODES, default="symlink")
    args = parser.parse_args()

    output_root = build_ffhq_six_class_split(
        manifest=args.manifest,
        image_root=args.image_root,
        output_dir=args.output_dir,
        path_column=args.path_column,
        split_column=args.split_column,
        class_column=args.class_column,
        gender_column=args.gender_column,
        age_column=args.age_column,
        age_group_column=args.age_group_column,
        train_ratio=args.train_ratio,
        seed=args.seed,
        child_max_age=args.child_max_age,
        old_min_age=args.old_min_age,
        materialize_mode=args.materialize_mode,
    )
    print(output_root)


if __name__ == "__main__":
    main()
