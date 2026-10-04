from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from huggingface_hub import snapshot_download

from ..utils import default_runtime_roots, normalize_path

DEFAULT_FFHQ_AGING_HF_REPO_ID = "NUS-SRI-2025/FFHQ-Aging-Dataset"
HF_FFHQ_AGING_LABELS_PATH = "labels/ffhq_aging_labels.csv"
HF_FFHQ_AGING_ARCHIVE_GLOB = "images_zip/images1024x1024/*.zip"


def ffhq_aging_image_dir_name(resolution: int) -> str:
    return f"ffhq_aging{int(resolution)}x{int(resolution)}"


def ffhq_aging_relative_image_path(image_number: int) -> str:
    bucket = int(image_number) // 1000 * 1000
    return f"{bucket:05d}/{int(image_number):05d}.png"


def _sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _download_root(work_dir: Path, repo_id: str) -> Path:
    return work_dir / _sanitize_repo_id(repo_id)


def download_ffhq_aging_hf_snapshot(
    *,
    repo_id: str,
    work_dir: str | Path,
) -> Path:
    root = _download_root(Path(work_dir).resolve(), repo_id)
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[HF_FFHQ_AGING_LABELS_PATH, HF_FFHQ_AGING_ARCHIVE_GLOB],
        local_dir=str(root),
    )
    return root


def _archive_members(archive_path: Path) -> list[tuple[str, Path]]:
    members: list[tuple[str, Path]] = []
    with zipfile.ZipFile(archive_path) as handle:
        for name in handle.namelist():
            pure = PurePosixPath(name)
            if name.endswith("/") or pure.name == "":
                continue
            if pure.suffix.lower() != ".png":
                continue
            image_stem = pure.stem
            bucket_name = pure.parts[-2] if len(pure.parts) >= 2 else None
            if bucket_name and bucket_name.isdigit() and image_stem.isdigit():
                members.append((name, Path(bucket_name) / pure.parts[-1]))
                continue
            if image_stem.isdigit():
                members.append((name, Path(ffhq_aging_relative_image_path(int(image_stem)))))
                continue
            raise ValueError(
                "Archive member must be either bucket/image.png or flat NNNNN.png: "
                f"{name!r} in {archive_path}"
            )
    return members


def extract_ffhq_aging_archives(
    *,
    archives_dir: str | Path,
    image_root: str | Path,
) -> dict[str, int]:
    archives_root = Path(archives_dir).resolve()
    resolved_image_root = Path(image_root).resolve()
    archive_paths = sorted(archives_root.glob("*.zip"))
    if not archive_paths:
        raise FileNotFoundError(f"Missing FFHQ-Aging archive parts under {archives_root}")

    resolved_image_root.mkdir(parents=True, exist_ok=True)
    extracted_files = 0
    skipped_files = 0
    expected_files = 0
    archive_count = 0

    for archive_path in archive_paths:
        archive_count += 1
        try:
            members = _archive_members(archive_path)
        except zipfile.BadZipFile as exc:
            raise zipfile.BadZipFile(f"Invalid FFHQ-Aging archive: {archive_path}") from exc
        if not members:
            raise ValueError(f"Archive does not contain PNG members: {archive_path}")
        with zipfile.ZipFile(archive_path) as handle:
            for member_name, relative_path in members:
                expected_files += 1
                target_path = resolved_image_root / relative_path
                if target_path.is_file():
                    skipped_files += 1
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member_name) as src, target_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_files += 1

    if expected_files <= 0:
        raise ValueError(f"No FFHQ PNG images were found under {archives_root}")

    return {
        "archive_count": archive_count,
        "expected_files": expected_files,
        "extracted_files": extracted_files,
        "skipped_files": skipped_files,
    }


def build_ffhq_aging_manifest(
    *,
    labels_csv: str | Path,
    image_root: str | Path,
    manifest_out: str | Path,
) -> Path:
    labels_path = Path(labels_csv).resolve()
    image_root_path = Path(image_root).resolve()
    manifest_path = Path(manifest_out).resolve()
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing FFHQ-Aging labels CSV: {labels_path}")
    if not image_root_path.is_dir():
        raise FileNotFoundError(f"Missing FFHQ-Aging image root: {image_root_path}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, str]] = []

    with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader, start=2):
            raw_image_number = row.get("image_number")
            raw_gender = row.get("gender")
            raw_age_group = row.get("age_group")
            if raw_image_number is None or str(raw_image_number).strip() == "":
                raise ValueError(f"FFHQ-Aging labels row {row_idx} is missing 'image_number'")
            if raw_gender is None or str(raw_gender).strip() == "":
                raise ValueError(f"FFHQ-Aging labels row {row_idx} is missing 'gender'")
            if raw_age_group is None or str(raw_age_group).strip() == "":
                raise ValueError(f"FFHQ-Aging labels row {row_idx} is missing 'age_group'")
            try:
                image_number = int(str(raw_image_number).strip())
            except ValueError as exc:
                raise ValueError(
                    f"FFHQ-Aging labels row {row_idx} has non-integer 'image_number': {raw_image_number!r}"
                ) from exc

            relative_path = ffhq_aging_relative_image_path(image_number)
            image_path = image_root_path / relative_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Expected FFHQ-Aging image is missing: {image_path}")

            rows_out.append(
                {
                    "path": relative_path,
                    "gender": str(raw_gender).strip(),
                    "age_group": str(raw_age_group).strip(),
                }
            )

    if not rows_out:
        raise ValueError("FFHQ-Aging manifest would be empty after image validation")

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "gender", "age_group"])
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "labels_csv": str(labels_path),
        "image_root": str(image_root_path),
        "manifest_out": str(manifest_path),
        "kept_rows": len(rows_out),
        "missing_rows": 0,
    }
    (manifest_path.parent / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest_path


def _remove_tree_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def prepare_ffhq_aging_public_source(
    *,
    repo_id: str = DEFAULT_FFHQ_AGING_HF_REPO_ID,
    image_dir: str | Path | None = None,
    manifest_out: str | Path | None = None,
    work_dir: str | Path | None = None,
    keep_archives: bool = False,
) -> Path:
    runtime = default_runtime_roots()
    resolved_image_dir = (
        normalize_path(image_dir, runtime_roots=runtime, resolve_latest=False)
        if image_dir is not None
        else normalize_path(
            f"{{data_root}}/{ffhq_aging_image_dir_name(1024)}",
            runtime_roots=runtime,
            resolve_latest=False,
        )
    )
    resolved_manifest_out = (
        normalize_path(manifest_out, runtime_roots=runtime, resolve_latest=False)
        if manifest_out is not None
        else normalize_path("{data_root}/ffhq_aging/ffhq_aging_manifest.csv", runtime_roots=runtime, resolve_latest=False)
    )
    resolved_work_dir = (
        normalize_path(work_dir, runtime_roots=runtime, resolve_latest=False)
        if work_dir is not None
        else normalize_path("{data_root}/ffhq_aging_hf", runtime_roots=runtime, resolve_latest=False)
    )
    resolved_work_dir.mkdir(parents=True, exist_ok=True)

    download_root = download_ffhq_aging_hf_snapshot(repo_id=repo_id, work_dir=resolved_work_dir)
    labels_csv = download_root / HF_FFHQ_AGING_LABELS_PATH
    archives_dir = download_root / "images_zip" / "images1024x1024"
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Missing FFHQ-Aging labels file in Hugging Face snapshot: {labels_csv}")
    stats = extract_ffhq_aging_archives(archives_dir=archives_dir, image_root=resolved_image_dir)
    manifest_path = build_ffhq_aging_manifest(
        labels_csv=labels_csv,
        image_root=resolved_image_dir,
        manifest_out=resolved_manifest_out,
    )

    summary_path = manifest_path.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["repo_id"] = repo_id
    summary["work_dir"] = str(resolved_work_dir)
    summary["download_root"] = str(download_root)
    summary["keep_archives"] = bool(keep_archives)
    summary.update(stats)

    if not keep_archives:
        _remove_tree_if_exists(download_root)
        if resolved_work_dir.exists() and not any(resolved_work_dir.iterdir()):
            resolved_work_dir.rmdir()
        summary["archives_cleaned"] = True
    else:
        summary["archives_cleaned"] = False

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest_path


def _parse_args() -> argparse.Namespace:
    runtime_roots = default_runtime_roots()
    parser = argparse.ArgumentParser(
        description="Download FFHQ-Aging from a Hugging Face mirror, extract 1024x1024 images, and write a canonical manifest."
    )
    parser.add_argument("--repo-id", default=DEFAULT_FFHQ_AGING_HF_REPO_ID)
    parser.add_argument("--image-dir", default=f"{runtime_roots['data_root']}/{ffhq_aging_image_dir_name(1024)}")
    parser.add_argument("--manifest-out", default=f"{runtime_roots['data_root']}/ffhq_aging/ffhq_aging_manifest.csv")
    parser.add_argument("--work-dir", default=f"{runtime_roots['data_root']}/ffhq_aging_hf")
    parser.add_argument("--keep-archives", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = prepare_ffhq_aging_public_source(
        repo_id=args.repo_id,
        image_dir=args.image_dir,
        manifest_out=args.manifest_out,
        work_dir=args.work_dir,
        keep_archives=args.keep_archives,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
