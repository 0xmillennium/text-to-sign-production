"""Fixed Colab staging, packaging, and publish helpers for Dataset Build."""

from __future__ import annotations

import shutil
from pathlib import Path

from ..data.constants import (
    ARCHIVES_RELATIVE_ROOT,
    BFH_KEYPOINTS_ROOT,
    COLAB_DOWNLOAD_ROOT,
    COLAB_DRIVE_ARTIFACTS_ROOT,
    COLAB_DRIVE_MOUNT_ROOT,
    MANIFESTS_AND_REPORTS_ARCHIVE_NAME,
    REPO_ROOT,
    SAMPLE_ARCHIVE_NAME_TEMPLATE,
    SPLIT_TO_COLAB_DRIVE_KEYPOINT_ARCHIVE_PATH,
    SPLIT_TO_COLAB_DRIVE_TRANSLATION_PATH,
    SPLIT_TO_KEYPOINT_DIR,
    SPLIT_TO_TRANSLATION_FILE,
    SPLITS,
    TRANSLATIONS_DIR,
)
from ..data.jsonl import iter_jsonl
from ..data.schemas import ProcessedManifestEntry
from ..data.utils import ensure_directory, remove_stale_split_files
from .archive_ops import (
    copy_archive_to_drive,
    copy_file_with_progress,
    create_tar_zst_archive,
    extract_tar_zst_with_progress,
)

MANIFESTS_AND_REPORTS_MEMBERS = (
    Path("data/interim/raw_manifests"),
    Path("data/interim/normalized_manifests"),
    Path("data/interim/filtered_manifests"),
    Path("data/interim/reports"),
    Path("data/processed/v1/manifests"),
    Path("data/processed/v1/reports"),
)


def stage_colab_inputs(
    *,
    splits: tuple[str, ...] = SPLITS,
    download_root: Path = COLAB_DOWNLOAD_ROOT,
) -> list[dict[str, str]]:
    """Stage the fixed Drive inputs into the canonical repo raw layout."""

    if not splits:
        raise ValueError("At least one split must be requested for Colab staging.")
    _require_colab_drive_mount()
    stage_inputs = _collect_stage_inputs(splits)
    _reset_directory(TRANSLATIONS_DIR)
    _reset_directory(BFH_KEYPOINTS_ROOT)
    ensure_directory(download_root)

    staged_summaries: list[dict[str, str]] = []
    for split, translation_source, archive_source, translation_file, keypoint_dir in stage_inputs:
        translation_destination = TRANSLATIONS_DIR / translation_file
        local_archive = download_root / archive_source.name
        extract_root = download_root / f"{split}_extract"
        canonical_destination = BFH_KEYPOINTS_ROOT / keypoint_dir

        translation_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(translation_source, translation_destination)
        local_archive.unlink(missing_ok=True)
        if extract_root.exists():
            shutil.rmtree(extract_root)
        if canonical_destination.exists():
            shutil.rmtree(canonical_destination)

        try:
            copy_file_with_progress(
                archive_source,
                local_archive,
                label=f"[{split}] Copy archive",
            )
            extract_tar_zst_with_progress(
                local_archive,
                extract_root,
                label=f"[{split}] Extract archive",
            )
            split_dir = find_extracted_split_dir(extract_root, keypoint_dir)
            shutil.move(str(split_dir), str(canonical_destination))
        finally:
            local_archive.unlink(missing_ok=True)
            if extract_root.exists():
                shutil.rmtree(extract_root)

        staged_summaries.append(
            {
                "split": split,
                "translation_path": translation_destination.as_posix(),
                "keypoints_path": canonical_destination.as_posix(),
            }
        )
    return staged_summaries


def _collect_stage_inputs(splits: tuple[str, ...]) -> list[tuple[str, Path, Path, str, str]]:
    stage_inputs: list[tuple[str, Path, Path, str, str]] = []
    for split in splits:
        if split not in SPLITS:
            expected = ", ".join(SPLITS)
            raise ValueError(f"Unsupported split {split!r}; expected one of: {expected}")

        try:
            translation_source = SPLIT_TO_COLAB_DRIVE_TRANSLATION_PATH[split]
            archive_source = SPLIT_TO_COLAB_DRIVE_KEYPOINT_ARCHIVE_PATH[split]
            translation_file = SPLIT_TO_TRANSLATION_FILE[split]
            keypoint_dir = SPLIT_TO_KEYPOINT_DIR[split]
        except KeyError as exc:
            raise ValueError(f"Missing fixed Colab input path mapping for split {split!r}") from exc
        if not translation_source.is_file():
            raise FileNotFoundError(
                f"Missing fixed Drive translation CSV for {split}: {translation_source}"
            )
        if not archive_source.is_file():
            raise FileNotFoundError(
                f"Missing fixed Drive keypoint archive for {split}: {archive_source}"
            )
        stage_inputs.append(
            (split, translation_source, archive_source, translation_file, keypoint_dir)
        )
    return stage_inputs


def package_dataset_build_outputs(
    *,
    project_root: Path = REPO_ROOT,
    splits: tuple[str, ...] = SPLITS,
) -> list[Path]:
    """Create Dataset Build output archives under the fixed local archive directory."""

    resolved_output_dir = project_root / ARCHIVES_RELATIVE_ROOT
    ensure_directory(resolved_output_dir)

    archives = [
        create_tar_zst_archive(
            archive_path=resolved_output_dir / MANIFESTS_AND_REPORTS_ARCHIVE_NAME,
            members=MANIFESTS_AND_REPORTS_MEMBERS,
            cwd=project_root,
            label="Archive manifests and reports",
        )
    ]
    for split in splits:
        archives.append(
            create_tar_zst_archive(
                archive_path=resolved_output_dir / SAMPLE_ARCHIVE_NAME_TEMPLATE.format(split=split),
                members=_sample_archive_members_for_split(project_root=project_root, split=split),
                cwd=project_root,
                label=f"Archive samples for {split}",
            )
        )
    remove_stale_split_files(
        resolved_output_dir,
        filename_template=SAMPLE_ARCHIVE_NAME_TEMPLATE,
        requested_splits=splits,
        all_splits=SPLITS,
    )
    return archives


def _sample_archive_members_for_split(*, project_root: Path, split: str) -> tuple[Path, ...]:
    manifest_path = project_root / "data/processed/v1/manifests" / f"{split}.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Processed manifest not found for {split}: {manifest_path}")

    resolved_project_root = project_root.resolve()
    split_samples_root = (project_root / "data/processed/v1/samples" / split).resolve()
    members: list[Path] = []
    seen_members: set[str] = set()
    for index, record in enumerate(iter_jsonl(manifest_path)):
        try:
            entry = ProcessedManifestEntry.from_record(record)
        except Exception as exc:
            raise ValueError(
                f"Could not parse processed manifest record {index} in {manifest_path}: {exc}"
            ) from exc

        if entry.split != split:
            raise ValueError(
                "Processed manifest record split does not match sample archive split "
                f"{split!r}: sample_id={entry.sample_id!r} record_split={entry.split!r}"
            )

        sample_path = Path(entry.sample_path.strip())
        if sample_path.is_absolute():
            raise ValueError(
                f"Processed manifest sample_path must be repo-relative: {entry.sample_path}"
            )
        if sample_path.suffix != ".npz":
            raise ValueError(f"Processed manifest sample_path must end with .npz: {sample_path}")

        resolved_sample_path = (project_root / sample_path).resolve()
        try:
            split_relative_path = resolved_sample_path.relative_to(split_samples_root)
        except ValueError as exc:
            raise ValueError(
                "Processed manifest sample_path must stay under "
                f"data/processed/v1/samples/{split}: {sample_path}"
            ) from exc

        if len(split_relative_path.parts) != 1:
            raise ValueError(
                "Processed manifest sample_path must follow "
                "data/processed/v1/samples/<split>/<sample_id>.npz: "
                f"{sample_path}"
            )
        expected_filename = f"{entry.sample_id}.npz"
        if split_relative_path.name != expected_filename:
            raise ValueError(
                "Processed manifest sample_path filename does not match sample_id "
                f"{entry.sample_id!r}: {sample_path}"
            )

        archive_member = resolved_sample_path.relative_to(resolved_project_root)
        archive_member_key = archive_member.as_posix()
        if archive_member_key in seen_members:
            raise ValueError(
                f"Duplicate processed manifest sample_path in {manifest_path}: {archive_member_key}"
            )
        seen_members.add(archive_member_key)
        members.append(archive_member)

    return tuple(members)


def publish_colab_outputs(*, splits: tuple[str, ...] = SPLITS) -> list[Path]:
    """Package Dataset Build outputs locally, then copy them to the fixed Drive artifact path."""

    _require_colab_drive_mount()
    archives = package_dataset_build_outputs(splits=splits)
    ensure_directory(COLAB_DRIVE_ARTIFACTS_ROOT)

    published_paths: list[Path] = []
    for archive in archives:
        destination = COLAB_DRIVE_ARTIFACTS_ROOT / archive.name
        published_paths.append(
            copy_archive_to_drive(
                archive,
                destination,
                label=f"[publish] Copy {archive.name}",
            )
        )
    return published_paths


def find_extracted_split_dir(search_root: Path, expected_dir_name: str) -> Path:
    """Locate the extracted split root that contains `openpose_output/json`."""

    expected_path = search_root / expected_dir_name
    if _contains_openpose_json(expected_path):
        return expected_path
    if _contains_openpose_json(search_root):
        return search_root

    expected_candidates = sorted(
        path
        for path in search_root.rglob(expected_dir_name)
        if path.is_dir() and _contains_openpose_json(path)
    )
    if len(expected_candidates) == 1:
        return expected_candidates[0]
    if len(expected_candidates) > 1:
        raise RuntimeError(
            f"Ambiguous extracted split directories under {search_root}: {expected_candidates}"
        )

    derived_candidates = sorted(
        {
            path.parent
            for path in search_root.rglob("openpose_output")
            if path.is_dir() and (path / "json").is_dir()
        },
        key=lambda path: str(path),
    )
    if not derived_candidates:
        raise RuntimeError(
            "Could not locate an extracted split directory with "
            f"openpose_output/json under {search_root}."
        )
    if len(derived_candidates) > 1:
        raise RuntimeError(
            f"Ambiguous extracted split directories under {search_root}: {derived_candidates}"
        )
    return derived_candidates[0]


def _contains_openpose_json(path: Path) -> bool:
    return (path / "openpose_output" / "json").is_dir()


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _require_colab_drive_mount() -> None:
    if not COLAB_DRIVE_MOUNT_ROOT.is_dir():
        raise FileNotFoundError(
            "Google Drive is not mounted at /content/drive. Mount Drive before running this step."
        )
