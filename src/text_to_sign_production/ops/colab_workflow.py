"""Fixed Sprint 2 Colab staging, packaging, and publish workflow helpers."""

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

    _require_colab_drive_mount()
    _reset_directory(TRANSLATIONS_DIR)
    _reset_directory(BFH_KEYPOINTS_ROOT)
    ensure_directory(download_root)

    staged_summaries: list[dict[str, str]] = []
    for split in splits:
        translation_source = SPLIT_TO_COLAB_DRIVE_TRANSLATION_PATH[split]
        archive_source = SPLIT_TO_COLAB_DRIVE_KEYPOINT_ARCHIVE_PATH[split]
        translation_destination = TRANSLATIONS_DIR / SPLIT_TO_TRANSLATION_FILE[split]
        local_archive = download_root / archive_source.name
        extract_root = download_root / f"{split}_extract"
        canonical_destination = BFH_KEYPOINTS_ROOT / SPLIT_TO_KEYPOINT_DIR[split]

        if not translation_source.is_file():
            raise FileNotFoundError(
                f"Missing fixed Drive translation CSV for {split}: {translation_source}"
            )
        if not archive_source.is_file():
            raise FileNotFoundError(
                f"Missing fixed Drive keypoint archive for {split}: {archive_source}"
            )

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
            split_dir = find_extracted_split_dir(extract_root, SPLIT_TO_KEYPOINT_DIR[split])
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


def package_sprint2_outputs(
    *,
    project_root: Path = REPO_ROOT,
    splits: tuple[str, ...] = SPLITS,
) -> list[Path]:
    """Create the Sprint 2 output archives under the fixed local archive directory."""

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
                members=(Path("data/processed/v1/samples") / split,),
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


def publish_colab_outputs(*, splits: tuple[str, ...] = SPLITS) -> list[Path]:
    """Package Sprint 2 outputs locally, then copy them to the fixed Drive artifact path."""

    _require_colab_drive_mount()
    archives = package_sprint2_outputs(splits=splits)
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
