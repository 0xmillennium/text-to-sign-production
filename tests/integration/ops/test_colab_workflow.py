"""Local integration tests for Colab workflow helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

import text_to_sign_production.ops.colab_workflow as colab_workflow_mod
from tests.support.paths import write_text

pytestmark = pytest.mark.integration


def test_find_extracted_split_dir_handles_supported_layouts_and_ambiguity(tmp_path: Path) -> None:
    layout_a_root = tmp_path / "layout_a"
    (layout_a_root / "train_2D_keypoints" / "openpose_output" / "json").mkdir(parents=True)
    assert colab_workflow_mod.find_extracted_split_dir(
        layout_a_root,
        "train_2D_keypoints",
    ) == (layout_a_root / "train_2D_keypoints")

    layout_b_root = tmp_path / "layout_b"
    (layout_b_root / "openpose_output" / "json").mkdir(parents=True)
    assert colab_workflow_mod.find_extracted_split_dir(layout_b_root, "val_2D_keypoints") == (
        layout_b_root
    )

    ambiguous_root = tmp_path / "ambiguous"
    (ambiguous_root / "a" / "openpose_output" / "json").mkdir(parents=True)
    (ambiguous_root / "b" / "openpose_output" / "json").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Ambiguous extracted split directories"):
        colab_workflow_mod.find_extracted_split_dir(ambiguous_root, "test_2D_keypoints")


def test_stage_colab_inputs_uses_fixed_paths_and_cleans_temp_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drive_mount = tmp_path / "drive"
    drive_mount.mkdir()
    translation_source = drive_mount / "train.csv"
    translation_source.write_text("translation", encoding="utf-8")
    archive_source = drive_mount / "train_2D_keypoints.tar.zst"
    archive_source.write_text("archive", encoding="utf-8")

    translations_dir = tmp_path / "repo" / "data" / "raw" / "how2sign" / "translations"
    keypoints_root = tmp_path / "repo" / "data" / "raw" / "how2sign" / "bfh_keypoints"
    downloads_root = tmp_path / "downloads"

    monkeypatch.setattr(colab_workflow_mod, "COLAB_DRIVE_MOUNT_ROOT", drive_mount)
    monkeypatch.setattr(colab_workflow_mod, "TRANSLATIONS_DIR", translations_dir)
    monkeypatch.setattr(colab_workflow_mod, "BFH_KEYPOINTS_ROOT", keypoints_root)
    monkeypatch.setattr(colab_workflow_mod, "COLAB_DOWNLOAD_ROOT", downloads_root)
    monkeypatch.setattr(
        colab_workflow_mod,
        "SPLIT_TO_COLAB_DRIVE_TRANSLATION_PATH",
        {"train": translation_source},
    )
    monkeypatch.setattr(
        colab_workflow_mod,
        "SPLIT_TO_COLAB_DRIVE_KEYPOINT_ARCHIVE_PATH",
        {"train": archive_source},
    )
    monkeypatch.setattr(colab_workflow_mod, "SPLIT_TO_TRANSLATION_FILE", {"train": "train.csv"})
    monkeypatch.setattr(
        colab_workflow_mod,
        "SPLIT_TO_KEYPOINT_DIR",
        {"train": "train_2D_keypoints"},
    )

    def fake_copy(source: Path, destination: Path, *, label: str) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"{label}:{source.name}", encoding="utf-8")
        return destination

    def fake_extract(archive_path: Path, destination: Path, *, label: str) -> None:
        split_root = destination / "train_2D_keypoints" / "openpose_output" / "json"
        split_root.mkdir(parents=True, exist_ok=True)
        (split_root / "frame.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(colab_workflow_mod, "copy_file_with_progress", fake_copy)
    monkeypatch.setattr(colab_workflow_mod, "extract_tar_zst_with_progress", fake_extract)

    staged = colab_workflow_mod.stage_colab_inputs(splits=("train",), download_root=downloads_root)

    assert staged == [
        {
            "split": "train",
            "translation_path": (translations_dir / "train.csv").as_posix(),
            "keypoints_path": (keypoints_root / "train_2D_keypoints").as_posix(),
        }
    ]
    assert (translations_dir / "train.csv").read_text(encoding="utf-8") == "translation"
    assert (
        keypoints_root / "train_2D_keypoints" / "openpose_output" / "json" / "frame.json"
    ).exists()
    assert not (downloads_root / "train_2D_keypoints.tar.zst").exists()
    assert not (downloads_root / "train_extract").exists()


def test_stage_colab_inputs_prevalidates_all_inputs_before_resetting_raw_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drive_mount = tmp_path / "drive"
    drive_mount.mkdir()
    train_translation = drive_mount / "train.csv"
    train_translation.write_text("translation", encoding="utf-8")
    train_archive = drive_mount / "train_2D_keypoints.tar.zst"
    train_archive.write_text("archive", encoding="utf-8")
    missing_val_translation = drive_mount / "val.csv"
    missing_val_archive = drive_mount / "val_2D_keypoints.tar.zst"

    translations_dir = tmp_path / "repo" / "data" / "raw" / "how2sign" / "translations"
    keypoints_root = tmp_path / "repo" / "data" / "raw" / "how2sign" / "bfh_keypoints"
    preserved_translation = translations_dir / "previous.csv"
    preserved_keypoints = keypoints_root / "previous" / "openpose_output" / "json" / "frame.json"
    write_text(preserved_translation, "old translation")
    write_text(preserved_keypoints, "{}")

    monkeypatch.setattr(colab_workflow_mod, "COLAB_DRIVE_MOUNT_ROOT", drive_mount)
    monkeypatch.setattr(colab_workflow_mod, "TRANSLATIONS_DIR", translations_dir)
    monkeypatch.setattr(colab_workflow_mod, "BFH_KEYPOINTS_ROOT", keypoints_root)
    monkeypatch.setattr(
        colab_workflow_mod,
        "SPLIT_TO_COLAB_DRIVE_TRANSLATION_PATH",
        {"train": train_translation, "val": missing_val_translation},
    )
    monkeypatch.setattr(
        colab_workflow_mod,
        "SPLIT_TO_COLAB_DRIVE_KEYPOINT_ARCHIVE_PATH",
        {"train": train_archive, "val": missing_val_archive},
    )
    monkeypatch.setattr(
        colab_workflow_mod,
        "SPLIT_TO_TRANSLATION_FILE",
        {"train": "train.csv", "val": "val.csv"},
    )
    monkeypatch.setattr(
        colab_workflow_mod,
        "SPLIT_TO_KEYPOINT_DIR",
        {"train": "train_2D_keypoints", "val": "val_2D_keypoints"},
    )

    with pytest.raises(FileNotFoundError, match="Missing fixed Drive translation CSV for val"):
        colab_workflow_mod.stage_colab_inputs(
            splits=("train", "val"),
            download_root=tmp_path / "downloads",
        )

    assert preserved_translation.read_text(encoding="utf-8") == "old translation"
    assert preserved_keypoints.read_text(encoding="utf-8") == "{}"


def test_stage_colab_inputs_rejects_empty_splits_without_resetting_raw_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translations_dir = tmp_path / "repo" / "data" / "raw" / "how2sign" / "translations"
    keypoints_root = tmp_path / "repo" / "data" / "raw" / "how2sign" / "bfh_keypoints"
    preserved_translation = translations_dir / "previous.csv"
    preserved_keypoints = keypoints_root / "previous" / "openpose_output" / "json" / "frame.json"
    write_text(preserved_translation, "old translation")
    write_text(preserved_keypoints, "{}")

    monkeypatch.setattr(colab_workflow_mod, "TRANSLATIONS_DIR", translations_dir)
    monkeypatch.setattr(colab_workflow_mod, "BFH_KEYPOINTS_ROOT", keypoints_root)

    with pytest.raises(ValueError, match="At least one split"):
        colab_workflow_mod.stage_colab_inputs(splits=(), download_root=tmp_path / "downloads")

    assert preserved_translation.read_text(encoding="utf-8") == "old translation"
    assert preserved_keypoints.read_text(encoding="utf-8") == "{}"


def test_publish_colab_outputs_copies_archives_to_fixed_drive_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drive_mount = tmp_path / "drive"
    drive_mount.mkdir()
    artifact_root = drive_mount / "MyDrive" / "text-to-sign-production" / "artifacts"
    local_archive = tmp_path / "local" / "dataset_build_manifests_reports.tar.zst"
    write_text(local_archive, "archive")

    monkeypatch.setattr(colab_workflow_mod, "COLAB_DRIVE_MOUNT_ROOT", drive_mount)
    monkeypatch.setattr(colab_workflow_mod, "COLAB_DRIVE_ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr(
        colab_workflow_mod,
        "package_dataset_build_outputs",
        lambda *, splits: [local_archive],
    )

    published = colab_workflow_mod.publish_colab_outputs(splits=("train",))

    assert published == [artifact_root / local_archive.name]
    assert (artifact_root / local_archive.name).read_text(encoding="utf-8") == "archive"
