"""Real local tar/zstd archive integration tests."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

import pytest

import text_to_sign_production.ops.archive_ops as archive_ops_mod
import text_to_sign_production.ops.colab_workflow as colab_workflow_mod
from tests.support.assertions import assert_archive_listing_contains
from tests.support.builders.archives import (
    add_tar_bytes,
    list_tar_zst_members,
    write_tar_zst_archive,
)
from tests.support.paths import write_text
from tests.support.scenarios import create_publish_ready_workspace
from text_to_sign_production.data.constants import ARCHIVES_RELATIVE_ROOT

pytestmark = pytest.mark.integration

requires_tar_and_zstd = pytest.mark.skipif(
    shutil.which("tar") is None or shutil.which("zstd") is None,
    reason="tar and zstd are required for archive tests",
)

requires_zstd = pytest.mark.skipif(
    shutil.which("zstd") is None,
    reason="zstd is required for extraction tests",
)


@requires_tar_and_zstd
def test_package_dataset_build_outputs_creates_expected_archives(tmp_path: Path) -> None:
    create_publish_ready_workspace(tmp_path, splits=("train", "val", "test"))

    archives = colab_workflow_mod.package_dataset_build_outputs(project_root=tmp_path)

    expected_names = {
        "dataset_build_manifests_reports.tar.zst",
        "dataset_build_samples_train.tar.zst",
        "dataset_build_samples_val.tar.zst",
        "dataset_build_samples_test.tar.zst",
    }
    assert {archive.name for archive in archives} == expected_names
    assert all(archive.exists() for archive in archives)

    manifests_archive = tmp_path / "data/archives/dataset_build_manifests_reports.tar.zst"
    assert_archive_listing_contains(
        list_tar_zst_members(manifests_archive),
        {
            "data/interim/raw_manifests/raw_train.jsonl",
            "data/processed/v1/reports/data-quality-report.md",
        },
    )

    train_archive = tmp_path / "data/archives/dataset_build_samples_train.tar.zst"
    assert_archive_listing_contains(
        list_tar_zst_members(train_archive),
        {"data/processed/v1/samples/train/train_sample.npz"},
    )


@requires_tar_and_zstd
def test_package_dataset_build_outputs_default_output_dir_is_project_relative(
    tmp_path: Path,
) -> None:
    assert not ARCHIVES_RELATIVE_ROOT.is_absolute()
    create_publish_ready_workspace(tmp_path, splits=("train",))

    archives = colab_workflow_mod.package_dataset_build_outputs(
        project_root=tmp_path,
        splits=("train",),
    )

    output_dir = tmp_path / "data" / "archives"
    assert {archive.parent for archive in archives} == {output_dir}
    assert (output_dir / "dataset_build_manifests_reports.tar.zst").exists()
    assert (output_dir / "dataset_build_samples_train.tar.zst").exists()


@requires_tar_and_zstd
def test_package_dataset_build_outputs_respects_requested_splits(tmp_path: Path) -> None:
    create_publish_ready_workspace(tmp_path, splits=("train",))
    output_dir = tmp_path / "data/archives"
    write_text(output_dir / "dataset_build_samples_val.tar.zst", "stale-val")
    write_text(output_dir / "dataset_build_samples_test.tar.zst", "stale-test")

    archives = colab_workflow_mod.package_dataset_build_outputs(
        project_root=tmp_path,
        splits=("train",),
    )

    assert {archive.name for archive in archives} == {
        "dataset_build_manifests_reports.tar.zst",
        "dataset_build_samples_train.tar.zst",
    }
    assert not (output_dir / "dataset_build_samples_val.tar.zst").exists()
    assert not (output_dir / "dataset_build_samples_test.tar.zst").exists()


@requires_zstd
def test_extract_tar_zst_with_progress_reports_zstd_stderr_for_invalid_archive(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "corrupt.tar.zst"
    archive_path.write_bytes(b"not-a-valid-archive")
    destination = tmp_path / "extract"
    preserved_file = destination / "existing.txt"
    write_text(preserved_file, "keep me")

    with pytest.raises(RuntimeError, match="zstd stderr") as exc_info:
        archive_ops_mod.extract_tar_zst_with_progress(
            archive_path,
            destination,
            label="Extract archive",
        )

    assert isinstance(exc_info.value.__cause__, tarfile.TarError)
    assert preserved_file.read_text(encoding="utf-8") == "keep me"
    assert not list(tmp_path.glob(".extract.extract-*"))


@requires_zstd
def test_extract_tar_zst_with_progress_extracts_archive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive_path = tmp_path / "sample.tar.zst"

    def build_archive(archive: tarfile.TarFile) -> None:
        add_tar_bytes(archive, "openpose_output/json/sample.json", b"{}")

    write_tar_zst_archive(archive_path, build_archive)

    destination = tmp_path / "extract_destination"
    archive_ops_mod.extract_tar_zst_with_progress(
        archive_path,
        destination,
        label="Extract archive",
    )

    captured = capsys.readouterr()
    assert "Extract archive" in captured.out
    assert (destination / "openpose_output" / "json" / "sample.json").exists()


@requires_zstd
def test_extract_tar_zst_with_progress_rejects_path_traversal_members(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "traversal.tar.zst"

    def build_archive(archive: tarfile.TarFile) -> None:
        add_tar_bytes(archive, "../escape.txt", b"escape")

    write_tar_zst_archive(archive_path, build_archive)

    with pytest.raises(ValueError, match="Unsafe tar member path"):
        archive_ops_mod.extract_tar_zst_with_progress(
            archive_path,
            tmp_path / "extract",
            label="Extract archive",
        )

    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "extract").exists()
    assert not list(tmp_path.glob(".extract.extract-*"))


@requires_zstd
def test_extract_tar_zst_with_progress_rejects_symlink_members(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "symlink.tar.zst"

    def build_archive(archive: tarfile.TarFile) -> None:
        info = tarfile.TarInfo("openpose_output/json/link.json")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../escape.txt"
        archive.addfile(info)

    write_tar_zst_archive(archive_path, build_archive)

    with pytest.raises(ValueError, match="links are not allowed"):
        archive_ops_mod.extract_tar_zst_with_progress(
            archive_path,
            tmp_path / "extract",
            label="Extract archive",
        )

    assert not (tmp_path / "extract").exists()
    assert not list(tmp_path.glob(".extract.extract-*"))


@requires_tar_and_zstd
def test_create_tar_zst_archive_does_not_require_tar_zstd_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_dir = tmp_path / "archive_source"
    write_text(source_dir / "sample.txt", "sample")
    archive_path = tmp_path / "archive.tar.zst"
    monkeypatch.setattr(archive_ops_mod, "tar_supports_zstd", lambda: False)

    created_path = archive_ops_mod.create_tar_zst_archive(
        archive_path=archive_path,
        members=(source_dir.relative_to(tmp_path),),
        cwd=tmp_path,
        label="Archive source",
    )

    captured = capsys.readouterr()
    assert created_path == archive_path
    assert archive_path.exists()
    assert "Archive source" in captured.out
