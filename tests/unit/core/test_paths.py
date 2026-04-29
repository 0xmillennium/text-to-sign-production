"""Neutral repository and data path resolution tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sign_production.core.paths import (
    ProjectLayout,
    data_root_relative_path,
    repo_relative_path,
    resolve_data_root_path,
    resolve_manifest_path,
    resolve_repo_path,
)

pytestmark = pytest.mark.unit


def test_project_layout_canonical_data_roots() -> None:
    root = Path("/runtime/project")
    layout = ProjectLayout(root)

    assert layout.data_root == root / "data"
    assert layout.raw_root == root / "data/raw"
    assert layout.interim_root == root / "data/interim"
    assert layout.processed_root == root / "data/processed"
    assert layout.artifacts_root == root / "data/artifacts"


def test_project_layout_runtime_and_drive_roots_mirror_relative_paths() -> None:
    runtime_layout = ProjectLayout(Path("/content/text-to-sign-production"))
    drive_layout = ProjectLayout(Path("/content/drive/MyDrive/text-to-sign-production"))

    runtime_relative = runtime_layout.processed_sample_archive("train").relative_to(
        runtime_layout.root
    )
    drive_relative = drive_layout.processed_sample_archive("train").relative_to(drive_layout.root)

    assert runtime_relative == drive_relative
    assert runtime_relative.as_posix().startswith("data/")


def test_project_layout_how2sign_roots_and_archives() -> None:
    root = Path("/runtime/project")
    layout = ProjectLayout(root)

    assert layout.how2sign_translations_root == root / "data/raw/how2sign/translations"
    assert layout.how2sign_bfh_keypoints_root == root / "data/raw/how2sign/bfh_keypoints"
    assert (
        layout.raw_bfh_keypoints_archive("train")
        == root / "data/raw/how2sign/bfh_keypoints/train_2D_keypoints.tar.zst"
    )
    assert (
        layout.raw_bfh_keypoints_split_root("val")
        == root / "data/raw/how2sign/bfh_keypoints/val_2D_keypoints/openpose_output"
    )
    assert (
        layout.how2sign_translation_file("test")
        == root / "data/raw/how2sign/translations/how2sign_realigned_test.csv"
    )


def test_project_layout_processed_and_artifact_paths() -> None:
    root = Path("/runtime/project")
    layout = ProjectLayout(root)

    assert (
        layout.processed_manifests_reports_archive()
        == root / "data/processed/v1/dataset_build_manifests_reports.tar.zst"
    )
    assert (
        layout.processed_sample_archive("train")
        == root / "data/processed/v1/samples/dataset_build_samples_train.tar.zst"
    )
    assert layout.dataset_artifacts_root == root / "data/artifacts/dataset"
    assert layout.visualization_artifacts_root == root / "data/artifacts/visualization"
    assert layout.base_run_root("run1") == root / "data/artifacts/base/run1"
    assert layout.base_run_archive("run1") == root / "data/artifacts/base/run1.tar.zst"


def test_resolve_repo_path_resolves_repo_relative_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    resolved = resolve_repo_path("configs/data/filter-v2.yaml", repo_root=repo_root)

    assert resolved == (repo_root / "configs/data/filter-v2.yaml").resolve()


def test_repo_relative_path_rejects_paths_outside_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    outside_path = tmp_path / "outside.txt"
    repo_root.mkdir()
    outside_path.write_text("outside", encoding="utf-8")

    with pytest.raises(ValueError, match="must stay under"):
        repo_relative_path(outside_path, repo_root=repo_root)


def test_resolve_data_root_path_supports_repo_style_values(tmp_path: Path) -> None:
    data_root = tmp_path / "data"

    resolved = resolve_data_root_path(
        "data/processed/v1/manifests/train.jsonl",
        data_root=data_root,
    )

    assert resolved == (data_root / "processed/v1/manifests/train.jsonl").resolve()


def test_resolve_data_root_path_supports_data_root_relative_values(tmp_path: Path) -> None:
    data_root = tmp_path / "data"

    resolved = resolve_data_root_path("processed/v1/samples/train/sample.npz", data_root=data_root)

    assert resolved == (data_root / "processed/v1/samples/train/sample.npz").resolve()


def test_data_root_relative_path_returns_paths_under_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    path = data_root / "interim/raw_manifests/raw_train.jsonl"

    assert data_root_relative_path(path, data_root=data_root) == Path(
        "interim/raw_manifests/raw_train.jsonl"
    )


def test_resolve_manifest_path_rejects_absolute_paths_by_default(tmp_path: Path) -> None:
    data_root = tmp_path / "data"

    with pytest.raises(ValueError, match="must be relative"):
        resolve_manifest_path(data_root / "processed/v1/samples/train/sample.npz")


def test_resolve_manifest_path_allows_absolute_paths_only_when_requested(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    sample_path = data_root / "processed/v1/samples/train/sample.npz"

    resolved = resolve_manifest_path(
        sample_path,
        data_root=data_root,
        allow_absolute=True,
    )

    assert resolved == sample_path.resolve()


def test_resolve_manifest_path_rejects_parent_traversal(tmp_path: Path) -> None:
    data_root = tmp_path / "data"

    with pytest.raises(ValueError, match="parent traversal"):
        resolve_manifest_path("processed/v1/../outside.npz", data_root=data_root)


def test_resolve_manifest_path_enforces_allowed_root(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    allowed_root = data_root / "processed/v1/samples"

    with pytest.raises(ValueError, match="must stay under"):
        resolve_manifest_path(
            "data/processed/v1/reports/report.json",
            data_root=data_root,
            allowed_root=allowed_root,
        )

    resolved = resolve_manifest_path(
        "data/processed/v1/samples/train/sample.npz",
        data_root=data_root,
        allowed_root=allowed_root,
    )
    assert resolved == (allowed_root / "train/sample.npz").resolve()
