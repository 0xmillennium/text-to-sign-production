"""Operational support tests for Colab execution and artifact packaging."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

import scripts.package_sprint2_outputs as package_outputs_script


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_colab_raw_setup_namespace() -> dict[str, Any]:
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks/colab/sprint2_pipeline_colab.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    raw_setup_source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "def _extract_tar_gz_with_progress" in "".join(cell.get("source", []))
    )
    helper_source, separator, _ = raw_setup_source.partition("_reset_directory(TRANSLATIONS_DIR)")
    if not separator:
        raise AssertionError("Could not isolate raw setup helpers from the notebook cell.")

    original_gdown = sys.modules.get("gdown")
    gdown_stub = types.ModuleType("gdown")
    gdown_stub.download = lambda **kwargs: kwargs.get("output")  # type: ignore[attr-defined]
    sys.modules["gdown"] = gdown_stub

    namespace: dict[str, Any] = {}
    try:
        exec(helper_source, namespace)
    finally:
        if original_gdown is None:
            sys.modules.pop("gdown", None)
        else:
            sys.modules["gdown"] = original_gdown
    return namespace


@pytest.mark.skipif(
    shutil.which("zstd") is None or not package_outputs_script._tar_supports_zstd(),
    reason="tar with --zstd support and zstd are required for archive tests",
)
def test_package_sprint2_outputs_creates_expected_archives(tmp_path: Path) -> None:
    manifest_and_report_files = {
        "data/interim/raw_manifests/raw_train.jsonl": '{"sample_id": "train_sample"}\n',
        "data/interim/normalized_manifests/normalized_train.jsonl": (
            '{"sample_id": "train_sample"}\n'
        ),
        "data/interim/filtered_manifests/filtered_train.jsonl": '{"sample_id": "train_sample"}\n',
        "data/interim/reports/assumption-report.json": '{"generated_at": "2026-04-06"}\n',
        "data/processed/v1/manifests/train.jsonl": '{"sample_id": "train_sample"}\n',
        "data/processed/v1/reports/data-quality-report.md": "# Data Quality Report\n",
    }
    for relative_path, content in manifest_and_report_files.items():
        _write(tmp_path / relative_path, content)

    for split in ("train", "val", "test"):
        _write(
            tmp_path / "data/processed/v1/samples" / split / f"{split}_sample.npz",
            f"{split}-sample",
        )

    output_dir = tmp_path / "data/archives"
    archives = package_outputs_script.package_outputs(project_root=tmp_path, output_dir=output_dir)

    expected_names = {
        "sprint2_manifests_reports.tar.zst",
        "sprint2_samples_train.tar.zst",
        "sprint2_samples_val.tar.zst",
        "sprint2_samples_test.tar.zst",
    }
    assert {archive.name for archive in archives} == expected_names
    assert all(archive.exists() for archive in archives)

    manifests_archive = output_dir / "sprint2_manifests_reports.tar.zst"
    manifests_listing = subprocess.run(
        ["tar", "--list", "--zstd", "-f", str(manifests_archive)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "data/interim/raw_manifests/raw_train.jsonl" in manifests_listing
    assert "data/processed/v1/reports/data-quality-report.md" in manifests_listing

    train_archive = output_dir / "sprint2_samples_train.tar.zst"
    train_listing = subprocess.run(
        ["tar", "--list", "--zstd", "-f", str(train_archive)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "data/processed/v1/samples/train/train_sample.npz" in train_listing


@pytest.mark.skipif(
    shutil.which("zstd") is None or not package_outputs_script._tar_supports_zstd(),
    reason="tar with --zstd support and zstd are required for archive tests",
)
def test_package_sprint2_outputs_respects_requested_splits(tmp_path: Path) -> None:
    for relative_path in (
        "data/interim/raw_manifests/raw_train.jsonl",
        "data/interim/normalized_manifests/normalized_train.jsonl",
        "data/interim/filtered_manifests/filtered_train.jsonl",
        "data/interim/reports/assumption-report.json",
        "data/processed/v1/manifests/train.jsonl",
        "data/processed/v1/reports/data-quality-report.md",
    ):
        _write(tmp_path / relative_path, "placeholder\n")

    _write(tmp_path / "data/processed/v1/samples/train/train_sample.npz", "train-sample")
    output_dir = tmp_path / "data/archives"
    _write(output_dir / "sprint2_samples_val.tar.zst", "stale-val")
    _write(output_dir / "sprint2_samples_test.tar.zst", "stale-test")

    archives = package_outputs_script.package_outputs(
        project_root=tmp_path,
        output_dir=output_dir,
        splits=("train",),
    )

    assert {archive.name for archive in archives} == {
        "sprint2_manifests_reports.tar.zst",
        "sprint2_samples_train.tar.zst",
    }
    assert not (output_dir / "sprint2_samples_val.tar.zst").exists()
    assert not (output_dir / "sprint2_samples_test.tar.zst").exists()


def test_colab_find_split_dir_handles_supported_layouts_and_ambiguity(tmp_path: Path) -> None:
    namespace = _load_colab_raw_setup_namespace()
    find_split_dir = cast(Callable[[Path, str], Path], namespace["_find_split_dir"])

    layout_a_root = tmp_path / "layout_a"
    (layout_a_root / "train_2D_keypoints" / "openpose_output" / "json").mkdir(parents=True)
    assert (
        find_split_dir(layout_a_root, "train_2D_keypoints") == layout_a_root / "train_2D_keypoints"
    )

    layout_b_root = tmp_path / "layout_b"
    (layout_b_root / "openpose_output" / "json").mkdir(parents=True)
    assert find_split_dir(layout_b_root, "val_2D_keypoints") == layout_b_root

    ambiguous_root = tmp_path / "ambiguous"
    (ambiguous_root / "a" / "openpose_output" / "json").mkdir(parents=True)
    (ambiguous_root / "b" / "openpose_output" / "json").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Ambiguous extracted split directories"):
        find_split_dir(ambiguous_root, "test_2D_keypoints")


@pytest.mark.skipif(
    shutil.which("tar") is None,
    reason="tar is required for notebook extraction helper tests",
)
def test_colab_extract_tar_gz_with_progress_extracts_archive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    namespace = _load_colab_raw_setup_namespace()
    extract_tar_gz_with_progress = cast(
        Callable[[Path, Path], None], namespace["_extract_tar_gz_with_progress"]
    )

    source_root = tmp_path / "archive_source"
    sample_file = source_root / "openpose_output" / "json" / "sample.json"
    sample_file.parent.mkdir(parents=True)
    sample_file.write_text("{}", encoding="utf-8")

    archive_path = tmp_path / "sample.tar.gz"
    subprocess.run(["tar", "-czf", str(archive_path), "-C", str(source_root), "."], check=True)

    destination = tmp_path / "extract_destination"
    extract_tar_gz_with_progress(archive_path, destination)

    captured = capsys.readouterr()
    assert "Extraction progress" in captured.out
    assert (destination / "openpose_output" / "json" / "sample.json").exists()


def test_colab_stage_public_split_cleans_temp_artifacts_on_failure(tmp_path: Path) -> None:
    namespace = _load_colab_raw_setup_namespace()
    stage_public_split = cast(Callable[[str], None], namespace["_stage_public_split"])

    downloads_root = tmp_path / "downloads"
    translations_dir = tmp_path / "translations"
    keypoints_root = tmp_path / "bfh_keypoints"
    downloads_root.mkdir(parents=True)
    translations_dir.mkdir(parents=True)
    keypoints_root.mkdir(parents=True)

    namespace["DOWNLOAD_ROOT"] = downloads_root
    namespace["TRANSLATIONS_DIR"] = translations_dir
    namespace["BFH_KEYPOINTS_ROOT"] = keypoints_root
    namespace["TRANSLATION_URLS"] = {"train": "https://example.invalid/train.csv"}
    namespace["KEYPOINT_ARCHIVE_URLS"] = {"train": "https://example.invalid/train.tar.gz"}
    namespace["TRANSLATION_FILE_NAMES"] = {"train": "how2sign_realigned_train.csv"}
    namespace["SPLIT_DIR_NAMES"] = {"train": "train_2D_keypoints"}

    def fake_download(url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(url, encoding="utf-8")

    def fake_extract(archive_path: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "partial").mkdir()
        raise RuntimeError("boom")

    namespace["_download_file"] = fake_download
    namespace["_extract_tar_gz_with_progress"] = fake_extract

    with pytest.raises(RuntimeError, match="boom"):
        stage_public_split("train")

    assert (translations_dir / "how2sign_realigned_train.csv").exists()
    assert not (downloads_root / "train_keypoints_archive.tar.gz").exists()
    assert not (downloads_root / "train_extract").exists()


def test_colab_notebook_contains_required_stages_and_script_calls() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks/colab/sprint2_pipeline_colab.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    markdown_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    expected_headings = (
        "# Sprint 2 Colab Pipeline",
        "## 1. Environment Setup",
        "## 2. Repository Setup",
        "## 3. Raw Data Fetch And Setup",
        "## 4. Sprint 2 Script Execution",
        "## 5. Output Packaging",
        "## 6. Download Or Private Storage Copy",
    )
    for heading in expected_headings:
        assert heading in markdown_source

    expected_snippets = (
        "https://github.com/0xmillennium/text-to-sign-production.git",
        'REPO_REF = "master"',
        'PIPELINE_SPLITS = ["train", "val", "test"]',
        "configs/storage.local.yaml",
        "gdown",
        "gdown.download",
        "_is_google_drive_url",
        "_close_process_stdin",
        "_stop_process",
        "_extract_tar_gz_with_progress",
        "_find_split_dir",
        'f"{split}_keypoints_archive.tar.gz"',
        '["tar", "-xzf", "-", "-C", str(destination)]',
        "stderr=subprocess.STDOUT",
        "search_root.rglob(expected_dir_name)",
        'search_root.rglob("openpose_output")',
        "if archive_path.exists() or extract_root.exists():",
        "openpose_output",
        "json",
        "shutil.move",
        "drive.google.com",
        "docs.google.com",
        "drive.usercontent.google.com",
        "scripts/prepare_raw.py",
        "scripts/normalize_keypoints.py",
        "scripts/filter_samples.py",
        "configs/data/filter-v1.yaml",
        "scripts/export_training_manifest.py",
        "scripts/package_sprint2_outputs.py",
        "--splits",
        "*PIPELINE_SPLITS",
    )
    for snippet in expected_snippets:
        assert snippet in code_source
