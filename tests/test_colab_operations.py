"""Operational support tests for Colab execution and artifact packaging."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import scripts.package_sprint2_outputs as package_outputs_script


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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
        "_extract_tar_gz_with_progress",
        "_find_split_dir",
        'f"{split}_keypoints_archive.tar.gz"',
        '["tar", "-xzf", "-", "-C", str(destination)]',
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
