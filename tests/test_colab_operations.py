"""Operational support tests for the simplified Sprint 2 Colab workflow."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, BinaryIO, cast

import pytest

import scripts.package_sprint2_outputs as package_outputs_script
import text_to_sign_production.ops.archive_ops as archive_ops_mod
import text_to_sign_production.ops.colab_workflow as colab_workflow_mod
from text_to_sign_production.ops.progress import ProgressReporter


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_notebook_sources() -> tuple[str, str]:
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks/colab/sprint2_pipeline_colab.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
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
    return markdown_source, code_source


def _write_minimal_packaging_outputs(root: Path, *, splits: tuple[str, ...]) -> None:
    for relative_path in (
        "data/interim/raw_manifests/raw_train.jsonl",
        "data/interim/normalized_manifests/normalized_train.jsonl",
        "data/interim/filtered_manifests/filtered_train.jsonl",
        "data/interim/reports/assumption-report.json",
        "data/processed/v1/manifests/train.jsonl",
        "data/processed/v1/reports/data-quality-report.md",
    ):
        _write(root / relative_path, "placeholder\n")
    for split in splits:
        _write(root / "data/processed/v1/samples" / split / f"{split}_sample.npz", "sample")


class _BrokenPipeStdin:
    def write(self, chunk: bytes) -> int:
        raise BrokenPipeError("pipe closed")

    def close(self) -> None:
        return


class _OneChunkStdout:
    def __init__(self, chunk: bytes) -> None:
        self._chunk = chunk

    def read(self, size: int) -> bytes:
        if not self._chunk:
            return b""
        chunk = self._chunk
        self._chunk = b""
        return chunk


class _FakeExtractBrokenPipeProcess:
    def __init__(self, stderr_file: BinaryIO) -> None:
        self.stdin: _BrokenPipeStdin | None = _BrokenPipeStdin()
        stderr_file.write(b"tar: corrupt archive")

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 2

    def terminate(self) -> None:
        return

    def kill(self) -> None:
        return


class _FakeCreateTarProcess:
    def __init__(self, stderr_file: BinaryIO) -> None:
        self.stdin = None
        self.stdout: _OneChunkStdout | None = _OneChunkStdout(b"tar-stream")
        self._returncode: int | None = None

    def poll(self) -> int | None:
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:
        if self._returncode is None:
            self._returncode = 0
        return self._returncode

    def terminate(self) -> None:
        self._returncode = -15

    def kill(self) -> None:
        self._returncode = -9


class _FakeCreateZstdProcess:
    def __init__(self, stderr_file: BinaryIO) -> None:
        self.stdin: _BrokenPipeStdin | None = _BrokenPipeStdin()
        self.stdout = None
        stderr_file.write(b"zstd: simulated failure")

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 1

    def terminate(self) -> None:
        return

    def kill(self) -> None:
        return


class _TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_progress_reporter_uses_newline_updates_for_non_tty_streams() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(
        "Copy archive",
        total_bytes=10,
        stream=stream,
        min_interval_seconds=0,
    )

    reporter.update(0)
    reporter.update(5)
    reporter.finish(10)

    output = stream.getvalue()
    lines = output.splitlines()
    assert "\r" not in output
    assert len(lines) == 3
    assert "0 B / 10 B" in lines[0]
    assert "50.0%" in lines[1]
    assert "100.0%" in lines[2]


def test_progress_reporter_env_forces_line_updates_for_colab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("T2SP_PROGRESS_MODE", "lines")
    stream = _TtyStringIO()
    reporter = ProgressReporter(
        "Copy archive",
        total_bytes=10,
        stream=stream,
        min_interval_seconds=0,
    )

    reporter.update(0)
    reporter.finish(10)

    output = stream.getvalue()
    assert "\r" not in output
    assert "Copy archive" in output
    assert "100.0%" in output


def test_progress_reporter_pads_shorter_carriage_return_updates() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(
        "Archive",
        total_bytes=None,
        stream=stream,
        min_interval_seconds=0,
        use_carriage_return=True,
    )

    reporter.update(1024**3)
    reporter.update(1)

    long_line = "Archive: 1.0 GiB transferred"
    short_line = "Archive: 1 B transferred"
    expected_padding = " " * (len(long_line) - len(short_line))
    assert f"\r{short_line}{expected_padding}" in stream.getvalue()


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
def test_package_sprint2_outputs_default_output_dir_is_project_relative(
    tmp_path: Path,
) -> None:
    assert not package_outputs_script.DEFAULT_OUTPUT_DIR.is_absolute()
    _write_minimal_packaging_outputs(tmp_path, splits=("train",))

    archives = package_outputs_script.package_outputs(project_root=tmp_path, splits=("train",))

    output_dir = tmp_path / "data" / "archives"
    assert {archive.parent for archive in archives} == {output_dir}
    assert (output_dir / "sprint2_manifests_reports.tar.zst").exists()
    assert (output_dir / "sprint2_samples_train.tar.zst").exists()


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


def test_extract_tar_zst_with_progress_rejects_non_tar_zst_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "sample.tar.gz"
    archive_path.write_text("not-a-tar-zst", encoding="utf-8")
    monkeypatch.setattr(archive_ops_mod, "ensure_tar_zst_extract_prerequisites", lambda: None)

    with pytest.raises(ValueError, match=r"\.tar\.zst"):
        archive_ops_mod.extract_tar_zst_with_progress(
            archive_path,
            tmp_path / "extract",
            label="Extract archive",
        )


def test_extract_tar_zst_with_progress_rejects_missing_archive_without_destination_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "missing.tar.zst"
    destination = tmp_path / "extract"
    monkeypatch.setattr(archive_ops_mod, "ensure_tar_zst_extract_prerequisites", lambda: None)

    with pytest.raises(FileNotFoundError, match="Archive file not found"):
        archive_ops_mod.extract_tar_zst_with_progress(
            archive_path,
            destination,
            label="Extract archive",
        )

    assert not destination.exists()


def test_extract_tar_zst_with_progress_reports_stderr_after_broken_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "corrupt.tar.zst"
    archive_path.write_bytes(b"not-a-valid-archive")
    monkeypatch.setattr(archive_ops_mod, "ensure_tar_zst_extract_prerequisites", lambda: None)

    def fake_popen(command: Sequence[str], **kwargs: Any) -> Any:
        stderr_file = cast(BinaryIO, kwargs["stderr"])
        return _FakeExtractBrokenPipeProcess(stderr_file)

    monkeypatch.setattr("text_to_sign_production.ops.archive_ops.subprocess.Popen", fake_popen)

    with pytest.raises(RuntimeError, match="tar stderr: tar: corrupt archive") as exc_info:
        archive_ops_mod.extract_tar_zst_with_progress(
            archive_path,
            tmp_path / "extract",
            label="Extract archive",
        )

    assert isinstance(exc_info.value.__cause__, BrokenPipeError)


@pytest.mark.skipif(
    shutil.which("zstd") is None or not package_outputs_script._tar_supports_zstd(),
    reason="tar with --zstd support and zstd are required for extraction tests",
)
def test_extract_tar_zst_with_progress_extracts_archive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_root = tmp_path / "archive_source"
    sample_file = source_root / "openpose_output" / "json" / "sample.json"
    sample_file.parent.mkdir(parents=True)
    sample_file.write_text("{}", encoding="utf-8")

    archive_path = tmp_path / "sample.tar.zst"
    subprocess.run(
        ["tar", "--zstd", "-cf", str(archive_path), "-C", str(source_root), "."],
        check=True,
    )

    destination = tmp_path / "extract_destination"
    archive_ops_mod.extract_tar_zst_with_progress(
        archive_path,
        destination,
        label="Extract archive",
    )

    captured = capsys.readouterr()
    assert "Extract archive" in captured.out
    assert (destination / "openpose_output" / "json" / "sample.json").exists()


@pytest.mark.skipif(
    shutil.which("tar") is None or shutil.which("zstd") is None,
    reason="tar and zstd are required for archive creation tests",
)
def test_create_tar_zst_archive_does_not_require_tar_zstd_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_dir = tmp_path / "archive_source"
    _write(source_dir / "sample.txt", "sample")
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
    assert "transferred" in captured.out


def test_create_tar_zst_archive_reports_subprocess_stderr_after_broken_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "archive_source"
    _write(source_dir / "sample.txt", "sample")
    archive_path = tmp_path / "archive.tar.zst"
    monkeypatch.setattr(archive_ops_mod, "ensure_tar_zst_create_prerequisites", lambda: None)

    def fake_popen(command: Sequence[str], **kwargs: Any) -> Any:
        stderr_file = cast(BinaryIO, kwargs["stderr"])
        if command[0] == "tar":
            return _FakeCreateTarProcess(stderr_file)
        if command[0] == "zstd":
            return _FakeCreateZstdProcess(stderr_file)
        raise AssertionError(f"Unexpected subprocess command: {command}")

    monkeypatch.setattr("text_to_sign_production.ops.archive_ops.subprocess.Popen", fake_popen)

    with pytest.raises(RuntimeError, match="zstd stderr: zstd: simulated failure") as exc_info:
        archive_ops_mod.create_tar_zst_archive(
            archive_path=archive_path,
            members=(source_dir.relative_to(tmp_path),),
            cwd=tmp_path,
            label="Archive source",
        )

    assert isinstance(exc_info.value.__cause__, BrokenPipeError)
    assert not archive_path.exists()


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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


def test_publish_colab_outputs_copies_archives_to_fixed_drive_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive_mount = tmp_path / "drive"
    drive_mount.mkdir()
    artifact_root = drive_mount / "MyDrive" / "text-to-sign-production" / "artifacts"
    local_archive = tmp_path / "local" / "sprint2_manifests_reports.tar.zst"
    _write(local_archive, "archive")

    monkeypatch.setattr(colab_workflow_mod, "COLAB_DRIVE_MOUNT_ROOT", drive_mount)
    monkeypatch.setattr(colab_workflow_mod, "COLAB_DRIVE_ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr(
        colab_workflow_mod,
        "package_sprint2_outputs",
        lambda *, splits: [local_archive],
    )

    published = colab_workflow_mod.publish_colab_outputs(splits=("train",))

    assert published == [artifact_root / local_archive.name]
    assert (artifact_root / local_archive.name).read_text(encoding="utf-8") == "archive"


def test_colab_notebook_contains_only_the_supported_fixed_workflow() -> None:
    markdown_source, code_source = _load_notebook_sources()

    expected_headings = (
        "# Sprint 2 Colab Pipeline",
        "## 1. Environment And Repository Setup",
        "## 2. Mount Google Drive",
        "## 3. Stage Fixed Colab Inputs",
        "## 4. Run Sprint 2 Pipeline Scripts",
        "## 5. Package And Publish Outputs",
    )
    for heading in expected_headings:
        assert heading in markdown_source

    required_snippets = (
        'PIPELINE_SPLITS = ["train", "val", "test"]',
        'drive.mount("/content/drive", force_remount=False)',
        "scripts/stage_colab_inputs.py",
        'env["PYTHONUNBUFFERED"] = "1"',
        'env["T2SP_PROGRESS_MODE"] = "lines"',
        "scripts/prepare_raw.py",
        "scripts/normalize_keypoints.py",
        "scripts/filter_samples.py",
        "configs/data/filter-v1.yaml",
        "scripts/export_training_manifest.py",
        "scripts/publish_colab_outputs.py",
        "/content/how2sign_downloads",
        "/content/drive/MyDrive/text-to-sign-production/artifacts/sprint2/processed-v1/",
        "train_2D_keypoints.tar.zst",
        "val_2D_keypoints.tar.zst",
        "test_2D_keypoints.tar.zst",
    )
    for snippet in required_snippets:
        assert snippet in (markdown_source + "\n" + code_source)

    removed_snippets = (
        "public_urls",
        "mounted_paths",
        "RAW_INPUT_MODE",
        "TRANSLATION_URLS",
        "KEYPOINT_ARCHIVE_URLS",
        "MOUNTED_TRANSLATION_FILES",
        "MOUNTED_KEYPOINT_SPLIT_DIRS",
        "gdown",
        "storage.local.yaml",
        "storage.example.yaml",
        ".tar.gz",
        "scripts/package_sprint2_outputs.py",
        "google.colab.files",
    )
    for snippet in removed_snippets:
        assert snippet not in (markdown_source + "\n" + code_source)
