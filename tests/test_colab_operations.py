"""Operational support tests for the simplified Sprint 2 Colab workflow."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, BinaryIO, cast

import pytest

import scripts.package_sprint2_outputs as package_outputs_script
import text_to_sign_production.ops.archive_ops as archive_ops_mod
import text_to_sign_production.ops.colab_workflow as colab_workflow_mod
import text_to_sign_production.ops.progress as progress_mod


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _add_tar_bytes(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    archive.addfile(info, io.BytesIO(content))


def _write_tar_zst_archive(
    archive_path: Path,
    build_archive: Callable[[tarfile.TarFile], None],
) -> None:
    tar_path = archive_path.with_suffix("")
    with tarfile.open(tar_path, "w") as archive:
        build_archive(archive)
    subprocess.run(["zstd", "-q", "-f", "-o", str(archive_path), str(tar_path)], check=True)


def _list_tar_zst_members(archive_path: Path) -> str:
    decompressed = subprocess.run(
        ["zstd", "-dc", str(archive_path)],
        check=True,
        capture_output=True,
    ).stdout
    return subprocess.run(
        ["tar", "-tf", "-"],
        input=decompressed,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")


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


class _WritableStdin:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, chunk: bytes) -> int:
        self.chunks.append(chunk)
        return len(chunk)

    def close(self) -> None:
        return


class _FakeCreateZstdSuccessProcess:
    def __init__(self, stderr_file: BinaryIO) -> None:
        self.stdin: _WritableStdin | None = _WritableStdin()
        self.stdout = None

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def terminate(self) -> None:
        return

    def kill(self) -> None:
        return


def test_iter_with_progress_yields_items_and_labels_progress(
    capsys: pytest.CaptureFixture[str],
) -> None:
    items = list(
        progress_mod.iter_with_progress([1, 2, 3], total=3, desc="Progress items", unit="items")
    )

    captured = capsys.readouterr()
    assert items == [1, 2, 3]
    assert "Progress items" in captured.out


def test_progress_bar_supports_manual_updates(capsys: pytest.CaptureFixture[str]) -> None:
    with progress_mod.progress_bar(total=2, desc="Manual progress", unit="items") as bar:
        bar.update(1)
        bar.update(1)

    captured = capsys.readouterr()
    assert "Manual progress" in captured.out


def test_stream_file_with_progress_streams_bytes_to_callback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    chunks: list[bytes] = []

    completed = progress_mod.stream_file_with_progress(
        io.BytesIO(b"hello world"),
        chunks.append,
        total_bytes=11,
        desc="Stream bytes",
    )

    captured = capsys.readouterr()
    assert completed == 11
    assert b"".join(chunks) == b"hello world"
    assert "Stream bytes" in captured.out


def test_copy_file_with_progress_replaces_destination_after_success(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "artifact.bin"
    source.write_bytes(b"new artifact")
    destination.write_bytes(b"old artifact")

    copied_path = archive_ops_mod.copy_file_with_progress(
        source,
        destination,
        label="Copy artifact",
    )

    assert copied_path == destination
    assert destination.read_bytes() == b"new artifact"
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


def test_copy_file_with_progress_preserves_destination_after_stream_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "artifact.bin"
    source.write_bytes(b"new artifact")
    destination.write_bytes(b"old artifact")

    def fake_stream_file_with_progress(
        src: Any,
        writer: Any,
        total_bytes: int | None,
        desc: str = "",
    ) -> int:
        writer.write(b"partial")
        raise OSError("simulated copy failure")

    monkeypatch.setattr(
        archive_ops_mod,
        "stream_file_with_progress",
        fake_stream_file_with_progress,
    )

    with pytest.raises(OSError, match="simulated copy failure"):
        archive_ops_mod.copy_file_with_progress(source, destination, label="Copy artifact")

    assert destination.read_bytes() == b"old artifact"
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


@pytest.mark.skipif(
    shutil.which("tar") is None or shutil.which("zstd") is None,
    reason="tar and zstd are required for archive tests",
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
    archives = package_outputs_script.package_outputs(project_root=tmp_path)

    expected_names = {
        "sprint2_manifests_reports.tar.zst",
        "sprint2_samples_train.tar.zst",
        "sprint2_samples_val.tar.zst",
        "sprint2_samples_test.tar.zst",
    }
    assert {archive.name for archive in archives} == expected_names
    assert all(archive.exists() for archive in archives)

    manifests_archive = output_dir / "sprint2_manifests_reports.tar.zst"
    manifests_listing = _list_tar_zst_members(manifests_archive)
    assert "data/interim/raw_manifests/raw_train.jsonl" in manifests_listing
    assert "data/processed/v1/reports/data-quality-report.md" in manifests_listing

    train_archive = output_dir / "sprint2_samples_train.tar.zst"
    train_listing = _list_tar_zst_members(train_archive)
    assert "data/processed/v1/samples/train/train_sample.npz" in train_listing


@pytest.mark.skipif(
    shutil.which("tar") is None or shutil.which("zstd") is None,
    reason="tar and zstd are required for archive tests",
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
    shutil.which("tar") is None or shutil.which("zstd") is None,
    reason="tar and zstd are required for archive tests",
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

    archives = package_outputs_script.package_outputs(project_root=tmp_path, splits=("train",))

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


@pytest.mark.skipif(
    shutil.which("zstd") is None,
    reason="zstd is required for extraction tests",
)
def test_extract_tar_zst_with_progress_reports_zstd_stderr_for_invalid_archive(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "corrupt.tar.zst"
    archive_path.write_bytes(b"not-a-valid-archive")
    destination = tmp_path / "extract"
    preserved_file = destination / "existing.txt"
    _write(preserved_file, "keep me")

    with pytest.raises(RuntimeError, match="zstd stderr") as exc_info:
        archive_ops_mod.extract_tar_zst_with_progress(
            archive_path,
            destination,
            label="Extract archive",
        )

    assert isinstance(exc_info.value.__cause__, tarfile.TarError)
    assert preserved_file.read_text(encoding="utf-8") == "keep me"
    assert not list(tmp_path.glob(".extract.extract-*"))


@pytest.mark.skipif(
    shutil.which("zstd") is None,
    reason="zstd is required for extraction tests",
)
def test_extract_tar_zst_with_progress_extracts_archive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive_path = tmp_path / "sample.tar.zst"

    def build_archive(archive: tarfile.TarFile) -> None:
        _add_tar_bytes(archive, "openpose_output/json/sample.json", b"{}")

    _write_tar_zst_archive(archive_path, build_archive)

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
    shutil.which("zstd") is None,
    reason="zstd is required for extraction tests",
)
def test_extract_tar_zst_with_progress_rejects_path_traversal_members(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "traversal.tar.zst"

    def build_archive(archive: tarfile.TarFile) -> None:
        _add_tar_bytes(archive, "../escape.txt", b"escape")

    _write_tar_zst_archive(archive_path, build_archive)

    with pytest.raises(ValueError, match="Unsafe tar member path"):
        archive_ops_mod.extract_tar_zst_with_progress(
            archive_path,
            tmp_path / "extract",
            label="Extract archive",
        )

    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "extract").exists()
    assert not list(tmp_path.glob(".extract.extract-*"))


@pytest.mark.skipif(
    shutil.which("zstd") is None,
    reason="zstd is required for extraction tests",
)
def test_extract_tar_zst_with_progress_rejects_symlink_members(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "symlink.tar.zst"

    def build_archive(archive: tarfile.TarFile) -> None:
        info = tarfile.TarInfo("openpose_output/json/link.json")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../escape.txt"
        archive.addfile(info)

    _write_tar_zst_archive(archive_path, build_archive)

    with pytest.raises(ValueError, match="links are not allowed"):
        archive_ops_mod.extract_tar_zst_with_progress(
            archive_path,
            tmp_path / "extract",
            label="Extract archive",
        )

    assert not (tmp_path / "extract").exists()
    assert not list(tmp_path.glob(".extract.extract-*"))


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


def test_create_tar_zst_archive_reports_known_tar_stream_total(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "archive_source"
    _write(source_dir / "sample.txt", "sample")
    archive_path = tmp_path / "archive.tar.zst"
    monkeypatch.setattr(archive_ops_mod, "ensure_tar_zst_create_prerequisites", lambda: None)
    stream_calls: list[dict[str, Any]] = []

    def fake_stream_file_with_progress(
        src: Any,
        writer: Any,
        total_bytes: int | None,
        desc: str = "",
    ) -> int:
        stream_calls.append({"total_bytes": total_bytes, "desc": desc})
        writer(b"tar-stream")
        return len(b"tar-stream")

    def fake_popen(command: Sequence[str], **kwargs: Any) -> Any:
        stderr_file = cast(BinaryIO, kwargs["stderr"])
        if command[0] == "tar":
            return _FakeCreateTarProcess(stderr_file)
        if command[0] == "zstd":
            return _FakeCreateZstdSuccessProcess(stderr_file)
        raise AssertionError(f"Unexpected subprocess command: {command}")

    monkeypatch.setattr(
        archive_ops_mod,
        "stream_file_with_progress",
        fake_stream_file_with_progress,
    )
    monkeypatch.setattr("text_to_sign_production.ops.archive_ops.subprocess.Popen", fake_popen)

    created_path = archive_ops_mod.create_tar_zst_archive(
        archive_path=archive_path,
        members=(source_dir.relative_to(tmp_path),),
        cwd=tmp_path,
        label="Archive source",
    )

    assert created_path == archive_path
    assert stream_calls == [{"total_bytes": 10240, "desc": "Archive source (tar stream)"}]


def test_create_tar_zst_archive_keeps_unknown_total_for_unmodeled_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "archive_source"
    _write(source_dir / "sample.txt", "sample")
    try:
        (source_dir / "sample-link.txt").symlink_to("sample.txt")
    except OSError as exc:
        pytest.skip(f"Could not create symlink fixture: {exc}")

    archive_path = tmp_path / "archive.tar.zst"
    monkeypatch.setattr(archive_ops_mod, "ensure_tar_zst_create_prerequisites", lambda: None)
    stream_calls: list[dict[str, Any]] = []

    def fake_stream_file_with_progress(
        src: Any,
        writer: Any,
        total_bytes: int | None,
        desc: str = "",
    ) -> int:
        stream_calls.append({"total_bytes": total_bytes, "desc": desc})
        writer(b"tar-stream")
        return len(b"tar-stream")

    def fake_popen(command: Sequence[str], **kwargs: Any) -> Any:
        stderr_file = cast(BinaryIO, kwargs["stderr"])
        if command[0] == "tar":
            return _FakeCreateTarProcess(stderr_file)
        if command[0] == "zstd":
            return _FakeCreateZstdSuccessProcess(stderr_file)
        raise AssertionError(f"Unexpected subprocess command: {command}")

    monkeypatch.setattr(
        archive_ops_mod,
        "stream_file_with_progress",
        fake_stream_file_with_progress,
    )
    monkeypatch.setattr("text_to_sign_production.ops.archive_ops.subprocess.Popen", fake_popen)

    created_path = archive_ops_mod.create_tar_zst_archive(
        archive_path=archive_path,
        members=(source_dir.relative_to(tmp_path),),
        cwd=tmp_path,
        label="Archive source",
    )

    assert created_path == archive_path
    assert stream_calls == [{"total_bytes": None, "desc": "Archive source (tar stream)"}]


def test_create_tar_zst_archive_rejects_non_tar_zst_archives(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.tar\.zst"):
        archive_ops_mod.create_tar_zst_archive(
            archive_path=tmp_path / "archive.zip",
            members=(),
            cwd=tmp_path,
            label="Archive source",
        )


def test_create_tar_zst_archive_requires_absolute_members_under_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other_root = tmp_path.parent / f"{tmp_path.name}_other"
    outside_member = other_root / "sample.txt"
    _write(outside_member, "outside")
    monkeypatch.setattr(archive_ops_mod, "ensure_tar_zst_create_prerequisites", lambda: None)

    with pytest.raises(ValueError, match="Archive member must be under cwd"):
        archive_ops_mod.create_tar_zst_archive(
            archive_path=tmp_path / "archive.tar.zst",
            members=(outside_member,),
            cwd=tmp_path,
            label="Archive source",
        )


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


def test_stage_colab_inputs_prevalidates_all_inputs_before_resetting_raw_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    _write(preserved_translation, "old translation")
    _write(preserved_keypoints, "{}")

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    translations_dir = tmp_path / "repo" / "data" / "raw" / "how2sign" / "translations"
    keypoints_root = tmp_path / "repo" / "data" / "raw" / "how2sign" / "bfh_keypoints"
    preserved_translation = translations_dir / "previous.csv"
    preserved_keypoints = keypoints_root / "previous" / "openpose_output" / "json" / "frame.json"
    _write(preserved_translation, "old translation")
    _write(preserved_keypoints, "{}")

    monkeypatch.setattr(colab_workflow_mod, "TRANSLATIONS_DIR", translations_dir)
    monkeypatch.setattr(colab_workflow_mod, "BFH_KEYPOINTS_ROOT", keypoints_root)

    with pytest.raises(ValueError, match="At least one split"):
        colab_workflow_mod.stage_colab_inputs(splits=(), download_root=tmp_path / "downloads")

    assert preserved_translation.read_text(encoding="utf-8") == "old translation"
    assert preserved_keypoints.read_text(encoding="utf-8") == "{}"


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
        "## 4. Run Sprint 2 Pipeline Functions",
        "## 5. Package And Publish Outputs",
    )
    for heading in expected_headings:
        assert heading in markdown_source

    required_snippets = (
        'PIPELINE_SPLITS = ["train", "val", "test"]',
        'drive.mount("/content/drive", force_remount=False)',
        'if str(WORKTREE_ROOT / "src") not in sys.path:',
        'sys.path.insert(0, str(WORKTREE_ROOT / "src"))',
        "from text_to_sign_production.ops.colab_workflow import stage_colab_inputs",
        "from text_to_sign_production.data.raw import build_raw_manifests",
        "from text_to_sign_production.data.normalize import normalize_all_splits",
        "from text_to_sign_production.data.filtering import filter_all_splits",
        "from text_to_sign_production.data.manifests import export_final_manifests",
        "from text_to_sign_production.ops.colab_workflow import publish_colab_outputs",
        "stage_colab_inputs(splits=tuple(PIPELINE_SPLITS))",
        "build_raw_manifests(splits=tuple(PIPELINE_SPLITS))",
        "normalize_all_splits(splits=tuple(PIPELINE_SPLITS))",
        "filter_all_splits(",
        'config_path=WORKTREE_ROOT / "configs/data/filter-v1.yaml"',
        "assumption_report = build_raw_manifests(splits=tuple(PIPELINE_SPLITS))",
        "filter_report = filter_all_splits(",
        "export_final_manifests(",
        "assumption_report=assumption_report",
        "filter_report=filter_report",
        "publish_colab_outputs(splits=tuple(PIPELINE_SPLITS))",
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
        "T2SP_PROGRESS_MODE",
        ".tar.gz",
        "scripts/package_sprint2_outputs.py",
        "google.colab.files",
    )
    for snippet in removed_snippets:
        assert snippet not in (markdown_source + "\n" + code_source)
