"""Isolated archive and copy operation tests."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, BinaryIO, cast

import pytest

import text_to_sign_production.ops.archive_ops as archive_ops_mod
from tests.support.paths import write_text

pytestmark = pytest.mark.unit


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
    def __init__(self, stderr_file: BinaryIO, output_path: Path | None = None) -> None:
        self.stdin: _WritableStdin | None = _WritableStdin()
        self.stdout = None
        self._output_path = output_path

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        if self._output_path is not None:
            self._output_path.write_bytes(b"archive")
        return 0

    def terminate(self) -> None:
        return

    def kill(self) -> None:
        return


class _FakeProgressBar:
    def __init__(self, updates: list[int]) -> None:
        self._updates = updates

    def __enter__(self) -> _FakeProgressBar:
        return self

    def __exit__(self, *args: object) -> None:
        return

    def update(self, n: int = 1) -> None:
        self._updates.append(n)


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


def test_extract_tar_zst_with_progress_rejects_non_tar_zst_archives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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


def test_create_tar_zst_archive_reports_subprocess_stderr_after_broken_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "archive_source"
    write_text(source_dir / "sample.txt", "sample")
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


def test_create_tar_zst_archive_preserves_existing_archive_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "archive_source"
    write_text(source_dir / "sample.txt", "sample")
    archive_path = tmp_path / "archive.tar.zst"
    archive_path.write_bytes(b"previous archive")
    monkeypatch.setattr(archive_ops_mod, "ensure_tar_zst_create_prerequisites", lambda: None)

    def fake_popen(command: Sequence[str], **kwargs: Any) -> Any:
        stderr_file = cast(BinaryIO, kwargs["stderr"])
        if command[0] == "tar":
            return _FakeCreateTarProcess(stderr_file)
        if command[0] == "zstd":
            return _FakeCreateZstdProcess(stderr_file)
        raise AssertionError(f"Unexpected subprocess command: {command}")

    monkeypatch.setattr("text_to_sign_production.ops.archive_ops.subprocess.Popen", fake_popen)

    with pytest.raises(RuntimeError, match="zstd stderr: zstd: simulated failure"):
        archive_ops_mod.create_tar_zst_archive(
            archive_path=archive_path,
            members=(source_dir.relative_to(tmp_path),),
            cwd=tmp_path,
            label="Archive source",
        )

    assert archive_path.read_bytes() == b"previous archive"
    assert not list(tmp_path.glob(".archive.tar.zst.*.tmp"))


def test_create_tar_zst_archive_reports_known_tar_stream_total(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "archive_source"
    write_text(source_dir / "sample.txt", "sample")
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
            output_path = Path(command[command.index("-o") + 1])
            return _FakeCreateZstdSuccessProcess(stderr_file, output_path)
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
    write_text(source_dir / "sample.txt", "sample")
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
            output_path = Path(command[command.index("-o") + 1])
            return _FakeCreateZstdSuccessProcess(stderr_file, output_path)
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


def test_create_tar_zst_archive_passes_absolute_files_from_path_for_relative_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_cwd = tmp_path / "process_cwd"
    tar_cwd = tmp_path / "tar_cwd"
    process_cwd.mkdir()
    write_text(tar_cwd / "sample.txt", "sample")
    archive_path = Path("out/archive.tar.zst")
    monkeypatch.chdir(process_cwd)
    monkeypatch.setattr(archive_ops_mod, "ensure_tar_zst_create_prerequisites", lambda: None)
    files_from_paths: list[Path] = []

    def fake_stream_file_with_progress(
        src: Any,
        writer: Any,
        total_bytes: int | None,
        desc: str = "",
    ) -> int:
        writer(b"tar-stream")
        return len(b"tar-stream")

    def fake_popen(command: Sequence[str], **kwargs: Any) -> Any:
        stderr_file = cast(BinaryIO, kwargs["stderr"])
        if command[0] == "tar":
            files_from_path = Path(command[command.index("--files-from") + 1])
            files_from_paths.append(files_from_path)
            assert files_from_path.is_absolute()
            assert files_from_path.read_bytes() == b"sample.txt\0"
            assert kwargs["cwd"] == tar_cwd
            return _FakeCreateTarProcess(stderr_file)
        if command[0] == "zstd":
            output_path = Path(command[command.index("-o") + 1])
            return _FakeCreateZstdSuccessProcess(stderr_file, output_path)
        raise AssertionError(f"Unexpected subprocess command: {command}")

    monkeypatch.setattr(
        archive_ops_mod,
        "stream_file_with_progress",
        fake_stream_file_with_progress,
    )
    monkeypatch.setattr("text_to_sign_production.ops.archive_ops.subprocess.Popen", fake_popen)

    created_path = archive_ops_mod.create_tar_zst_archive(
        archive_path=archive_path,
        members=(Path("sample.txt"),),
        cwd=tar_cwd,
        label="Archive source",
    )

    assert created_path == archive_path
    assert files_from_paths
    assert not files_from_paths[0].exists()
    assert (process_cwd / archive_path).exists()


def test_create_tar_zst_archive_from_snapshot_uses_local_snapshot_and_then_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_root = tmp_path / "drive" / "run"
    write_text(live_root / "config" / "baseline.yaml", "config")
    write_text(live_root / "checkpoints" / "last.pt", "checkpoint")
    final_archive = live_root / "archives" / "baseline_training_outputs.tar.zst"
    snapshot_parent = tmp_path / "local-snapshots"
    events: list[str] = []
    progress_calls: list[dict[str, object]] = []
    progress_updates: list[int] = []
    observed_snapshot_cwd: Path | None = None

    def fake_progress_bar(
        total: int | None = None,
        desc: str = "",
        unit: str = "items",
    ) -> _FakeProgressBar:
        progress_calls.append({"total": total, "desc": desc, "unit": unit})
        return _FakeProgressBar(progress_updates)

    def fake_create_tar_zst_archive(
        *,
        archive_path: Path,
        members: Sequence[Path],
        cwd: Path,
        label: str,
    ) -> Path:
        nonlocal observed_snapshot_cwd
        assert label == "Archive training"
        assert cwd != live_root
        assert cwd.is_relative_to(snapshot_parent.resolve())
        assert tuple(members) == (Path("config"), Path("checkpoints"))
        assert (cwd / "config" / "baseline.yaml").read_text(encoding="utf-8") == "config"
        assert (cwd / "checkpoints" / "last.pt").read_text(encoding="utf-8") == "checkpoint"
        observed_snapshot_cwd = cwd
        events.append("local-create")
        archive_path.write_bytes(b"finished local archive")
        return archive_path

    def fake_copy_file_with_progress(source: Path, destination: Path, *, label: str) -> Path:
        assert events == ["local-create"]
        assert source.read_bytes() == b"finished local archive"
        assert destination == final_archive
        assert label == "Archive training (publish archive)"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        events.append("publish")
        return destination

    monkeypatch.setattr(archive_ops_mod, "ensure_tar_zst_create_prerequisites", lambda: None)
    monkeypatch.setattr(archive_ops_mod, "progress_bar", fake_progress_bar)
    monkeypatch.setattr(archive_ops_mod, "create_tar_zst_archive", fake_create_tar_zst_archive)
    monkeypatch.setattr(archive_ops_mod, "copy_file_with_progress", fake_copy_file_with_progress)

    result = archive_ops_mod.create_tar_zst_archive_from_snapshot(
        archive_path=final_archive,
        members=(live_root / "config", live_root / "checkpoints"),
        cwd=live_root,
        label="Archive training",
        snapshot_parent=snapshot_parent,
    )

    assert result == final_archive
    assert final_archive.read_bytes() == b"finished local archive"
    assert events == ["local-create", "publish"]
    assert observed_snapshot_cwd is not None
    assert not observed_snapshot_cwd.exists()
    assert progress_calls == [
        {
            "total": len("config") + len("checkpoint"),
            "desc": "Archive training (snapshot copy)",
            "unit": "B",
        }
    ]
    assert sum(progress_updates) == len("config") + len("checkpoint")


def test_create_tar_zst_archive_from_snapshot_cleans_snapshot_after_local_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_root = tmp_path / "drive" / "run"
    write_text(live_root / "qualitative" / "records.jsonl", "{}\n")
    final_archive = live_root / "archives" / "baseline_qualitative_outputs.tar.zst"
    snapshot_parent = tmp_path / "local-snapshots"

    def fake_create_tar_zst_archive(
        *,
        archive_path: Path,
        members: Sequence[Path],
        cwd: Path,
        label: str,
    ) -> Path:
        del archive_path, members, cwd, label
        raise RuntimeError("simulated local archive failure")

    def unexpected_copy_file_with_progress(source: Path, destination: Path, *, label: str) -> Path:
        del source, destination, label
        raise AssertionError("archive should not be published after local creation failure")

    monkeypatch.setattr(archive_ops_mod, "ensure_tar_zst_create_prerequisites", lambda: None)
    monkeypatch.setattr(archive_ops_mod, "create_tar_zst_archive", fake_create_tar_zst_archive)
    monkeypatch.setattr(
        archive_ops_mod,
        "copy_file_with_progress",
        unexpected_copy_file_with_progress,
    )

    with pytest.raises(RuntimeError, match="simulated local archive failure"):
        archive_ops_mod.create_tar_zst_archive_from_snapshot(
            archive_path=final_archive,
            members=(live_root / "qualitative",),
            cwd=live_root,
            label="Archive qualitative",
            snapshot_parent=snapshot_parent,
        )

    assert not final_archive.exists()
    assert list(snapshot_parent.iterdir()) == []


def test_write_tar_member_list_removes_partial_file_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingMemberList:
        def __init__(self, path: Path) -> None:
            self.name = path.as_posix()
            self._handle = path.open("wb")
            self._writes = 0

        def __enter__(self) -> _FailingMemberList:
            return self

        def __exit__(self, *args: object) -> None:
            self._handle.close()

        def write(self, data: bytes) -> int:
            self._writes += 1
            if self._writes == 2:
                raise OSError("simulated member list write failure")
            return self._handle.write(data)

    def fake_named_temporary_file(
        *,
        prefix: str,
        suffix: str,
        dir: Path,
        delete: bool,
    ) -> _FailingMemberList:
        assert delete is False
        return _FailingMemberList(Path(dir) / f"{prefix}partial{suffix}")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fake_named_temporary_file)

    with pytest.raises(OSError, match="simulated member list write failure"):
        archive_ops_mod._write_tar_member_list(
            tmp_path,
            "archive.tar.zst",
            ("sample.txt",),
        )

    assert not list(tmp_path.glob(".archive.tar.zst.members.*.txt"))


def test_create_tar_zst_archive_rejects_non_tar_zst_archives(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.tar\.zst"):
        archive_ops_mod.create_tar_zst_archive(
            archive_path=tmp_path / "archive.zip",
            members=(),
            cwd=tmp_path,
            label="Archive source",
        )


def test_create_tar_zst_archive_requires_absolute_members_under_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cwd = tmp_path / "cwd"
    outside_member = tmp_path / "outside" / "sample.txt"
    write_text(cwd / "sample.txt", "inside")
    write_text(outside_member, "outside")
    monkeypatch.setattr(archive_ops_mod, "ensure_tar_zst_create_prerequisites", lambda: None)

    with pytest.raises(ValueError, match="Archive member must be under cwd"):
        archive_ops_mod.create_tar_zst_archive(
            archive_path=cwd / "archive.tar.zst",
            members=(outside_member,),
            cwd=cwd,
            label="Archive source",
        )
