"""Reusable archive and large-file operations for Colab and local packaging."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

from ..data.utils import ensure_directory
from .progress import ProgressReporter

CHUNK_SIZE = 8 * 1024 * 1024


def tar_supports_zstd() -> bool:
    """Return whether the installed `tar` advertises `--zstd` support."""

    tar_path = shutil.which("tar")
    if tar_path is None:
        return False
    help_result = subprocess.run(
        [tar_path, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    return "--zstd" in (help_result.stdout + help_result.stderr)


def ensure_tar_zst_prerequisites() -> None:
    """Validate the system tools required for `.tar.zst` operations."""

    if shutil.which("tar") is None:
        raise RuntimeError("`tar` is required for `.tar.zst` archive operations.")
    if shutil.which("zstd") is None:
        raise RuntimeError("`zstd` is required for `.tar.zst` archive operations.")
    if not tar_supports_zstd():
        raise RuntimeError("`tar` with `--zstd` support is required for `.tar.zst` extraction.")


def copy_file_with_progress(source: Path, destination: Path, *, label: str) -> Path:
    """Copy a file while emitting byte progress."""

    if not source.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")

    ensure_directory(destination.parent)
    total_bytes = source.stat().st_size
    reporter = ProgressReporter(label=label, total_bytes=total_bytes)
    completed = 0

    with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
        while chunk := source_handle.read(CHUNK_SIZE):
            destination_handle.write(chunk)
            completed += len(chunk)
            reporter.update(completed)
    shutil.copystat(source, destination)
    reporter.finish(total_bytes)
    return destination


def extract_tar_zst_with_progress(archive_path: Path, destination: Path, *, label: str) -> None:
    """Extract a `.tar.zst` archive by streaming it through `tar` with progress."""

    ensure_tar_zst_prerequisites()
    if not archive_path.name.endswith(".tar.zst"):
        raise ValueError(f"Expected a .tar.zst archive, got: {archive_path.name}")

    ensure_directory(destination)
    total_bytes = archive_path.stat().st_size
    reporter = ProgressReporter(label=label, total_bytes=total_bytes)
    process = subprocess.Popen(
        ["tar", "--zstd", "-xf", "-", "-C", str(destination)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    completed = 0
    try:
        with archive_path.open("rb") as archive_handle:
            while chunk := archive_handle.read(CHUNK_SIZE):
                if process.stdin is None:
                    raise RuntimeError(f"tar stdin was not available for {archive_path}")
                process.stdin.write(chunk)
                completed += len(chunk)
                reporter.update(completed)
    except BaseException:
        _close_process_stdin(process)
        _stop_process(process)
        reporter.finish(completed)
        raise

    _close_process_stdin(process)
    stderr_text = _read_process_stderr(process)
    returncode = process.wait()
    reporter.finish(total_bytes)
    if returncode != 0:
        raise RuntimeError(
            f"tar extraction failed for {archive_path} with exit code {returncode}: {stderr_text}"
        )


def create_tar_zst_archive(
    *,
    archive_path: Path,
    members: Sequence[Path],
    cwd: Path,
    label: str,
) -> Path:
    """Create a `.tar.zst` archive while reporting streamed tar progress."""

    ensure_tar_zst_prerequisites()
    member_names = [str(_relative_member_path(member, cwd)) for member in members]
    _assert_required_members(cwd, members)
    ensure_directory(archive_path.parent)
    if archive_path.exists():
        archive_path.unlink()

    total_tar_bytes = _estimate_tar_stream_size(cwd, members)
    reporter = ProgressReporter(label=label, total_bytes=total_tar_bytes)
    tar_process = subprocess.Popen(
        ["tar", "-cf", "-", *member_names],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    zstd_process = subprocess.Popen(
        ["zstd", "-q", "-T0", "-o", str(archive_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    completed = 0
    try:
        if tar_process.stdout is None:
            raise RuntimeError("tar stdout was not available while creating the archive.")
        while chunk := tar_process.stdout.read(CHUNK_SIZE):
            if zstd_process.stdin is None:
                raise RuntimeError("zstd stdin was not available while creating the archive.")
            zstd_process.stdin.write(chunk)
            completed += len(chunk)
            reporter.update(completed)
    except BaseException:
        _close_process_stdin(zstd_process)
        _stop_process(tar_process)
        _stop_process(zstd_process)
        archive_path.unlink(missing_ok=True)
        reporter.finish(completed)
        raise

    _close_process_stdin(zstd_process)
    tar_stderr = _read_process_stderr(tar_process)
    tar_returncode = tar_process.wait()
    zstd_stderr = _read_process_stderr(zstd_process)
    zstd_returncode = zstd_process.wait()
    reporter.finish(total_tar_bytes)

    if tar_returncode != 0:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            "tar failed while creating "
            f"{archive_path} with exit code {tar_returncode}: {tar_stderr}"
        )
    if zstd_returncode != 0:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            "zstd failed while creating "
            f"{archive_path} with exit code {zstd_returncode}: {zstd_stderr}"
        )
    return archive_path


def copy_archive_to_drive(source: Path, destination: Path, *, label: str) -> Path:
    """Copy a packaged archive to Drive with progress output."""

    return copy_file_with_progress(source, destination, label=label)


def _assert_required_members(cwd: Path, members: Iterable[Path]) -> None:
    missing = [member for member in members if not _absolute_member_path(member, cwd).exists()]
    if missing:
        formatted = "\n".join(f"- {member}" for member in missing)
        raise FileNotFoundError(f"Missing required Sprint 2 outputs:\n{formatted}")


def _absolute_member_path(member: Path, cwd: Path) -> Path:
    return member if member.is_absolute() else cwd / member


def _relative_member_path(member: Path, cwd: Path) -> Path:
    if not member.is_absolute():
        return member
    return member.resolve().relative_to(cwd.resolve())


def _estimate_tar_stream_size(cwd: Path, members: Sequence[Path]) -> int:
    total = 1024
    for member in members:
        total += _estimate_member_size(_absolute_member_path(member, cwd))
    return total


def _estimate_member_size(path: Path) -> int:
    total = _estimate_entry_size(path)
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            total += _estimate_entry_size(child)
    return total


def _estimate_entry_size(path: Path) -> int:
    header_size = 512
    if not path.is_file():
        return header_size
    payload_size = path.stat().st_size
    padded_payload_size = ((payload_size + 511) // 512) * 512
    return header_size + padded_payload_size


def _close_process_stdin(process: subprocess.Popen[bytes]) -> None:
    if process.stdin is None:
        return
    try:
        process.stdin.close()
    except BrokenPipeError:
        pass


def _stop_process(process: subprocess.Popen[bytes]) -> int:
    if process.poll() is None:
        process.terminate()
        try:
            return process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    return process.wait()


def _read_process_stderr(process: subprocess.Popen[bytes]) -> str:
    if process.stderr is None:
        return ""
    raw = process.stderr.read()
    return raw.decode("utf-8", errors="replace").strip()
