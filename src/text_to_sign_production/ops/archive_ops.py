"""Reusable archive and large-file operations for Colab and local packaging."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import BinaryIO, NoReturn

from ..data.utils import ensure_directory
from .progress import stream_file_with_progress


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


def ensure_tar_zst_create_prerequisites() -> None:
    """Validate the system tools required to create `.tar.zst` archives."""

    if shutil.which("tar") is None:
        raise RuntimeError("`tar` is required to create `.tar.zst` archives.")
    if shutil.which("zstd") is None:
        raise RuntimeError("`zstd` is required to create `.tar.zst` archives.")


def ensure_tar_zst_extract_prerequisites() -> None:
    """Validate the system tools required to extract `.tar.zst` archives."""

    ensure_tar_zst_create_prerequisites()
    if not tar_supports_zstd():
        raise RuntimeError("`tar` with `--zstd` support is required for `.tar.zst` extraction.")


def copy_file_with_progress(source: Path, destination: Path, *, label: str) -> Path:
    """Copy a file while emitting byte progress."""

    if not source.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")

    ensure_directory(destination.parent)
    total_bytes = source.stat().st_size

    with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
        stream_file_with_progress(source_handle, destination_handle, total_bytes, desc=label)
    shutil.copystat(source, destination)
    return destination


def extract_tar_zst_with_progress(archive_path: Path, destination: Path, *, label: str) -> None:
    """Extract a `.tar.zst` archive by streaming it through `tar` with progress."""

    if not archive_path.name.endswith(".tar.zst"):
        raise ValueError(f"Expected a .tar.zst archive, got: {archive_path.name}")
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive file not found: {archive_path}")

    archive_path = archive_path.resolve()
    ensure_tar_zst_extract_prerequisites()
    ensure_directory(destination)
    total_bytes = archive_path.stat().st_size
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            ["tar", "--zstd", "-xf", "-", "-C", str(destination)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
        )

        stream_error: OSError | None = None

        def write_to_tar(chunk: bytes) -> None:
            if process.stdin is None:
                raise RuntimeError(f"tar stdin was not available for {archive_path}")
            if process.poll() is not None:
                raise BrokenPipeError("tar exited before the full archive stream was written")
            process.stdin.write(chunk)

        try:
            with archive_path.open("rb") as archive_handle:
                try:
                    stream_file_with_progress(
                        archive_handle,
                        write_to_tar,
                        total_bytes,
                        desc=label,
                    )
                except OSError as exc:
                    stream_error = exc
        except BaseException:
            _close_process_stdin(process)
            _stop_process(process)
            raise

        _close_process_stdin(process)
        returncode = process.wait()
        if stream_error is not None or returncode != 0:
            stderr_text = _read_stderr_file(stderr_file)
            _raise_tar_extraction_error(
                archive_path=archive_path,
                returncode=returncode,
                stderr_text=stderr_text,
                stream_error=stream_error,
            )


def create_tar_zst_archive(
    *,
    archive_path: Path,
    members: Sequence[Path],
    cwd: Path,
    label: str,
) -> Path:
    """Create a `.tar.zst` archive while reporting streamed tar progress."""

    if not archive_path.name.endswith(".tar.zst"):
        raise ValueError(f"Expected a .tar.zst archive, got: {archive_path.name}")

    ensure_tar_zst_create_prerequisites()
    member_names = [str(_relative_member_path(member, cwd)) for member in members]
    _assert_required_members(cwd, members)
    ensure_directory(archive_path.parent)
    if archive_path.exists():
        archive_path.unlink()

    with (
        tempfile.TemporaryFile() as tar_stderr_file,
        tempfile.TemporaryFile() as zstd_stderr_file,
    ):
        tar_process = subprocess.Popen(
            ["tar", "-cf", "-", *member_names],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=tar_stderr_file,
        )
        zstd_process = subprocess.Popen(
            ["zstd", "-q", "-T0", "-o", str(archive_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=zstd_stderr_file,
        )

        stream_error: OSError | None = None

        def write_to_zstd(chunk: bytes) -> None:
            if zstd_process.stdin is None:
                raise RuntimeError("zstd stdin was not available while creating the archive.")
            if zstd_process.poll() is not None:
                raise BrokenPipeError("zstd exited before the full tar stream was written")
            zstd_process.stdin.write(chunk)

        try:
            if tar_process.stdout is None:
                raise RuntimeError("tar stdout was not available while creating the archive.")
            try:
                stream_file_with_progress(
                    tar_process.stdout,
                    write_to_zstd,
                    None,
                    desc=label,
                )
            except OSError as exc:
                stream_error = exc
        except BaseException:
            _close_process_stdin(zstd_process)
            _stop_process(tar_process)
            _stop_process(zstd_process)
            archive_path.unlink(missing_ok=True)
            raise

        _close_process_stdin(zstd_process)
        tar_returncode = (
            _stop_process(tar_process) if stream_error is not None else tar_process.wait()
        )
        zstd_returncode = zstd_process.wait()

        if stream_error is not None or tar_returncode != 0 or zstd_returncode != 0:
            tar_stderr = _read_stderr_file(tar_stderr_file)
            zstd_stderr = _read_stderr_file(zstd_stderr_file)
            archive_path.unlink(missing_ok=True)
            _raise_archive_creation_error(
                archive_path=archive_path,
                tar_returncode=tar_returncode,
                tar_stderr=tar_stderr,
                zstd_returncode=zstd_returncode,
                zstd_stderr=zstd_stderr,
                stream_error=stream_error,
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


def _read_stderr_file(stderr_file: BinaryIO) -> str:
    stderr_file.seek(0)
    return stderr_file.read().decode("utf-8", errors="replace").strip()


def _raise_tar_extraction_error(
    *,
    archive_path: Path,
    returncode: int,
    stderr_text: str,
    stream_error: OSError | None,
) -> NoReturn:
    details = [f"tar exit code {returncode}"]
    if stderr_text:
        details.append(f"tar stderr: {stderr_text}")
    if stream_error is not None:
        details.append(f"streaming error: {stream_error}")
    message = f"tar extraction failed for {archive_path}: {'; '.join(details)}"
    if stream_error is None:
        raise RuntimeError(message)
    raise RuntimeError(message) from stream_error


def _raise_archive_creation_error(
    *,
    archive_path: Path,
    tar_returncode: int,
    tar_stderr: str,
    zstd_returncode: int,
    zstd_stderr: str,
    stream_error: OSError | None,
) -> NoReturn:
    details: list[str] = []
    if stream_error is not None:
        details.append(f"streaming to zstd failed: {stream_error}")
    if tar_returncode != 0:
        details.append(f"tar exit code {tar_returncode}")
    if tar_stderr:
        details.append(f"tar stderr: {tar_stderr}")
    if zstd_returncode != 0:
        details.append(f"zstd exit code {zstd_returncode}")
    if zstd_stderr:
        details.append(f"zstd stderr: {zstd_stderr}")
    message = f"archive creation failed for {archive_path}: {'; '.join(details)}"
    if stream_error is None:
        raise RuntimeError(message)
    raise RuntimeError(message) from stream_error
