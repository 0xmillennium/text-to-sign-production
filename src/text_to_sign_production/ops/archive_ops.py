"""Reusable archive and large-file operations for Colab and local packaging."""

from __future__ import annotations

import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import BinaryIO, NoReturn

from ..data.utils import ensure_directory
from .progress import stream_file_with_progress

_TAR_BLOCK_SIZE = 512
_TAR_RECORD_BLOCKS = 20
_TAR_RECORD_SIZE = _TAR_BLOCK_SIZE * _TAR_RECORD_BLOCKS
_TAR_END_BLOCK_BYTES = 2 * _TAR_BLOCK_SIZE
_MAX_SIMPLE_TAR_NAME_BYTES = 100


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
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(
            f"Extraction destination exists and is not a directory: {destination}"
        )
    ensure_directory(destination.parent)
    temp_destination = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.extract-",
            dir=destination.parent,
        )
    )
    total_bytes = archive_path.stat().st_size
    extraction_committed = False
    try:
        with tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                ["tar", "--zstd", "-xf", "-", "-C", str(temp_destination)],
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

        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(temp_destination), str(destination))
        extraction_committed = True
    finally:
        if not extraction_committed and temp_destination.exists():
            shutil.rmtree(temp_destination)


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
    tar_stream_total_bytes = _tar_stream_total_bytes(cwd, member_names)
    progress_label = f"{label} (tar stream)"
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
                    tar_stream_total_bytes,
                    desc=progress_label,
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
    resolved_cwd = cwd.resolve()
    resolved_member = _absolute_member_path(member, cwd).resolve()
    if not resolved_member.is_relative_to(resolved_cwd):
        raise ValueError(f"Archive member must be under cwd {resolved_cwd}: {member}")
    return resolved_member.relative_to(resolved_cwd)


def _tar_stream_total_bytes(cwd: Path, member_names: Sequence[str]) -> int | None:
    total_bytes = 0
    for member_name in member_names:
        member_total = _tar_member_stream_bytes(cwd / member_name, member_name)
        if member_total is None:
            return None
        total_bytes += member_total
    return _round_up(total_bytes + _TAR_END_BLOCK_BYTES, _TAR_RECORD_SIZE)


def _tar_member_stream_bytes(path: Path, archive_name: str) -> int | None:
    try:
        path_stat = path.lstat()
    except OSError:
        return None

    if stat.S_ISDIR(path_stat.st_mode):
        header_name = _tar_directory_name(archive_name)
        if not _is_simple_tar_header_name(header_name):
            return None

        total_bytes = _TAR_BLOCK_SIZE
        try:
            child_names = sorted(child.name for child in path.iterdir())
        except OSError:
            return None
        for child_name in child_names:
            child_total = _tar_member_stream_bytes(
                path / child_name,
                _join_tar_name(archive_name, child_name),
            )
            if child_total is None:
                return None
            total_bytes += child_total
        return total_bytes

    if stat.S_ISREG(path_stat.st_mode):
        if path_stat.st_nlink > 1 or not _is_simple_tar_header_name(archive_name):
            return None
        return _TAR_BLOCK_SIZE + _round_up(path_stat.st_size, _TAR_BLOCK_SIZE)

    return None


def _tar_directory_name(archive_name: str) -> str:
    return archive_name if archive_name.endswith("/") else f"{archive_name}/"


def _join_tar_name(parent_name: str, child_name: str) -> str:
    parent = parent_name.rstrip("/")
    if parent == "":
        return child_name
    return f"{parent}/{child_name}"


def _is_simple_tar_header_name(archive_name: str) -> bool:
    try:
        encoded_name = archive_name.encode("ascii")
    except UnicodeEncodeError:
        return False
    return 0 < len(encoded_name) <= _MAX_SIMPLE_TAR_NAME_BYTES


def _round_up(value: int, block_size: int) -> int:
    remainder = value % block_size
    if remainder == 0:
        return value
    return value + block_size - remainder


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
