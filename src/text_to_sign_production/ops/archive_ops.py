"""Reusable archive and large-file operations for Colab and local packaging."""

from __future__ import annotations

import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import IO, BinaryIO, NoReturn

from ..data.utils import ensure_directory
from .progress import progress_bar, stream_file_with_progress

_TAR_BLOCK_SIZE = 512
_TAR_RECORD_BLOCKS = 20
_TAR_RECORD_SIZE = _TAR_BLOCK_SIZE * _TAR_RECORD_BLOCKS
_TAR_END_BLOCK_BYTES = 2 * _TAR_BLOCK_SIZE
_MAX_SIMPLE_TAR_NAME_BYTES = 100
_COPY_CHUNK_SIZE = 8 * 1024 * 1024


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

    if shutil.which("zstd") is None:
        raise RuntimeError("`zstd` is required to extract `.tar.zst` archives.")


def copy_file_with_progress(source: Path, destination: Path, *, label: str) -> Path:
    """Copy a file while emitting byte progress."""

    if not source.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")

    ensure_directory(destination.parent)
    total_bytes = source.stat().st_size
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temp_handle:
            temp_path = Path(temp_handle.name)
            with source.open("rb") as source_handle:
                stream_file_with_progress(source_handle, temp_handle, total_bytes, desc=label)
        shutil.copystat(source, temp_path)
        temp_path.replace(destination)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return destination


def extract_tar_zst_with_progress(archive_path: Path, destination: Path, *, label: str) -> None:
    """Extract a `.tar.zst` archive while validating tar members stay under destination."""

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
        with tempfile.TemporaryFile() as zstd_stderr_file:
            zstd_process = subprocess.Popen(
                ["zstd", "-q", "-dc"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=zstd_stderr_file,
            )
            stream_errors: list[BaseException] = []

            def write_to_zstd(chunk: bytes) -> None:
                if zstd_process.stdin is None:
                    raise RuntimeError(f"zstd stdin was not available for {archive_path}")
                if zstd_process.poll() is not None:
                    raise BrokenPipeError("zstd exited before the full archive stream was written")
                zstd_process.stdin.write(chunk)

            def feed_archive_to_zstd() -> None:
                with archive_path.open("rb") as archive_handle:
                    try:
                        stream_file_with_progress(
                            archive_handle,
                            write_to_zstd,
                            total_bytes,
                            desc=label,
                        )
                    except BaseException as exc:
                        stream_errors.append(exc)
                    finally:
                        _close_process_stdin(zstd_process)

            if zstd_process.stdout is None:
                _close_process_stdin(zstd_process)
                _stop_process(zstd_process)
                raise RuntimeError(f"zstd stdout was not available for {archive_path}")

            feed_thread = threading.Thread(
                target=feed_archive_to_zstd,
                name=f"feed-{archive_path.name}-to-zstd",
            )
            extraction_error: BaseException | None = None
            feed_thread.start()
            try:
                _extract_tar_stream_safely(zstd_process.stdout, temp_destination)
            except BaseException as exc:
                extraction_error = exc
                _close_process_stdin(zstd_process)
                zstd_returncode = _stop_process(zstd_process)
            else:
                zstd_returncode = zstd_process.wait()
            finally:
                feed_thread.join()

            stream_error = stream_errors[0] if stream_errors else None
            if isinstance(extraction_error, ValueError):
                raise extraction_error
            if extraction_error is not None or stream_error is not None or zstd_returncode != 0:
                zstd_stderr = _read_stderr_file(zstd_stderr_file)
                _raise_zstd_extraction_error(
                    archive_path=archive_path,
                    zstd_returncode=zstd_returncode,
                    zstd_stderr=zstd_stderr,
                    stream_error=stream_error,
                    extraction_error=extraction_error,
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
    temp_archive_path = _reserve_temp_archive_path(archive_path)
    member_list_path: Path | None = None
    archive_committed = False
    try:
        member_list_path = _write_tar_member_list(
            archive_path.parent,
            archive_path.name,
            member_names,
        )
        with (
            tempfile.TemporaryFile() as tar_stderr_file,
            tempfile.TemporaryFile() as zstd_stderr_file,
        ):
            tar_process = subprocess.Popen(
                ["tar", "-cf", "-", "--null", "--files-from", str(member_list_path)],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=tar_stderr_file,
            )
            zstd_process = subprocess.Popen(
                ["zstd", "-q", "-T0", "-o", str(temp_archive_path)],
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
                raise

            _close_process_stdin(zstd_process)
            tar_returncode = (
                _stop_process(tar_process) if stream_error is not None else tar_process.wait()
            )
            zstd_returncode = zstd_process.wait()

            if stream_error is not None or tar_returncode != 0 or zstd_returncode != 0:
                tar_stderr = _read_stderr_file(tar_stderr_file)
                zstd_stderr = _read_stderr_file(zstd_stderr_file)
                _raise_archive_creation_error(
                    archive_path=archive_path,
                    tar_returncode=tar_returncode,
                    tar_stderr=tar_stderr,
                    zstd_returncode=zstd_returncode,
                    zstd_stderr=zstd_stderr,
                    stream_error=stream_error,
                )
        temp_archive_path.replace(archive_path)
        archive_committed = True
    finally:
        if not archive_committed:
            temp_archive_path.unlink(missing_ok=True)
        if member_list_path is not None:
            member_list_path.unlink(missing_ok=True)
    return archive_path


def create_tar_zst_archive_from_snapshot(
    *,
    archive_path: Path,
    members: Sequence[Path],
    cwd: Path,
    label: str,
    snapshot_parent: Path | None = None,
) -> Path:
    """Create a `.tar.zst` from a stable local snapshot, then publish the finished archive."""

    if not archive_path.name.endswith(".tar.zst"):
        raise ValueError(f"Expected a .tar.zst archive, got: {archive_path.name}")

    ensure_tar_zst_create_prerequisites()
    member_names = [str(_relative_member_path(member, cwd)) for member in members]
    _assert_required_members(cwd, members)
    local_snapshot_parent = _resolve_snapshot_parent(snapshot_parent)
    ensure_directory(local_snapshot_parent)

    with tempfile.TemporaryDirectory(
        prefix=f".{archive_path.name}.snapshot-",
        dir=local_snapshot_parent,
    ) as temp_root:
        temp_root_path = Path(temp_root)
        snapshot_cwd = temp_root_path / "snapshot"
        snapshot_cwd.mkdir()
        _copy_members_to_snapshot(
            cwd=cwd,
            member_names=member_names,
            snapshot_cwd=snapshot_cwd,
            label=f"{label} (snapshot copy)",
        )
        local_archive_path = temp_root_path / archive_path.name
        create_tar_zst_archive(
            archive_path=local_archive_path,
            members=tuple(Path(member_name) for member_name in member_names),
            cwd=snapshot_cwd,
            label=label,
        )
        return copy_file_with_progress(
            local_archive_path,
            archive_path,
            label=f"{label} (publish archive)",
        )


def copy_archive_to_drive(source: Path, destination: Path, *, label: str) -> Path:
    """Copy a packaged archive to Drive with progress output."""

    return copy_file_with_progress(source, destination, label=label)


def _reserve_temp_archive_path(archive_path: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f".{archive_path.name}.",
        suffix=".tmp",
        dir=archive_path.parent,
        delete=False,
    ) as temp_handle:
        temp_archive_path = Path(temp_handle.name)
    temp_archive_path.unlink()
    return temp_archive_path


def _write_tar_member_list(root: Path, archive_name: str, member_names: Sequence[str]) -> Path:
    member_list_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{archive_name}.members.",
            suffix=".txt",
            dir=root,
            delete=False,
        ) as member_list:
            member_list_path = Path(member_list.name).resolve()
            for member_name in member_names:
                member_list.write(member_name.encode("utf-8"))
                member_list.write(b"\0")
        return member_list_path
    except BaseException:
        if member_list_path is not None:
            member_list_path.unlink(missing_ok=True)
        raise


def _resolve_snapshot_parent(snapshot_parent: Path | None) -> Path:
    if snapshot_parent is not None:
        return snapshot_parent.expanduser().resolve()
    return Path(tempfile.gettempdir()).resolve()


def _copy_members_to_snapshot(
    *,
    cwd: Path,
    member_names: Sequence[str],
    snapshot_cwd: Path,
    label: str,
) -> None:
    total_bytes = _snapshot_members_total_bytes(cwd, member_names)
    with progress_bar(total=total_bytes, desc=label, unit="B") as bar:
        for member_name in member_names:
            _copy_snapshot_member(
                cwd / member_name,
                snapshot_cwd / member_name,
                progress_update=bar.update,
            )


def _copy_snapshot_member(
    source: Path,
    destination: Path,
    *,
    progress_update: Callable[[int], object],
) -> None:
    source_stat = source.lstat()
    if stat.S_ISDIR(source_stat.st_mode):
        destination.mkdir(parents=True, exist_ok=True)
        for child in sorted(source.iterdir(), key=lambda path: path.name):
            _copy_snapshot_member(
                child,
                destination / child.name,
                progress_update=progress_update,
            )
        shutil.copystat(source, destination, follow_symlinks=False)
        return
    if stat.S_ISREG(source_stat.st_mode):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
            while chunk := source_handle.read(_COPY_CHUNK_SIZE):
                destination_handle.write(chunk)
                progress_update(len(chunk))
        shutil.copystat(source, destination, follow_symlinks=False)
        return
    raise ValueError(f"Unsupported snapshot archive member type: {source}")


def _snapshot_members_total_bytes(cwd: Path, member_names: Sequence[str]) -> int:
    return sum(_snapshot_member_total_bytes(cwd / member_name) for member_name in member_names)


def _snapshot_member_total_bytes(path: Path) -> int:
    path_stat = path.lstat()
    if stat.S_ISDIR(path_stat.st_mode):
        return sum(_snapshot_member_total_bytes(child) for child in path.iterdir())
    if stat.S_ISREG(path_stat.st_mode):
        return path_stat.st_size
    raise ValueError(f"Unsupported snapshot archive member type: {path}")


def _assert_required_members(cwd: Path, members: Iterable[Path]) -> None:
    missing = [member for member in members if not _absolute_member_path(member, cwd).exists()]
    if missing:
        formatted = "\n".join(f"- {member}" for member in missing)
        raise FileNotFoundError(f"Missing required Dataset Build outputs:\n{formatted}")


def _absolute_member_path(member: Path, cwd: Path) -> Path:
    return member if member.is_absolute() else cwd / member


def _relative_member_path(member: Path, cwd: Path) -> Path:
    resolved_cwd = cwd.resolve()
    resolved_member = _absolute_member_path(member, cwd).resolve()
    if not resolved_member.is_relative_to(resolved_cwd):
        raise ValueError(f"Archive member must be under cwd {resolved_cwd}: {member}")
    return resolved_member.relative_to(resolved_cwd)


def _extract_tar_stream_safely(tar_stream: IO[bytes], destination: Path) -> None:
    resolved_destination = destination.resolve()
    with tarfile.open(fileobj=tar_stream, mode="r|") as archive:
        for member in archive:
            target_path = _validated_tar_member_path(member, resolved_destination)
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            if member.isfile():
                source_file = archive.extractfile(member)
                if source_file is None:
                    raise RuntimeError(f"Could not read tar member payload: {member.name}")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with target_path.open("wb") as target_handle:
                        shutil.copyfileobj(source_file, target_handle)
                finally:
                    source_file.close()
                continue
            raise ValueError(f"Unsupported tar member type for {member.name!r}")


def _validated_tar_member_path(member: tarfile.TarInfo, destination: Path) -> Path:
    member_path = PurePosixPath(member.name)
    if member.name == "" or member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"Unsafe tar member path: {member.name!r}")

    target_path = destination.joinpath(*member_path.parts).resolve()
    if target_path != destination and not target_path.is_relative_to(destination):
        raise ValueError(f"Unsafe tar member path: {member.name!r}")
    if member.issym() or member.islnk():
        raise ValueError(f"Unsupported tar member type for {member.name!r}: links are not allowed")
    if not member.isdir() and not member.isfile():
        raise ValueError(f"Unsupported tar member type for {member.name!r}")
    return target_path


def _tar_stream_total_bytes(cwd: Path, member_names: Sequence[str]) -> int | None:
    """Return a known tar stream size, or None only when tar encoding can change it."""

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


def _raise_zstd_extraction_error(
    *,
    archive_path: Path,
    zstd_returncode: int,
    zstd_stderr: str,
    stream_error: BaseException | None,
    extraction_error: BaseException | None,
) -> NoReturn:
    details = [f"zstd exit code {zstd_returncode}"]
    if zstd_stderr:
        details.append(f"zstd stderr: {zstd_stderr}")
    if stream_error is not None:
        details.append(f"streaming error: {stream_error}")
    if extraction_error is not None:
        details.append(f"tar stream error: {extraction_error}")
    message = f"tar.zst extraction failed for {archive_path}: {'; '.join(details)}"
    cause = extraction_error if extraction_error is not None else stream_error
    if cause is None:
        raise RuntimeError(message)
    raise RuntimeError(message) from cause


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
