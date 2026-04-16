"""Archive builders and inspection helpers for tests."""

from __future__ import annotations

import io
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path


def add_tar_bytes(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    archive.addfile(info, io.BytesIO(content))


def write_tar_zst_archive(
    archive_path: Path,
    build_archive: Callable[[tarfile.TarFile], None],
) -> None:
    tar_path = archive_path.with_suffix("")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as archive:
        build_archive(archive)
    subprocess.run(["zstd", "-q", "-f", "-o", str(archive_path), str(tar_path)], check=True)
    tar_path.unlink(missing_ok=True)


def list_tar_zst_members(archive_path: Path) -> str:
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
