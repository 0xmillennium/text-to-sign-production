"""Tests for generic core filesystem contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sign_production.core.files import (
    OutputExistsPolicy,
    ensure_dir,
    prepare_output_dir,
    prepare_output_file,
    require_dir,
    require_dir_contains,
    require_file,
    require_non_empty_file,
    verify_output_dir,
    verify_output_file,
)

pytestmark = pytest.mark.unit


def test_require_file_and_non_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("x", encoding="utf-8")

    assert require_file(path, label="file") == path.resolve()
    assert require_non_empty_file(path, label="file") == path.resolve()

    empty_path = tmp_path / "empty.txt"
    empty_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        require_non_empty_file(empty_path, label="empty file")


def test_require_dir_and_dir_contains(tmp_path: Path) -> None:
    root = tmp_path / "root"
    child = root / "child.txt"
    child.parent.mkdir()
    child.write_text("x", encoding="utf-8")

    assert require_dir(root, label="root") == root.resolve()
    assert require_dir_contains(root, "*.txt", label="root") == root.resolve()

    with pytest.raises(FileNotFoundError, match="no matches"):
        require_dir_contains(root, "*.json", label="root")


def test_ensure_dir_creates_parents(tmp_path: Path) -> None:
    path = tmp_path / "a/b/c"

    assert ensure_dir(path, label="nested") == path.resolve()
    assert path.is_dir()


def test_prepare_output_file_policies(tmp_path: Path) -> None:
    path = tmp_path / "out/file.txt"

    assert prepare_output_file(path, label="output") == path.resolve()
    path.write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_output_file(path, label="output")

    assert (
        prepare_output_file(
            path,
            policy=OutputExistsPolicy.SKIP,
            label="output",
        )
        == path.resolve()
    )
    assert path.read_text(encoding="utf-8") == "old"

    assert (
        prepare_output_file(
            path,
            policy=OutputExistsPolicy.OVERWRITE,
            label="output",
        )
        == path.resolve()
    )
    assert not path.exists()


def test_prepare_output_dir_policies(tmp_path: Path) -> None:
    path = tmp_path / "out"

    assert prepare_output_dir(path, label="output dir") == path.resolve()
    (path / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_output_dir(path, label="output dir")

    assert (
        prepare_output_dir(
            path,
            policy=OutputExistsPolicy.SKIP,
            label="output dir",
        )
        == path.resolve()
    )
    assert (path / "old.txt").is_file()

    assert (
        prepare_output_dir(
            path,
            policy=OutputExistsPolicy.OVERWRITE,
            label="output dir",
        )
        == path.resolve()
    )
    assert path.is_dir()
    assert not (path / "old.txt").exists()


def test_verify_output_file_and_dir(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    dir_path = tmp_path / "dir"
    file_path.write_text("x", encoding="utf-8")
    dir_path.mkdir()

    assert verify_output_file(file_path, label="file") == file_path.resolve()
    assert verify_output_dir(dir_path, label="dir") == dir_path.resolve()
