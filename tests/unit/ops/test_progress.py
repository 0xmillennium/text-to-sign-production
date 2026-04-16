"""Progress helper tests."""

from __future__ import annotations

import io

import pytest

import text_to_sign_production.ops.progress as progress_mod

pytestmark = pytest.mark.unit


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
