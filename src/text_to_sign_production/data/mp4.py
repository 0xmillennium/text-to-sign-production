"""Lightweight MP4 header parsing without media decoding dependencies."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from struct import unpack
from typing import BinaryIO

from .schemas import VideoMetadata

CONTAINER_BOX_TYPES = frozenset(
    {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta"}
)


@dataclass(frozen=True, slots=True)
class MP4Box:
    """A single MP4 box entry."""

    start: int
    size: int
    box_type: bytes
    header_size: int

    @property
    def end(self) -> int:
        return self.start + self.size

    @property
    def content_start(self) -> int:
        return self.start + self.header_size

    @property
    def content_size(self) -> int:
        return self.size - self.header_size


def iter_boxes(handle: BinaryIO, start: int, end: int) -> Iterator[MP4Box]:
    """Iterate over MP4 boxes within a byte range."""

    position = start
    while position + 8 <= end:
        handle.seek(position)
        header = handle.read(8)
        if len(header) < 8:
            return

        size = unpack(">I", header[:4])[0]
        box_type = header[4:8]
        header_size = 8
        if size == 1:
            extended_size = handle.read(8)
            if len(extended_size) < 8:
                return
            size = unpack(">Q", extended_size)[0]
            header_size = 16
        elif size == 0:
            size = end - position

        if size < header_size:
            return

        yield MP4Box(position, size, box_type, header_size)
        position += size


def find_child_box(handle: BinaryIO, start: int, end: int, box_type: bytes) -> MP4Box | None:
    """Return the first immediate child box of the given type."""

    for box in iter_boxes(handle, start, end):
        if box.box_type == box_type:
            return box
    return None


def find_video_track(handle: BinaryIO, file_size: int) -> MP4Box | None:
    """Locate the first video track in an MP4 container."""

    moov = find_child_box(handle, 0, file_size, b"moov")
    if moov is None:
        return None

    for trak in iter_boxes(handle, moov.content_start, moov.end):
        if trak.box_type != b"trak":
            continue
        mdia = find_child_box(handle, trak.content_start, trak.end, b"mdia")
        if mdia is None:
            continue
        hdlr = find_child_box(handle, mdia.content_start, mdia.end, b"hdlr")
        if hdlr is None or hdlr.content_size < 12:
            continue
        handle.seek(hdlr.content_start)
        payload = handle.read(min(hdlr.content_size, 32))
        if payload[8:12] == b"vide":
            return trak
    return None


def parse_tkhd_dimensions(handle: BinaryIO, trak: MP4Box) -> tuple[int | None, int | None]:
    """Parse width and height from a track header."""

    tkhd = find_child_box(handle, trak.content_start, trak.end, b"tkhd")
    if tkhd is None or tkhd.content_size < 8:
        return None, None

    handle.seek(tkhd.end - 8)
    raw = handle.read(8)
    if len(raw) < 8:
        return None, None

    width = unpack(">I", raw[:4])[0] / 65536
    height = unpack(">I", raw[4:])[0] / 65536
    return int(round(width)), int(round(height))


def parse_mdhd_timescale(handle: BinaryIO, trak: MP4Box) -> int | None:
    """Return the track timescale."""

    mdia = find_child_box(handle, trak.content_start, trak.end, b"mdia")
    if mdia is None:
        return None
    mdhd = find_child_box(handle, mdia.content_start, mdia.end, b"mdhd")
    if mdhd is None or mdhd.content_size < 24:
        return None

    handle.seek(mdhd.content_start)
    payload = handle.read(min(mdhd.content_size, 32))
    if len(payload) < 24:
        return None

    version = payload[0]
    if version == 0:
        return int(unpack(">I", payload[12:16])[0])
    if len(payload) < 28:
        return None
    return int(unpack(">I", payload[20:24])[0])


def parse_stts_entries(handle: BinaryIO, trak: MP4Box) -> list[tuple[int, int]] | None:
    """Return `(sample_count, sample_delta)` pairs from the `stts` box."""

    mdia = find_child_box(handle, trak.content_start, trak.end, b"mdia")
    if mdia is None:
        return None
    minf = find_child_box(handle, mdia.content_start, mdia.end, b"minf")
    if minf is None:
        return None
    stbl = find_child_box(handle, minf.content_start, minf.end, b"stbl")
    if stbl is None:
        return None
    stts = find_child_box(handle, stbl.content_start, stbl.end, b"stts")
    if stts is None or stts.content_size < 8:
        return None

    handle.seek(stts.content_start)
    payload = handle.read(stts.content_size)
    if len(payload) < 8:
        return None

    entry_count = unpack(">I", payload[4:8])[0]
    offset = 8
    entries: list[tuple[int, int]] = []
    for _ in range(entry_count):
        if offset + 8 > len(payload):
            return None
        entries.append(unpack(">II", payload[offset : offset + 8]))
        offset += 8
    return entries


def read_video_metadata(path: Path) -> VideoMetadata:
    """Read lightweight width, height, and FPS metadata from an MP4."""

    if not path.exists():
        return VideoMetadata(width=None, height=None, fps=None, error="video_file_missing")

    try:
        with path.open("rb") as handle:
            file_size = path.stat().st_size
            trak = find_video_track(handle, file_size)
            if trak is None:
                return VideoMetadata(width=None, height=None, fps=None, error="video_track_missing")

            width, height = parse_tkhd_dimensions(handle, trak)
            timescale = parse_mdhd_timescale(handle, trak)
            entries = parse_stts_entries(handle, trak)
            if timescale is None or entries is None or not entries:
                return VideoMetadata(
                    width=width, height=height, fps=None, error="video_timing_missing"
                )

            total_samples = sum(sample_count for sample_count, _ in entries)
            total_time_units = sum(
                sample_count * sample_delta for sample_count, sample_delta in entries
            )
            if total_samples <= 0 or total_time_units <= 0:
                return VideoMetadata(
                    width=width, height=height, fps=None, error="video_timing_invalid"
                )

            fps = total_samples / (total_time_units / timescale)
            return VideoMetadata(width=width, height=height, fps=round(fps, 6), error=None)
    except OSError as exc:
        return VideoMetadata(
            width=None, height=None, fps=None, error=f"oserror:{exc.__class__.__name__}"
        )
