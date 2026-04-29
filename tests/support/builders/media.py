"""Tiny media builders for tests that need readable MP4 metadata."""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np


def _make_box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, box_type) + payload


def write_minimal_mp4(
    path: Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: float = 24.0,
) -> None:
    timescale = 12288
    sample_count = 12
    sample_delta = int(round(timescale / fps))
    duration = sample_count * sample_delta
    tkhd_payload = (
        b"\x00\x00\x00\x00" + (b"\x00" * 72) + struct.pack(">II", width << 16, height << 16)
    )
    mdhd_payload = (
        b"\x00\x00\x00\x00"
        + (b"\x00" * 8)
        + struct.pack(">II", timescale, duration)
        + (b"\x00" * 4)
    )
    hdlr_payload = b"\x00\x00\x00\x00" + (b"\x00" * 4) + b"vide" + (b"\x00" * 12)
    stts_payload = (
        b"\x00\x00\x00\x00" + struct.pack(">I", 1) + struct.pack(">II", sample_count, sample_delta)
    )
    stbl = _make_box(b"stbl", _make_box(b"stts", stts_payload))
    minf = _make_box(b"minf", stbl)
    mdia = _make_box(
        b"mdia",
        _make_box(b"mdhd", mdhd_payload) + _make_box(b"hdlr", hdlr_payload) + minf,
    )
    trak = _make_box(b"trak", _make_box(b"tkhd", tkhd_payload) + mdia)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_make_box(b"moov", trak))


def write_tiny_decodable_mp4(
    path: Path,
    *,
    frame_count: int = 3,
    width: int = 64,
    height: int = 48,
    fps: float = 24.0,
) -> None:
    """Write a tiny MP4 that OpenCV can decode frame-by-frame."""

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open tiny MP4 writer: {path}")
    try:
        for index in range(frame_count):
            value = min(index * 40, 255)
            frame = np.full((height, width, 3), value, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
