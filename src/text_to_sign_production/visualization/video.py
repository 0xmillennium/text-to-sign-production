"""MP4 rendering helpers for processed pose debug outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from .pose import PoseSample
from .skeleton import SkeletonRenderConfig, render_pose_frame


class VideoRenderError(RuntimeError):
    """Raised when a debug MP4 cannot be rendered."""


CANONICAL_MP4_CODEC = "mp4v"


def render_skeleton_video(
    *,
    pose_sample: PoseSample,
    output_path: Path | str,
    fps: float,
    config: SkeletonRenderConfig | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Write a skeleton-only MP4 and return render metadata."""

    resolved_config = config or SkeletonRenderConfig()
    resolved_output_path = Path(output_path).expanduser().resolve()
    _require_prepared_output_parent(resolved_output_path)
    writer = _video_writer(
        resolved_output_path,
        fps=fps,
        frame_size=(resolved_config.canvas_width, resolved_config.canvas_height),
    )
    try:
        for frame_index in range(pose_sample.num_frames):
            writer.write(
                render_pose_frame(
                    pose_sample,
                    frame_index,
                    config=resolved_config,
                    label=label,
                )
            )
    finally:
        writer.release()

    return {
        "output_path": resolved_output_path.as_posix(),
        "skeleton_frames": pose_sample.num_frames,
        "video_frames": None,
        "fps": fps,
        "mode": "skeleton_only",
        "codec": CANONICAL_MP4_CODEC,
    }


def render_side_by_side_video(
    *,
    source_video_path: Path | str,
    pose_sample: PoseSample,
    output_path: Path | str,
    fps: float | None = None,
    config: SkeletonRenderConfig | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Write source video + skeleton debug MP4 and return render metadata."""

    resolved_config = config or SkeletonRenderConfig()
    resolved_source_video_path = Path(source_video_path).expanduser().resolve()
    resolved_output_path = Path(output_path).expanduser().resolve()
    if not resolved_source_video_path.is_file():
        raise FileNotFoundError(f"Source video not found: {resolved_source_video_path}")
    _require_prepared_output_parent(resolved_output_path)

    capture = cv2.VideoCapture(str(resolved_source_video_path))
    if not capture.isOpened():
        raise VideoRenderError(f"Source video could not be opened: {resolved_source_video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    output_fps = _usable_fps(source_fps) or _usable_fps(fps) or 24.0
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output_frame_count = source_frame_count if source_frame_count > 0 else pose_sample.num_frames
    writer = _video_writer(
        resolved_output_path,
        fps=output_fps,
        frame_size=(resolved_config.canvas_width * 2, resolved_config.canvas_height),
    )

    written_frames = 0
    try:
        for output_index in range(output_frame_count):
            ok, source_frame = capture.read()
            if ok:
                left_frame = cv2.resize(
                    source_frame,
                    (resolved_config.canvas_width, resolved_config.canvas_height),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                left_frame = _missing_source_frame(resolved_config, output_index)
            skeleton_index = _map_skeleton_index(
                output_index,
                output_frame_count=output_frame_count,
                skeleton_frames=pose_sample.num_frames,
                source_frames=source_frame_count,
            )
            right_frame = render_pose_frame(
                pose_sample,
                skeleton_index,
                config=resolved_config,
                label=label,
            )
            writer.write(np.concatenate([left_frame, right_frame], axis=1))
            written_frames += 1
    finally:
        capture.release()
        writer.release()

    return {
        "output_path": resolved_output_path.as_posix(),
        "video_path": resolved_source_video_path.as_posix(),
        "skeleton_frames": pose_sample.num_frames,
        "video_frames": written_frames,
        "source_video_frames": source_frame_count,
        "fps": output_fps,
        "mode": "side_by_side",
        "codec": CANONICAL_MP4_CODEC,
    }


def _usable_fps(value: float | None) -> float | None:
    if value is None or not np.isfinite(value) or value <= 0:
        return None
    return float(value)


def _require_prepared_output_parent(output_path: Path) -> None:
    parent = output_path.parent
    if not parent.exists():
        raise VideoRenderError(f"Output parent directory does not exist: {parent}")
    if not parent.is_dir():
        raise VideoRenderError(f"Output parent path is not a directory: {parent}")
    if not os.access(parent, os.W_OK):
        raise VideoRenderError(f"Output parent directory is not writable: {parent}")


def _validate_codec(codec: str) -> str:
    resolved_codec = codec.strip()
    if len(resolved_codec) != 4:
        raise VideoRenderError(f"MP4 codec must be exactly 4 characters; received {codec!r}.")
    return resolved_codec


def _video_writer(
    output_path: Path,
    *,
    fps: float,
    frame_size: tuple[int, int],
    codec: str = CANONICAL_MP4_CODEC,
) -> cv2.VideoWriter:
    resolved_codec = _validate_codec(codec)
    fourcc = cv2.VideoWriter.fourcc(*resolved_codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise VideoRenderError(
            f"Could not open MP4 writer for codec {resolved_codec!r}: {output_path}."
        )
    return writer


def _map_skeleton_index(
    output_index: int,
    *,
    output_frame_count: int,
    skeleton_frames: int,
    source_frames: int,
) -> int:
    if skeleton_frames <= 1 or output_frame_count <= 1:
        return 0
    if source_frames > 0:
        tolerance = max(1, round(max(source_frames, skeleton_frames) * 0.05))
        if abs(source_frames - skeleton_frames) <= tolerance:
            return min(output_index, skeleton_frames - 1)
    scaled = round(output_index * (skeleton_frames - 1) / (output_frame_count - 1))
    return max(0, min(int(scaled), skeleton_frames - 1))


def _missing_source_frame(
    config: SkeletonRenderConfig,
    output_index: int,
) -> npt.NDArray[np.uint8]:
    frame = np.full(
        (config.canvas_height, config.canvas_width, 3),
        (12, 14, 18),
        dtype=np.uint8,
    )
    cv2.putText(
        frame,
        f"source frame {output_index}: unavailable",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (60, 220, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return frame
