"""Generic 2D skeleton/keypoint rendering for processed pose arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt

from text_to_sign_production.data.constants import CANVAS_HEIGHT, CANVAS_WIDTH

from .pose import PoseSample

BODY_25_EDGES: Final[tuple[tuple[int, int], ...]] = (
    (1, 8),
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (8, 9),
    (9, 10),
    (10, 11),
    (8, 12),
    (12, 13),
    (13, 14),
    (1, 0),
    (0, 15),
    (15, 17),
    (0, 16),
    (16, 18),
    (14, 19),
    (19, 20),
    (14, 21),
    (11, 22),
    (22, 23),
    (11, 24),
)
HAND_21_EDGES: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

_Color = tuple[int, int, int]
_BACKGROUND: Final[_Color] = (10, 12, 16)
_GRID: Final[_Color] = (28, 34, 42)
_TEXT: Final[_Color] = (235, 238, 244)
_INVALID_TEXT: Final[_Color] = (60, 220, 255)
_BODY_COLOR: Final[_Color] = (75, 210, 255)
_LEFT_HAND_COLOR: Final[_Color] = (90, 230, 130)
_RIGHT_HAND_COLOR: Final[_Color] = (255, 185, 75)
_FACE_COLOR: Final[_Color] = (235, 115, 210)


@dataclass(frozen=True, slots=True)
class SkeletonRenderConfig:
    """Configuration for rendering one pose frame."""

    canvas_width: int = CANVAS_WIDTH
    canvas_height: int = CANVAS_HEIGHT
    confidence_threshold: float = 0.0
    draw_face: bool = True
    draw_labels: bool = True
    background_color: _Color = _BACKGROUND


def render_pose_frame(
    pose_sample: PoseSample,
    frame_index: int,
    *,
    config: SkeletonRenderConfig | None = None,
    label: str | None = None,
) -> npt.NDArray[np.uint8]:
    """Render one processed pose frame as a BGR image suitable for OpenCV."""

    resolved_config = config or SkeletonRenderConfig()
    frame = np.full(
        (resolved_config.canvas_height, resolved_config.canvas_width, 3),
        resolved_config.background_color,
        dtype=np.uint8,
    )
    _draw_reference_grid(frame)

    if frame_index < 0 or frame_index >= pose_sample.num_frames:
        _draw_status(frame, f"frame {frame_index}: outside sample", invalid=True)
        return frame
    if not bool(pose_sample.frame_valid_mask[frame_index]):
        _draw_status(frame, f"frame {frame_index}: invalid pose frame", invalid=True)
        return frame

    _draw_channel(
        frame,
        pose_sample.body[frame_index],
        pose_sample.body_confidence[frame_index],
        BODY_25_EDGES,
        color=_BODY_COLOR,
        config=resolved_config,
        radius=3,
        thickness=2,
    )
    _draw_channel(
        frame,
        pose_sample.left_hand[frame_index],
        pose_sample.left_hand_confidence[frame_index],
        HAND_21_EDGES,
        color=_LEFT_HAND_COLOR,
        config=resolved_config,
        radius=2,
        thickness=1,
    )
    _draw_channel(
        frame,
        pose_sample.right_hand[frame_index],
        pose_sample.right_hand_confidence[frame_index],
        HAND_21_EDGES,
        color=_RIGHT_HAND_COLOR,
        config=resolved_config,
        radius=2,
        thickness=1,
    )
    if resolved_config.draw_face:
        _draw_points(
            frame,
            pose_sample.face[frame_index],
            pose_sample.face_confidence[frame_index],
            color=_FACE_COLOR,
            config=resolved_config,
            radius=1,
        )

    if resolved_config.draw_labels:
        resolved_label = f"frame {frame_index}"
        if label:
            resolved_label = f"{resolved_label} | {label}"
        _draw_status(frame, resolved_label, invalid=False)
    return frame


def _draw_channel(
    frame: npt.NDArray[np.uint8],
    coords: npt.NDArray[np.float32],
    confidence: npt.NDArray[np.float32],
    edges: tuple[tuple[int, int], ...],
    *,
    color: _Color,
    config: SkeletonRenderConfig,
    radius: int,
    thickness: int,
) -> None:
    for start, end in edges:
        start_point = _pixel_point(coords, confidence, start, config)
        end_point = _pixel_point(coords, confidence, end, config)
        if start_point is None or end_point is None:
            continue
        cv2.line(frame, start_point, end_point, color, thickness, lineType=cv2.LINE_AA)
    _draw_points(frame, coords, confidence, color=color, config=config, radius=radius)


def _draw_points(
    frame: npt.NDArray[np.uint8],
    coords: npt.NDArray[np.float32],
    confidence: npt.NDArray[np.float32],
    *,
    color: _Color,
    config: SkeletonRenderConfig,
    radius: int,
) -> None:
    for index in range(coords.shape[0]):
        point = _pixel_point(coords, confidence, index, config)
        if point is None:
            continue
        cv2.circle(frame, point, radius, color, thickness=-1, lineType=cv2.LINE_AA)


def _pixel_point(
    coords: npt.NDArray[np.float32],
    confidence: npt.NDArray[np.float32],
    index: int,
    config: SkeletonRenderConfig,
) -> tuple[int, int] | None:
    if index >= coords.shape[0] or index >= confidence.shape[0]:
        return None
    if float(confidence[index]) <= config.confidence_threshold:
        return None
    x_value = float(coords[index, 0])
    y_value = float(coords[index, 1])
    if not (np.isfinite(x_value) and np.isfinite(y_value)):
        return None
    x_pixel = int(round(x_value * config.canvas_width))
    y_pixel = int(round(y_value * config.canvas_height))
    if x_pixel < 0 or x_pixel >= config.canvas_width:
        return None
    if y_pixel < 0 or y_pixel >= config.canvas_height:
        return None
    return x_pixel, y_pixel


def _draw_reference_grid(frame: npt.NDArray[np.uint8]) -> None:
    height, width = frame.shape[:2]
    for x_pixel in range(0, width, max(width // 8, 1)):
        cv2.line(frame, (x_pixel, 0), (x_pixel, height - 1), _GRID, 1)
    for y_pixel in range(0, height, max(height // 6, 1)):
        cv2.line(frame, (0, y_pixel), (width - 1, y_pixel), _GRID, 1)


def _draw_status(frame: npt.NDArray[np.uint8], text: str, *, invalid: bool) -> None:
    color = _INVALID_TEXT if invalid else _TEXT
    cv2.rectangle(frame, (12, 12), (min(frame.shape[1] - 12, 700), 48), (18, 22, 28), -1)
    cv2.putText(
        frame,
        text,
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        lineType=cv2.LINE_AA,
    )
