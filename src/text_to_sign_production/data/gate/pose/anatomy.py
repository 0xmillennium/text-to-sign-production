"""Pose-domain semantic landmark and segment groupings."""

from __future__ import annotations

from typing import Final

FACE_UPPER_REGION_INDICES: Final[tuple[int, ...]] = tuple(range(17, 48)) + (68, 69)
FACE_LOWER_REGION_INDICES: Final[tuple[int, ...]] = tuple(range(0, 17)) + tuple(range(48, 68))
FACE_MOUTH_REGION_INDICES: Final[tuple[int, ...]] = tuple(range(48, 68))
FACE_EYEBROW_REGION_INDICES: Final[tuple[int, ...]] = tuple(range(17, 27))
FACE_HEAD_ANCHOR_INDICES: Final[tuple[int, ...]] = tuple(range(27, 36)) + tuple(range(36, 48))

HAND_FINGERTIP_INDICES: Final[tuple[int, ...]] = (4, 8, 12, 16, 20)
HAND_DISTAL_JOINT_INDICES: Final[tuple[int, ...]] = (3, 7, 11, 15, 19)
HAND_DISTAL_CHAIN_INDICES: Final[tuple[int, ...]] = (
    3,
    4,
    7,
    8,
    11,
    12,
    15,
    16,
    19,
    20,
)
HAND_PALM_ANCHOR_INDICES: Final[tuple[int, ...]] = (0, 1, 5, 9, 13, 17)
HAND_FINGER_CHAINS: Final[tuple[tuple[int, ...], ...]] = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)

UPPER_BODY_TRACKING_LANDMARK_INDICES: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7)
UPPER_BODY_BONE_SEGMENTS: Final[tuple[tuple[int, int], ...]] = (
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (2, 5),
)
HAND_BONE_SEGMENTS: Final[tuple[tuple[int, int], ...]] = tuple(
    (start, end)
    for chain in HAND_FINGER_CHAINS
    for start, end in zip(chain, chain[1:], strict=False)
)
CROSS_CHANNEL_BODY_SCALE_SEGMENTS: Final[tuple[tuple[int, int], ...]] = ((2, 5), (1, 2), (1, 5))
CROSS_CHANNEL_HAND_SCALE_SEGMENTS: Final[tuple[tuple[int, int], ...]] = (
    (0, 5),
    (0, 17),
    (5, 17),
)

__all__ = [
    "CROSS_CHANNEL_BODY_SCALE_SEGMENTS",
    "CROSS_CHANNEL_HAND_SCALE_SEGMENTS",
    "FACE_EYEBROW_REGION_INDICES",
    "FACE_HEAD_ANCHOR_INDICES",
    "FACE_LOWER_REGION_INDICES",
    "FACE_MOUTH_REGION_INDICES",
    "FACE_UPPER_REGION_INDICES",
    "HAND_BONE_SEGMENTS",
    "HAND_DISTAL_CHAIN_INDICES",
    "HAND_DISTAL_JOINT_INDICES",
    "HAND_FINGER_CHAINS",
    "HAND_FINGERTIP_INDICES",
    "HAND_PALM_ANCHOR_INDICES",
    "UPPER_BODY_BONE_SEGMENTS",
    "UPPER_BODY_TRACKING_LANDMARK_INDICES",
]
