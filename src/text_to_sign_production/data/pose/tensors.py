"""Pose tensor construction from parsed frames."""

from __future__ import annotations

from collections import Counter

import numpy as np

from text_to_sign_production.data.pose.parser import parse_frame
from text_to_sign_production.data.pose.people import (
    build_person_metadata,
    resolve_target_person_index,
)
from text_to_sign_production.data.pose.schema import (
    CANONICAL_POSE_CHANNELS,
    OPENPOSE_CHANNEL_SPECS,
)
from text_to_sign_production.data.pose.types import (
    ParsedFrameResult,
    PoseBuildDiagnostics,
    PoseBuildInput,
    PoseBuildOutput,
)
from text_to_sign_production.data.samples.types import (
    BfhPosePayload,
    FrameQualitySummary,
    PoseChannelPayload,
)


def _empty_channel_tensor(
    num_frames: int,
    channel: str,
) -> tuple[np.ndarray, np.ndarray]:
    _, point_count = OPENPOSE_CHANNEL_SPECS[channel]
    return (
        np.zeros((num_frames, point_count, 2), dtype=np.float32),
        np.zeros((num_frames, point_count), dtype=np.float32),
    )


def build_pose_tensors(build_input: PoseBuildInput) -> PoseBuildOutput:
    """Build pose tensors and facts for a candidate."""

    candidate = build_input.candidate
    frames = build_input.frames
    num_frames = candidate.frame_count

    coord_tensors = {}
    confidence_tensors = {}
    for channel in OPENPOSE_CHANNEL_SPECS:
        coord_tensors[channel], confidence_tensors[channel] = _empty_channel_tensor(
            num_frames, channel
        )

    people_per_frame = np.zeros((num_frames,), dtype=np.int16)
    frame_valid_mask = np.zeros((num_frames,), dtype=np.bool_)

    issue_counter: Counter[str] = Counter()
    face_missing_frame_count = 0
    out_of_bounds_coordinate_count = 0
    frames_with_any_zeroed_canonical_joint = 0
    multi_person_frame_count = 0
    max_people_per_frame = 0

    parse_error: str | None = None
    parsed_frames: list[ParsedFrameResult] = []

    try:
        for frame_path in frames.files:
            parsed_frames.append(parse_frame(frame_path))
    except Exception as exc:
        parse_error = f"{exc.__class__.__name__}:{exc}"

    target_person_index = resolve_target_person_index(parsed_frames)

    for index, parsed in enumerate(parsed_frames):
        people_count = len(parsed.people)
        people_per_frame[index] = people_count
        max_people_per_frame = max(max_people_per_frame, people_count)
        if people_count > 1:
            multi_person_frame_count += 1

        issue_counter.update(parsed.issue_codes)

        if target_person_index >= people_count:
            frame_valid_mask[index] = False
            issue_counter["target_person_missing"] += 1
            continue

        person = parsed.people[target_person_index]
        frame_valid_mask[index] = person.person_valid

        if person.face_missing:
            face_missing_frame_count += 1
        if person.has_any_zeroed_canonical_joint:
            frames_with_any_zeroed_canonical_joint += 1

        out_of_bounds_coordinate_count += person.out_of_bounds_coordinate_count
        issue_counter.update(person.issue_codes)

        for channel in OPENPOSE_CHANNEL_SPECS:
            coord_tensors[channel][index] = person.coords[channel]
            confidence_tensors[channel][index] = person.confidences[channel]

    channel_nonzero_frames = {
        channel: int(np.count_nonzero(np.any(confidence_tensors[channel] > 0.0, axis=1)))
        for channel in CANONICAL_POSE_CHANNELS
    }

    selected_person = build_person_metadata(
        target_index=target_person_index,
        multi_person_frame_count=multi_person_frame_count,
        max_people_per_frame=max_people_per_frame,
    )

    frame_valid_count = int(frame_valid_mask.sum())
    frame_quality = FrameQualitySummary(
        valid_frame_count=frame_valid_count,
        invalid_frame_count=num_frames - frame_valid_count,
        face_missing_frame_count=face_missing_frame_count,
        out_of_bounds_coordinate_count=out_of_bounds_coordinate_count,
        frames_with_any_zeroed_required_joint=frames_with_any_zeroed_canonical_joint,
        frame_issue_counts={str(key): int(value) for key, value in sorted(issue_counter.items())},
        channel_nonzero_frames=channel_nonzero_frames,
    )

    pose_payload = BfhPosePayload(
        body=PoseChannelPayload(
            coordinates=coord_tensors["body"], confidence=confidence_tensors["body"]
        ),
        left_hand=PoseChannelPayload(
            coordinates=coord_tensors["left_hand"], confidence=confidence_tensors["left_hand"]
        ),
        right_hand=PoseChannelPayload(
            coordinates=coord_tensors["right_hand"], confidence=confidence_tensors["right_hand"]
        ),
        face=PoseChannelPayload(
            coordinates=coord_tensors["face"], confidence=confidence_tensors["face"]
        ),
    )

    diagnostics = PoseBuildDiagnostics(parse_error=parse_error)

    return PoseBuildOutput(
        pose=pose_payload,
        frame_quality=frame_quality,
        selected_person=selected_person,
        diagnostics=diagnostics,
    )
