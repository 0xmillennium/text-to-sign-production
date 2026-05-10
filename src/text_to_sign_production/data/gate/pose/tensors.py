"""Pose tensorization from parsed frames and tracking truth."""

from __future__ import annotations

from collections import Counter

import numpy as np

from text_to_sign_production.data.gate.pose.schema import (
    CANONICAL_POSE_CHANNELS,
    OPENPOSE_CHANNEL_SPECS,
)
from text_to_sign_production.data.gate.pose.types import (
    FrameFileListing,
    ParsedFrame,
    PersonTrackingResult,
    PoseBuildDiagnostics,
    PoseBuildInput,
    PoseBuildOutput,
    PoseChannel,
    PoseChannelTensor,
    PoseDiagnostic,
    PoseDiagnosticCode,
    PoseDiagnosticSeverity,
    PoseTensorOutput,
)
from text_to_sign_production.data.gate.sources import SourceCandidate


def _empty_channel_tensor(
    frame_count: int,
    channel: PoseChannel,
) -> tuple[np.ndarray, np.ndarray]:
    _, point_count = OPENPOSE_CHANNEL_SPECS[channel]
    return (
        np.zeros((frame_count, point_count, 2), dtype=np.float32),
        np.zeros((frame_count, point_count), dtype=np.float32),
    )


def build_pose_input(
    *,
    candidate: SourceCandidate,
    frame_listing: FrameFileListing,
    parsed_frames: tuple[ParsedFrame, ...],
    tracking: PersonTrackingResult,
) -> PoseBuildInput:
    """Bundle explicit pose build inputs without discovering source truth."""
    return PoseBuildInput(
        candidate=candidate,
        frame_listing=frame_listing,
        parsed_frames=parsed_frames,
        tracking=tracking,
    )


def build_pose_tensors(build_input: PoseBuildInput) -> PoseTensorOutput:
    """Tensorize parsed frame truth using frame-wise tracking truth."""
    candidate = build_input.candidate
    parsed_frames = build_input.parsed_frames
    frame_count = len(parsed_frames)
    coordinate_tensors: dict[PoseChannel, np.ndarray] = {}
    confidence_tensors: dict[PoseChannel, np.ndarray] = {}
    for channel in CANONICAL_POSE_CHANNELS:
        coordinate_tensors[channel], confidence_tensors[channel] = _empty_channel_tensor(
            frame_count, channel
        )

    people_per_frame = np.zeros((frame_count,), dtype=np.int16)
    frame_valid_mask = np.zeros((frame_count,), dtype=np.bool_)
    diagnostics: list[PoseDiagnostic] = []

    selections_by_frame = {
        selection.frame_index: selection for selection in build_input.tracking.frame_selections
    }
    for position, frame in enumerate(parsed_frames):
        people_per_frame[position] = len(frame.people)
        diagnostics.extend(frame.diagnostics)
        selection = selections_by_frame.get(frame.frame_index)
        selected_index = selection.selected_person_index if selection is not None else None
        if selection is not None:
            diagnostics.extend(selection.diagnostics)
        if selected_index is None or selected_index >= len(frame.people):
            diagnostics.append(
                PoseDiagnostic(
                    code=PoseDiagnosticCode.TARGET_PERSON_MISSING,
                    severity=PoseDiagnosticSeverity.WARNING,
                    message="No tracked person available for tensorization.",
                    frame_index=frame.frame_index,
                    person_index=selected_index,
                )
            )
            continue

        person = frame.people[selected_index]
        diagnostics.extend(person.diagnostics)
        frame_valid_mask[position] = person.person_valid
        for channel in CANONICAL_POSE_CHANNELS:
            coordinate_tensors[channel][position] = person.channels[channel].coordinates
            confidence_tensors[channel][position] = person.channels[channel].confidences

    channels = {
        channel: PoseChannelTensor(
            channel=channel,
            coordinates=coordinate_tensors[channel],
            confidences=confidence_tensors[channel],
        )
        for channel in CANONICAL_POSE_CHANNELS
    }
    channel_nonzero_counts = {
        channel: int(np.count_nonzero(np.any(channels[channel].confidences > 0.0, axis=1)))
        for channel in CANONICAL_POSE_CHANNELS
    }

    return PoseTensorOutput(
        candidate=candidate,
        channels=channels,
        people_per_frame=people_per_frame,
        frame_valid_mask=frame_valid_mask,
        selected_person_indices=build_input.tracking.selected_person_indices,
        channel_nonzero_frame_counts=channel_nonzero_counts,
        diagnostics=tuple(diagnostics),
    )


def build_pose_output(build_input: PoseBuildInput) -> PoseBuildOutput:
    """Build final pose output from explicit parsed/tracking input truth."""
    tensors = build_pose_tensors(build_input)
    frame_diagnostics = tuple(
        diagnostic for frame in build_input.parsed_frames for diagnostic in frame.diagnostics
    )
    tensor_counter = Counter(diagnostic.code for diagnostic in tensors.diagnostics)
    diagnostics = PoseBuildDiagnostics(
        diagnostics=tuple(
            list(build_input.frame_listing.diagnostics)
            + list(build_input.tracking.diagnostics)
            + list(tensors.diagnostics)
        ),
        frame_diagnostic_count=len(frame_diagnostics),
        tracking_diagnostic_count=len(build_input.tracking.diagnostics),
        tensor_diagnostic_count=sum(tensor_counter.values()),
    )
    return PoseBuildOutput(tensors=tensors, tracking=build_input.tracking, diagnostics=diagnostics)


__all__ = ["build_pose_input", "build_pose_output", "build_pose_tensors"]
