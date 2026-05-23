"""Prepared-sample confidence canonicalization diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from text_to_sign_production.data.gate.pose import PoseBuildOutput, PoseChannel


@dataclass(frozen=True, slots=True)
class ConfidenceChannelCanonicalizationSummary:
    channel: str
    value_count: int
    lower_clipped_count: int
    upper_clipped_count: int
    nonfinite_count: int
    min_before_clip: float | None
    max_before_clip: float | None


@dataclass(frozen=True, slots=True)
class ConfidenceCanonicalizationSummary:
    sample_id: str
    split: str
    channel_summaries: tuple[ConfidenceChannelCanonicalizationSummary, ...]

    @property
    def clipped_value_count(self) -> int:
        return sum(
            item.lower_clipped_count + item.upper_clipped_count
            for item in self.channel_summaries
        )

    @property
    def upper_clipped_value_count(self) -> int:
        return sum(item.upper_clipped_count for item in self.channel_summaries)

    @property
    def lower_clipped_value_count(self) -> int:
        return sum(item.lower_clipped_count for item in self.channel_summaries)

    @property
    def nonfinite_count(self) -> int:
        return sum(item.nonfinite_count for item in self.channel_summaries)

    @property
    def max_before_clip(self) -> float | None:
        values = [
            item.max_before_clip
            for item in self.channel_summaries
            if item.max_before_clip is not None
        ]
        return max(values) if values else None

    @property
    def min_before_clip(self) -> float | None:
        values = [
            item.min_before_clip
            for item in self.channel_summaries
            if item.min_before_clip is not None
        ]
        return min(values) if values else None


_CHANNEL_FIELD_NAMES = {
    PoseChannel.BODY: "body_xyc",
    PoseChannel.LEFT_HAND: "left_hand_xyc",
    PoseChannel.RIGHT_HAND: "right_hand_xyc",
    PoseChannel.FACE: "face_xyc",
}


def summarize_pose_output_confidence(
    *,
    sample_id: str,
    split: str,
    pose_output: PoseBuildOutput,
) -> ConfidenceCanonicalizationSummary:
    """Summarize raw pose confidence values before prepared-sample clamping."""

    summaries: list[ConfidenceChannelCanonicalizationSummary] = []
    for channel, field_name in _CHANNEL_FIELD_NAMES.items():
        raw = np.asarray(pose_output.tensors.channels[channel].confidences, dtype=np.float32)
        finite = np.isfinite(raw)
        finite_values = raw[finite]
        summaries.append(
            ConfidenceChannelCanonicalizationSummary(
                channel=field_name,
                value_count=int(raw.size),
                lower_clipped_count=int(np.count_nonzero(finite & (raw < 0.0))),
                upper_clipped_count=int(np.count_nonzero(finite & (raw > 1.0))),
                nonfinite_count=int(np.count_nonzero(~finite)),
                min_before_clip=(
                    None if finite_values.size == 0 else float(np.min(finite_values))
                ),
                max_before_clip=(
                    None if finite_values.size == 0 else float(np.max(finite_values))
                ),
            )
        )
    return ConfidenceCanonicalizationSummary(
        sample_id=sample_id,
        split=split,
        channel_summaries=tuple(summaries),
    )


__all__ = [
    "ConfidenceCanonicalizationSummary",
    "ConfidenceChannelCanonicalizationSummary",
    "summarize_pose_output_confidence",
]
