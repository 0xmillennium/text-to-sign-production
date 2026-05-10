"""Frame and duration samples admission gate."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.core.models import (
    GateDecision,
    GateIssueCode,
    GateName,
    GateStatus,
    PreparedSample,
)
from text_to_sign_production.data.gate.policies.config import (
    FramesGateThresholds,
)


def evaluate_frames_gate(
    sample: PreparedSample,
    thresholds: FramesGateThresholds,
) -> GateDecision:
    """Evaluate whether frame truth is admissible into the samples checkpoint."""
    issues: list[GateIssueCode] = []
    valid_frame_count = int(np.count_nonzero(sample.pose.valid_frame_mask))
    if sample.pose.frame_count < thresholds.min_frame_count:
        issues.append(GateIssueCode.FRAME_COUNT_TOO_LOW)
    if valid_frame_count < thresholds.min_valid_frame_count:
        issues.append(GateIssueCode.VALID_FRAME_COUNT_TOO_LOW)
    duration_seconds = sample.pose.frame_count / sample.source.fps
    if duration_seconds < thresholds.min_duration_seconds:
        issues.append(GateIssueCode.DURATION_OUT_OF_RANGE)
    if (
        thresholds.max_duration_seconds is not None
        and duration_seconds > thresholds.max_duration_seconds
    ):
        issues.append(GateIssueCode.DURATION_OUT_OF_RANGE)
    issue_codes = tuple(dict.fromkeys(issues))
    return GateDecision(
        gate=GateName.FRAMES,
        status=GateStatus.FAIL if issue_codes else GateStatus.PASS,
        issue_codes=issue_codes,
    )


__all__ = ["evaluate_frames_gate"]
