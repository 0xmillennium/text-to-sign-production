"""Manual-channel samples admission gate."""

from __future__ import annotations

from text_to_sign_production.core.models import (
    GateDecision,
    GateIssueCode,
    GateName,
    GateStatus,
    PreparedSample,
)
from text_to_sign_production.data.gate.policies.config import HandGateThresholds


def evaluate_hand_gate(
    sample: PreparedSample,
    thresholds: HandGateThresholds,
) -> GateDecision:
    """Evaluate whether manual support is admissible into the samples checkpoint."""
    issues: list[GateIssueCode] = []
    count = max(
        sample.pose.left_hand_nonzero_frame_count,
        sample.pose.right_hand_nonzero_frame_count,
    )
    if (
        thresholds.min_any_hand_nonzero_frames is not None
        and count < thresholds.min_any_hand_nonzero_frames
    ):
        issues.append(GateIssueCode.CHANNEL_EVIDENCE_TOO_LOW)
    if thresholds.min_any_hand_available_frame_ratio is not None:
        ratio = count / sample.pose.frame_count
        if ratio < thresholds.min_any_hand_available_frame_ratio:
            issues.append(GateIssueCode.MANUAL_VISIBILITY_TOO_LOW)
    if thresholds.max_tracked_target_missing_frame_ratio is not None:
        ratio = sample.pose.tracked_target_missing_frame_count / sample.pose.frame_count
        if ratio > thresholds.max_tracked_target_missing_frame_ratio:
            issues.append(GateIssueCode.TRACKING_MISSING_TOO_HIGH)
    issue_codes = tuple(dict.fromkeys(issues))
    return GateDecision(
        gate=GateName.HAND,
        status=GateStatus.FAIL if issue_codes else GateStatus.PASS,
        issue_codes=issue_codes,
    )


__all__ = ["evaluate_hand_gate"]
