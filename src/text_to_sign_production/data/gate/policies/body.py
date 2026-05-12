"""Body-channel gate admission gate."""

from __future__ import annotations

from text_to_sign_production.core.models import (
    GateDecision,
    GateIssueCode,
    GateName,
    GateStatus,
    PreparedSample,
)
from text_to_sign_production.data.gate.policies.config import BodyGateThresholds


def evaluate_body_gate(
    sample: PreparedSample,
    thresholds: BodyGateThresholds,
) -> GateDecision:
    """Evaluate whether body support is admissible into the dataset checkpoint."""
    issues: list[GateIssueCode] = []
    count = sample.pose.body_nonzero_frame_count
    if (
        thresholds.min_body_nonzero_frames is not None
        and count < thresholds.min_body_nonzero_frames
    ):
        issues.append(GateIssueCode.CHANNEL_EVIDENCE_TOO_LOW)
    if thresholds.min_body_available_frame_ratio is not None:
        ratio = count / sample.pose.frame_count
        if ratio < thresholds.min_body_available_frame_ratio:
            issues.append(GateIssueCode.UPPER_BODY_SUPPORT_TOO_LOW)
    issue_codes = tuple(dict.fromkeys(issues))
    return GateDecision(
        gate=GateName.BODY,
        status=GateStatus.FAIL if issue_codes else GateStatus.PASS,
        issue_codes=issue_codes,
    )


__all__ = ["evaluate_body_gate"]
