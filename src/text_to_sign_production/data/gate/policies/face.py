"""Face-channel samples admission gate."""

from __future__ import annotations

from text_to_sign_production.core.models import (
    GateDecision,
    GateIssueCode,
    GateName,
    GateStatus,
    PreparedSample,
)
from text_to_sign_production.data.gate.policies.config import FaceGateThresholds


def evaluate_face_gate(
    sample: PreparedSample,
    thresholds: FaceGateThresholds,
) -> GateDecision:
    """Evaluate whether face support is admissible into the samples checkpoint."""
    issues: list[GateIssueCode] = []
    count = sample.pose.face_nonzero_frame_count
    if (
        thresholds.min_face_nonzero_frames is not None
        and count < thresholds.min_face_nonzero_frames
    ):
        issues.append(GateIssueCode.CHANNEL_EVIDENCE_TOO_LOW)
    if thresholds.min_face_available_frame_ratio is not None:
        ratio = count / sample.pose.frame_count
        if ratio < thresholds.min_face_available_frame_ratio:
            issues.append(GateIssueCode.FACE_VISIBILITY_TOO_LOW)
    issue_codes = tuple(dict.fromkeys(issues))
    return GateDecision(
        gate=GateName.FACE,
        status=GateStatus.FAIL if issue_codes else GateStatus.PASS,
        issue_codes=issue_codes,
    )


__all__ = ["evaluate_face_gate"]
