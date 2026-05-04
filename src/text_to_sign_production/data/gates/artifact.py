"""Artifact contract structural gate."""

from __future__ import annotations

from text_to_sign_production.data.gates.types import GateResult, GateStatus
from text_to_sign_production.data.pose.types import PoseBuildOutput
from text_to_sign_production.data.samples.types import ProcessedSamplePayload
from text_to_sign_production.data.samples.validate import validate_payload
from text_to_sign_production.data.sources.types import SourceCandidate


def evaluate_artifact_gate(candidate: SourceCandidate, pose_output: PoseBuildOutput) -> GateResult:
    """Evaluate if a valid processed sample contract can be formed.

    Does not perform file I/O or decide filesystem layout.
    """
    reasons = []

    try:
        payload = ProcessedSamplePayload(
            sample_id=candidate.sample_id,
            text=candidate.text,
            split=candidate.split,
            num_frames=candidate.frame_count,
            fps=candidate.video_metadata.fps,
            pose=pose_output.pose,
            frame_quality=pose_output.frame_quality,
            selected_person=pose_output.selected_person,
        )
    except (TypeError, ValueError) as exc:
        reasons.append(f"payload_construction_failed:{exc}")
        return GateResult(status=GateStatus.DROPPED, reasons=tuple(reasons))

    validation_issues = validate_payload(payload)
    if validation_issues:
        for issue in validation_issues:
            reasons.append(f"contract_invalid:{issue.code}")
        return GateResult(status=GateStatus.DROPPED, reasons=tuple(reasons))

    return GateResult(status=GateStatus.PASSED)
