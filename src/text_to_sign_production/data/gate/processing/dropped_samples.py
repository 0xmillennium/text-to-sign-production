"""Compute-only DroppedSample builders for gate-stage drops."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.core.models import (
    DroppedSample,
    DroppedSampleGateSnapshot,
    DroppedSamplePoseSnapshot,
    DroppedSampleSourceSnapshot,
    GateDecisionBundle,
    GateDropIssueCode,
    GateDropStage,
    PreparedSample,
)
from text_to_sign_production.data.dataset.dropped_payloads import (
    DROPPED_SAMPLE_PAYLOAD_SCHEMA_VERSION,
)
from text_to_sign_production.data.gate.sources import SourceCandidate, SourceMatchResult
from text_to_sign_production.data.gate.sources.candidates import sample_id_from_translation
from text_to_sign_production.data.gate.sources.types import CandidateViabilityReport


def build_source_dropped_sample(match: SourceMatchResult) -> DroppedSample:
    """Build source-stage dropped-sample evidence."""
    return DroppedSample(
        schema_version=DROPPED_SAMPLE_PAYLOAD_SCHEMA_VERSION,
        sample_id=sample_id_from_translation(match.translation),
        split=match.split,
        drop_stage=GateDropStage.SOURCE,
        issue_codes=(GateDropIssueCode.SOURCE_TRUTH_INVALID,),
        source=_source_snapshot(match),
        pose=None,
        gate=None,
    )


def build_pose_dropped_sample(
    *,
    match: SourceMatchResult,
    candidate: SourceCandidate,
    viability_report: CandidateViabilityReport,
    observed_frame_count: int | None,
    missing_frame_files: bool | None,
) -> DroppedSample:
    """Build pose-stage dropped-sample evidence."""
    return DroppedSample(
        schema_version=DROPPED_SAMPLE_PAYLOAD_SCHEMA_VERSION,
        sample_id=candidate.sample_id,
        split=candidate.split,
        drop_stage=GateDropStage.POSE,
        issue_codes=(GateDropIssueCode.POSE_TRUTH_INVALID,),
        source=_source_snapshot(match),
        pose=_pose_snapshot(
            candidate=candidate,
            viability_report=viability_report,
            observed_frame_count=observed_frame_count,
            missing_frame_files=missing_frame_files,
        ),
        gate=None,
    )


def build_gate_dropped_sample(
    *,
    match: SourceMatchResult,
    candidate: SourceCandidate,
    viability_report: CandidateViabilityReport,
    sample: PreparedSample,
    gate: GateDecisionBundle,
    observed_frame_count: int | None,
    missing_frame_files: bool | None,
) -> DroppedSample:
    """Build gate-stage dropped-sample evidence."""
    return DroppedSample(
        schema_version=DROPPED_SAMPLE_PAYLOAD_SCHEMA_VERSION,
        sample_id=sample.source.sample_id,
        split=sample.source.split,
        drop_stage=GateDropStage.GATES,
        issue_codes=(GateDropIssueCode.GATE_FAILED,),
        source=_source_snapshot(match),
        pose=_pose_snapshot(
            candidate=candidate,
            viability_report=viability_report,
            observed_frame_count=observed_frame_count,
            missing_frame_files=missing_frame_files,
        ),
        gate=_gate_snapshot(sample, gate),
    )


def _source_snapshot(match: SourceMatchResult) -> DroppedSampleSourceSnapshot:
    translation = match.translation
    return DroppedSampleSourceSnapshot(
        text=translation.text,
        source_video_id=translation.video_id,
        source_sentence_id=translation.sentence_id,
        source_sentence_name=translation.sentence_name,
        match_status=match.status.value,
        unmatched_reason=(
            None if match.unmatched_reason is None else match.unmatched_reason.code.value
        ),
        ambiguity_reasons=tuple(reason.code.value for reason in match.ambiguity_reasons),
        source_issue_codes=tuple(issue.code.value for issue in match.source_issues),
        video_match_count=len(match.video_matches),
        keypoint_match_count=len(match.keypoint_matches),
    )


def _pose_snapshot(
    *,
    candidate: SourceCandidate,
    viability_report: CandidateViabilityReport,
    observed_frame_count: int | None,
    missing_frame_files: bool | None,
) -> DroppedSamplePoseSnapshot:
    return DroppedSamplePoseSnapshot(
        candidate_available=True,
        video_path=str(candidate.video_path),
        keypoints_dir=str(candidate.keypoints_dir),
        candidate_frame_count=candidate.frame_count,
        observed_frame_count=observed_frame_count,
        video_metadata_readable=candidate.video_metadata.is_readable,
        video_metadata_error=candidate.video_metadata.error,
        missing_frame_files=missing_frame_files,
        viability_status=viability_report.status.value,
        viability_issue_codes=tuple(issue.code.value for issue in viability_report.issues),
        viability_issue_messages=tuple(issue.message for issue in viability_report.issues),
    )


def _gate_snapshot(
    sample: PreparedSample,
    gate: GateDecisionBundle,
) -> DroppedSampleGateSnapshot:
    return DroppedSampleGateSnapshot(
        prepared_sample_available=True,
        frame_count=sample.pose.frame_count,
        valid_frame_count=int(np.count_nonzero(sample.pose.valid_frame_mask)),
        body_nonzero_frame_count=sample.pose.body_nonzero_frame_count,
        face_nonzero_frame_count=sample.pose.face_nonzero_frame_count,
        left_hand_nonzero_frame_count=sample.pose.left_hand_nonzero_frame_count,
        right_hand_nonzero_frame_count=sample.pose.right_hand_nonzero_frame_count,
        final_status=gate.final_status,
        terminal_gate=gate.terminal_gate,
        failed_gates=gate.failed_gates,
        decision_issue_codes=tuple(
            code for decision in gate.decisions for code in decision.issue_codes
        ),
    )


__all__ = [
    "build_gate_dropped_sample",
    "build_pose_dropped_sample",
    "build_source_dropped_sample",
]
