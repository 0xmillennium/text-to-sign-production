"""Final gate composition entry point."""

from __future__ import annotations

from text_to_sign_production.data.gates.artifact import evaluate_artifact_gate
from text_to_sign_production.data.gates.body import evaluate_body_gate
from text_to_sign_production.data.gates.config import GatesConfig
from text_to_sign_production.data.gates.face import evaluate_face_gate
from text_to_sign_production.data.gates.frames import evaluate_frames_gate
from text_to_sign_production.data.gates.hand import evaluate_hand_gate
from text_to_sign_production.data.gates.schema import evaluate_schema_gate
from text_to_sign_production.data.gates.source import evaluate_source_gate
from text_to_sign_production.data.gates.types import (
    GateResult,
    GateStage,
    GateStatus,
    ProcessingDecision,
    ProcessingStatus,
)
from text_to_sign_production.data.pose.types import FrameFileListing, PoseBuildOutput
from text_to_sign_production.data.sources.types import SourceCandidate, SourceMatchResult


def evaluate_unmatched_source(match: SourceMatchResult) -> ProcessingDecision:
    """Build the canonical source-stage decision for an unmatched source row."""
    reasons = _unique_reasons(
        (*match.source_issues, match.unmatched_reason or "unmatched_source")
    )
    return ProcessingDecision(
        status=ProcessingStatus.DROPPED,
        drop_stage=GateStage.SOURCE,
        drop_reasons=reasons,
        can_materialize_debug=False,
        gate_results={
            GateStage.SOURCE: GateResult(status=GateStatus.DROPPED, reasons=reasons),
        },
    )


def evaluate_sample_processing(
    *,
    config: GatesConfig,
    candidate: SourceCandidate,
    frames_listing: FrameFileListing | None = None,
    pose_output: PoseBuildOutput | None = None,
) -> ProcessingDecision:
    """Evaluate structural gates in sequence to yield a final processing decision.

    The sequence stops early on fatal structural failures.
    """
    gate_results = {}

    def _fail(
        stage: GateStage, result: GateResult, materializable: bool
    ) -> ProcessingDecision:
        gate_results[stage] = result
        return ProcessingDecision(
            status=ProcessingStatus.DROPPED,
            drop_stage=stage,
            drop_reasons=result.reasons,
            can_materialize_debug=materializable,
            gate_results=gate_results,
        )

    # 1. Source
    source_res = evaluate_source_gate(candidate)
    gate_results[GateStage.SOURCE] = source_res
    if not source_res.passed:
        return _fail(GateStage.SOURCE, source_res, materializable=False)

    if frames_listing is None:
        return _fail(
            GateStage.FRAMES,
            GateResult(status=GateStatus.DROPPED, reasons=("missing_frame_listing",)),
            materializable=False,
        )

    # 2. Frames
    frames_res = evaluate_frames_gate(frames_listing, candidate, config)
    gate_results[GateStage.FRAMES] = frames_res
    if not frames_res.passed:
        return _fail(GateStage.FRAMES, frames_res, materializable=False)

    if pose_output is None:
        return _fail(
            GateStage.SCHEMA,
            GateResult(status=GateStatus.DROPPED, reasons=("missing_pose_output",)),
            materializable=False,
        )

    # 3. Schema
    schema_res = evaluate_schema_gate(pose_output, config)
    gate_results[GateStage.SCHEMA] = schema_res
    if not schema_res.passed:
        # Once we pass source/frames, we *might* be able to materialize a debug payload
        return _fail(GateStage.SCHEMA, schema_res, materializable=True)

    # 4. Body
    body_res = evaluate_body_gate(pose_output, config)
    gate_results[GateStage.BODY] = body_res
    if not body_res.passed:
        return _fail(GateStage.BODY, body_res, materializable=True)

    # 5. Hand
    hand_res = evaluate_hand_gate(pose_output, config)
    gate_results[GateStage.HAND] = hand_res
    if not hand_res.passed:
        return _fail(GateStage.HAND, hand_res, materializable=True)

    # 6. Face
    face_res = evaluate_face_gate(pose_output, config)
    gate_results[GateStage.FACE] = face_res
    if not face_res.passed:
        return _fail(GateStage.FACE, face_res, materializable=True)

    # 7. Artifact
    artifact_res = evaluate_artifact_gate(candidate, pose_output)
    gate_results[GateStage.ARTIFACT] = artifact_res
    if not artifact_res.passed:
        return _fail(GateStage.ARTIFACT, artifact_res, materializable=True)

    # Passed!
    return ProcessingDecision(
        status=ProcessingStatus.PROCESSED,
        drop_stage=None,
        drop_reasons=(),
        can_materialize_debug=True,
        gate_results=gate_results,
    )


def _unique_reasons(reasons: tuple[str, ...]) -> tuple[str, ...]:
    unique: list[str] = []
    for reason in reasons:
        if reason not in unique:
            unique.append(reason)
    return tuple(unique)
