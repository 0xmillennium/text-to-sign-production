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
    GateStatus,
    ProcessingDecision,
    ProcessingStatus,
)
from text_to_sign_production.data.pose.types import FrameFileListing, PoseBuildOutput
from text_to_sign_production.data.sources.types import SourceCandidate


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
    gate_results: dict[str, GateResult] = {}

    def _fail(stage: str, result: GateResult, materializable: bool) -> ProcessingDecision:
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
    gate_results["source"] = source_res
    if not source_res.passed:
        return _fail("source", source_res, materializable=False)

    if frames_listing is None:
        return _fail(
            "frames",
            GateResult(status=GateStatus.DROPPED, reasons=["missing_frame_listing"]),
            materializable=False,
        )

    # 2. Frames
    frames_res = evaluate_frames_gate(frames_listing, candidate, config)
    gate_results["frames"] = frames_res
    if not frames_res.passed:
        return _fail("frames", frames_res, materializable=False)

    if pose_output is None:
        return _fail(
            "schema",
            GateResult(status=GateStatus.DROPPED, reasons=["missing_pose_output"]),
            materializable=False,
        )

    # 3. Schema
    schema_res = evaluate_schema_gate(pose_output, config)
    gate_results["schema"] = schema_res
    if not schema_res.passed:
        # Once we pass source/frames, we *might* be able to materialize a debug payload
        return _fail("schema", schema_res, materializable=True)

    # 4. Body
    body_res = evaluate_body_gate(pose_output, config)
    gate_results["body"] = body_res
    if not body_res.passed:
        return _fail("body", body_res, materializable=True)

    # 5. Hand
    hand_res = evaluate_hand_gate(pose_output, config)
    gate_results["hand"] = hand_res
    if not hand_res.passed:
        return _fail("hand", hand_res, materializable=True)

    # 6. Face
    face_res = evaluate_face_gate(pose_output, config)
    gate_results["face"] = face_res
    if not face_res.passed:
        return _fail("face", face_res, materializable=True)

    # 7. Artifact
    artifact_res = evaluate_artifact_gate(candidate, pose_output)
    gate_results["artifact"] = artifact_res
    if not artifact_res.passed:
        return _fail("artifact", artifact_res, materializable=True)

    # Passed!
    return ProcessingDecision(
        status=ProcessingStatus.PROCESSED,
        drop_stage=None,
        drop_reasons=[],
        can_materialize_debug=True,
        gate_results=gate_results,
    )
