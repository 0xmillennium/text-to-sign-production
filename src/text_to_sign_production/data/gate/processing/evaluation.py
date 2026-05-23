"""Gate-stage prepared sample construction and gate evaluation boundary."""

from __future__ import annotations

from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.data.dataset.build import (
    PREPARED_SAMPLE_SCHEMA_VERSION,
    build_prepared_sample,
)
from text_to_sign_production.data.dataset.confidence import (
    summarize_pose_output_confidence,
)
from text_to_sign_production.data.gate.policies import (
    GatesConfig,
    evaluate_sample_gates,
)
from text_to_sign_production.data.gate.pose import (
    FrameFileListing,
    PersonSelectionPolicy,
    PoseBuildOutput,
    build_pose_input,
    build_pose_output,
    build_tracking_result,
    discover_frame_files,
    parse_frame_file,
)
from text_to_sign_production.data.gate.processing.outputs import (
    GateSourceEvaluation,
    GateSourceOutputRefs,
    plan_gate_failed_evaluation,
    plan_gate_passed_evaluation,
    plan_pose_dropped_evaluation,
    plan_unmatched_source_evaluation,
)
from text_to_sign_production.data.gate.processing.sources import GateSourceBundle
from text_to_sign_production.data.gate.sources import SourceCandidate, assemble_candidate
from text_to_sign_production.data.gate.sources.validate import build_candidate_viability_report


def evaluate_gate_source_bundle(
    *,
    source_bundle: GateSourceBundle,
    gates_config: GatesConfig,
    person_selection_policy: PersonSelectionPolicy,
    output_refs: GateSourceOutputRefs,
    manifest_schema_version: str,
) -> GateSourceEvaluation:
    """Compute gate truth and hand off explicit dataset output plans."""
    if not source_bundle.match.matched:
        return plan_unmatched_source_evaluation(
            match=source_bundle.match,
            output_refs=output_refs,
            manifest_schema_version=manifest_schema_version,
        )

    candidate = assemble_candidate(source_bundle.match)
    frame_listing = discover_frame_files(candidate)
    viability_report = build_candidate_viability_report(
        candidate,
        observed_frame_count=frame_listing.frame_count,
        missing_frame_files=frame_listing.missing,
    )
    if not viability_report.viable:
        return plan_pose_dropped_evaluation(
            match=source_bundle.match,
            candidate=candidate,
            viability_report=viability_report,
            output_refs=output_refs,
            manifest_schema_version=manifest_schema_version,
            observed_frame_count=frame_listing.frame_count,
            missing_frame_files=frame_listing.missing,
        )

    pose_output = _build_pose_output(
        candidate=candidate,
        frame_listing=frame_listing,
        person_selection_policy=person_selection_policy,
    )
    if pose_output is None:
        pose_viability_report = build_candidate_viability_report(
            candidate,
            observed_frame_count=frame_listing.frame_count,
            missing_frame_files=True,
        )
        return plan_pose_dropped_evaluation(
            match=source_bundle.match,
            candidate=candidate,
            viability_report=pose_viability_report,
            output_refs=output_refs,
            manifest_schema_version=manifest_schema_version,
            observed_frame_count=frame_listing.frame_count,
            missing_frame_files=True,
        )

    confidence_canonicalization = summarize_pose_output_confidence(
        sample_id=candidate.sample_id,
        split=candidate.split.value if hasattr(candidate.split, "value") else candidate.split,
        pose_output=pose_output,
    )
    sample = build_prepared_sample(
        candidate,
        pose_output,
        schema_version=PREPARED_SAMPLE_SCHEMA_VERSION,
    )
    gate = evaluate_sample_gates(sample, gates_config)
    if gate.final_status is SampleStatus.PASSED:
        return plan_gate_passed_evaluation(
            match=source_bundle.match,
            viability_report=viability_report,
            sample=sample,
            gate=gate,
            output_refs=output_refs,
            manifest_schema_version=manifest_schema_version,
            confidence_canonicalization=confidence_canonicalization,
        )

    return plan_gate_failed_evaluation(
        match=source_bundle.match,
        candidate=candidate,
        viability_report=viability_report,
        sample=sample,
        gate=gate,
        output_refs=output_refs,
        manifest_schema_version=manifest_schema_version,
        observed_frame_count=frame_listing.frame_count,
        missing_frame_files=frame_listing.missing,
        confidence_canonicalization=confidence_canonicalization,
    )


def _build_pose_output(
    *,
    candidate: SourceCandidate,
    frame_listing: FrameFileListing,
    person_selection_policy: PersonSelectionPolicy,
) -> PoseBuildOutput | None:
    if frame_listing.missing or frame_listing.frame_count <= 0:
        return None
    parsed_frames = tuple(
        parse_frame_file(path, frame_index=index) for index, path in enumerate(frame_listing.files)
    )
    tracking = build_tracking_result(parsed_frames, person_selection_policy)
    return build_pose_output(
        build_pose_input(
            candidate=candidate,
            frame_listing=frame_listing,
            parsed_frames=parsed_frames,
            tracking=tracking,
        )
    )


__all__ = ["evaluate_gate_source_bundle"]
