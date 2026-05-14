"""Gate-stage dataset output planning boundary."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit, SampleStatus
from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    DroppedSample,
    GateDecisionBundle,
    GateDropIssueCode,
    GateDropStage,
    PassedManifestEntry,
    PreparedSample,
)
from text_to_sign_production.data.dataset.build import (
    DatasetPayloadProduction,
    plan_prepared_sample_payload,
)
from text_to_sign_production.data.dataset.manifests import build_dropped_entry, build_passed_entry
from text_to_sign_production.data.dataset.types import DatasetDroppedSampleProduction
from text_to_sign_production.data.gate.processing.dropped_samples import (
    build_gate_dropped_sample,
    build_pose_dropped_sample,
    build_source_dropped_sample,
)
from text_to_sign_production.data.gate.sources import SourceCandidate, SourceMatchResult
from text_to_sign_production.data.gate.sources.types import CandidateViabilityReport


@dataclass(frozen=True, slots=True)
class GateSourceOutputRefs:
    """Physical output refs supplied by the workflow/artifact layer."""

    passed_payload_path: Path
    passed_payload_ref: str
    dropped_payload_path: Path
    dropped_payload_ref: str


@dataclass(frozen=True, slots=True)
class GateSourceEvaluation:
    """Pure gate-stage compute result plus explicit payload write plans."""

    match: SourceMatchResult
    candidate_viability_report: CandidateViabilityReport | None
    prepared_sample: PreparedSample | None
    gate: GateDecisionBundle | None
    passed_payload: DatasetPayloadProduction | None
    dropped_sample_payload: DatasetDroppedSampleProduction | None
    passed_entry: PassedManifestEntry | None
    dropped_entry: DroppedManifestEntry | None


@dataclass(frozen=True, slots=True)
class GatePassedOutputPlan:
    """Gate-owned passed payload and manifest-row plan."""

    payload: DatasetPayloadProduction
    entry: PassedManifestEntry


@dataclass(frozen=True, slots=True)
class GateDroppedOutputPlan:
    """Gate-owned dropped sample payload and manifest-row plan."""

    payload: DatasetDroppedSampleProduction
    entry: DroppedManifestEntry


def plan_unmatched_source_evaluation(
    *,
    match: SourceMatchResult,
    output_refs: GateSourceOutputRefs,
    manifest_schema_version: str,
) -> GateSourceEvaluation:
    """Plan payload and manifest output for an unmatched source row."""
    dropped = plan_source_dropped_outputs(
        match=match,
        output_refs=output_refs,
        manifest_schema_version=manifest_schema_version,
    )
    return GateSourceEvaluation(
        match=match,
        candidate_viability_report=None,
        prepared_sample=None,
        gate=None,
        passed_payload=None,
        dropped_sample_payload=dropped.payload,
        passed_entry=None,
        dropped_entry=dropped.entry,
    )


def plan_pose_dropped_evaluation(
    *,
    match: SourceMatchResult,
    candidate: SourceCandidate,
    viability_report: CandidateViabilityReport,
    output_refs: GateSourceOutputRefs,
    manifest_schema_version: str,
    observed_frame_count: int | None,
    missing_frame_files: bool | None,
) -> GateSourceEvaluation:
    """Plan payload and manifest output for a matched source without usable pose."""
    dropped = plan_pose_dropped_outputs(
        match=match,
        candidate=candidate,
        viability_report=viability_report,
        output_refs=output_refs,
        manifest_schema_version=manifest_schema_version,
        observed_frame_count=observed_frame_count,
        missing_frame_files=missing_frame_files,
    )
    return GateSourceEvaluation(
        match=match,
        candidate_viability_report=viability_report,
        prepared_sample=None,
        gate=None,
        passed_payload=None,
        dropped_sample_payload=dropped.payload,
        passed_entry=None,
        dropped_entry=dropped.entry,
    )


def plan_gate_passed_evaluation(
    *,
    match: SourceMatchResult,
    viability_report: CandidateViabilityReport,
    sample: PreparedSample,
    gate: GateDecisionBundle,
    output_refs: GateSourceOutputRefs,
    manifest_schema_version: str,
) -> GateSourceEvaluation:
    """Plan payload and manifest outputs for a gate-passed prepared sample."""
    passed = plan_gate_passed_outputs(
        sample=sample,
        gate=gate,
        payload_path=output_refs.passed_payload_path,
        payload_ref=output_refs.passed_payload_ref,
        manifest_schema_version=manifest_schema_version,
    )
    return GateSourceEvaluation(
        match=match,
        candidate_viability_report=viability_report,
        prepared_sample=sample,
        gate=gate,
        passed_payload=passed.payload,
        dropped_sample_payload=None,
        passed_entry=passed.entry,
        dropped_entry=None,
    )


def plan_gate_failed_evaluation(
    *,
    match: SourceMatchResult,
    candidate: SourceCandidate,
    viability_report: CandidateViabilityReport,
    sample: PreparedSample,
    gate: GateDecisionBundle,
    output_refs: GateSourceOutputRefs,
    manifest_schema_version: str,
    observed_frame_count: int | None,
    missing_frame_files: bool | None,
) -> GateSourceEvaluation:
    """Plan payload and manifest outputs for a gate-failed prepared sample."""
    dropped = plan_gate_failed_sample_outputs(
        match=match,
        candidate=candidate,
        viability_report=viability_report,
        sample=sample,
        gate=gate,
        manifest_schema_version=manifest_schema_version,
        dropped_payload_path=output_refs.dropped_payload_path,
        dropped_payload_ref=output_refs.dropped_payload_ref,
        observed_frame_count=observed_frame_count,
        missing_frame_files=missing_frame_files,
    )
    return GateSourceEvaluation(
        match=match,
        candidate_viability_report=viability_report,
        prepared_sample=sample,
        gate=gate,
        passed_payload=None,
        dropped_sample_payload=dropped.payload,
        passed_entry=None,
        dropped_entry=dropped.entry,
    )


def plan_gate_passed_outputs(
    *,
    sample: PreparedSample,
    gate: GateDecisionBundle,
    payload_path: str | Path,
    payload_ref: str,
    manifest_schema_version: str,
) -> GatePassedOutputPlan:
    """Build gate-passed output plans without filesystem mutation."""
    payload = plan_prepared_sample_payload(
        sample=sample,
        status=SampleStatus.PASSED,
        payload_path=payload_path,
        payload_ref=payload_ref,
    )
    entry = build_passed_entry(
        sample,
        gate,
        payload.payload_ref,
        schema_version=manifest_schema_version,
    )
    return GatePassedOutputPlan(payload=payload, entry=entry)


def plan_source_dropped_outputs(
    *,
    match: SourceMatchResult,
    output_refs: GateSourceOutputRefs,
    manifest_schema_version: str,
) -> GateDroppedOutputPlan:
    dropped_sample = build_source_dropped_sample(match)
    return _plan_dropped_outputs(
        manifest_schema_version=manifest_schema_version,
        sample_id=dropped_sample.sample_id,
        split=dropped_sample.split,
        drop_stage=dropped_sample.drop_stage,
        issue_codes=dropped_sample.issue_codes,
        text=dropped_sample.source.text,
        source_video_id=dropped_sample.source.source_video_id,
        source_sentence_id=dropped_sample.source.source_sentence_id,
        source_sentence_name=dropped_sample.source.source_sentence_name,
        dropped_sample=dropped_sample,
        dropped_payload_path=output_refs.dropped_payload_path,
        dropped_payload_ref=output_refs.dropped_payload_ref,
    )


def plan_pose_dropped_outputs(
    *,
    match: SourceMatchResult,
    candidate: SourceCandidate,
    viability_report: CandidateViabilityReport,
    output_refs: GateSourceOutputRefs,
    manifest_schema_version: str,
    observed_frame_count: int | None,
    missing_frame_files: bool | None,
) -> GateDroppedOutputPlan:
    dropped_sample = build_pose_dropped_sample(
        match=match,
        candidate=candidate,
        viability_report=viability_report,
        observed_frame_count=observed_frame_count,
        missing_frame_files=missing_frame_files,
    )
    return _plan_dropped_outputs(
        manifest_schema_version=manifest_schema_version,
        sample_id=dropped_sample.sample_id,
        split=dropped_sample.split,
        drop_stage=dropped_sample.drop_stage,
        issue_codes=dropped_sample.issue_codes,
        text=dropped_sample.source.text,
        source_video_id=dropped_sample.source.source_video_id,
        source_sentence_id=dropped_sample.source.source_sentence_id,
        source_sentence_name=dropped_sample.source.source_sentence_name,
        dropped_sample=dropped_sample,
        dropped_payload_path=output_refs.dropped_payload_path,
        dropped_payload_ref=output_refs.dropped_payload_ref,
    )


def plan_gate_failed_sample_outputs(
    *,
    match: SourceMatchResult,
    candidate: SourceCandidate,
    viability_report: CandidateViabilityReport,
    sample: PreparedSample,
    gate: GateDecisionBundle,
    manifest_schema_version: str,
    dropped_payload_path: str | Path,
    dropped_payload_ref: str,
    observed_frame_count: int | None,
    missing_frame_files: bool | None,
) -> GateDroppedOutputPlan:
    """Build output plans for a prepared sample rejected by gate policy."""
    dropped_sample = build_gate_dropped_sample(
        match=match,
        candidate=candidate,
        viability_report=viability_report,
        sample=sample,
        gate=gate,
        observed_frame_count=observed_frame_count,
        missing_frame_files=missing_frame_files,
    )
    return _plan_dropped_outputs(
        manifest_schema_version=manifest_schema_version,
        sample_id=sample.source.sample_id,
        split=sample.source.split,
        drop_stage=GateDropStage.GATES,
        issue_codes=(GateDropIssueCode.GATE_FAILED,),
        text=sample.source.text,
        source_video_id=sample.source.source_video_id,
        source_sentence_id=sample.source.source_sentence_id,
        source_sentence_name=sample.source.source_sentence_name,
        dropped_sample=dropped_sample,
        dropped_payload_path=dropped_payload_path,
        dropped_payload_ref=dropped_payload_ref,
    )


def _plan_dropped_outputs(
    *,
    manifest_schema_version: str,
    sample_id: str,
    split: SampleSplit | str,
    drop_stage: GateDropStage | str,
    issue_codes: Iterable[GateDropIssueCode | str],
    dropped_sample: DroppedSample,
    dropped_payload_path: str | Path,
    dropped_payload_ref: str,
    text: str | None = None,
    source_video_id: str | None = None,
    source_sentence_id: str | None = None,
    source_sentence_name: str | None = None,
) -> GateDroppedOutputPlan:
    payload = DatasetDroppedSampleProduction(
        sample=dropped_sample,
        path=Path(dropped_payload_path),
        payload_ref=dropped_payload_ref,
    )
    entry = build_dropped_entry(
        schema_version=manifest_schema_version,
        sample_id=sample_id,
        split=split,
        drop_stage=drop_stage,
        issue_codes=issue_codes,
        text=text,
        source_video_id=source_video_id,
        source_sentence_id=source_sentence_id,
        source_sentence_name=source_sentence_name,
        dropped_sample_ref=payload.payload_ref,
    )
    return GateDroppedOutputPlan(payload=payload, entry=entry)


__all__ = [
    "GateDroppedOutputPlan",
    "GatePassedOutputPlan",
    "GateSourceEvaluation",
    "GateSourceOutputRefs",
    "plan_gate_failed_evaluation",
    "plan_gate_failed_sample_outputs",
    "plan_gate_passed_evaluation",
    "plan_gate_passed_outputs",
    "plan_pose_dropped_evaluation",
    "plan_source_dropped_outputs",
    "plan_unmatched_source_evaluation",
]
