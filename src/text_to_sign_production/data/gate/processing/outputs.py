"""Gate-stage dataset output planning boundary."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit, SampleStatus
from text_to_sign_production.core.models import (
    DroppedManifestEntry,
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
from text_to_sign_production.data.gate.sources import SourceCandidate, SourceMatchResult
from text_to_sign_production.data.gate.sources.candidates import sample_id_from_translation
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
    dropped_debug_payload: DatasetPayloadProduction | None
    passed_entry: PassedManifestEntry | None
    dropped_entry: DroppedManifestEntry | None


@dataclass(frozen=True, slots=True)
class GatePassedOutputPlan:
    """Gate-owned passed payload and manifest-row plan."""

    payload: DatasetPayloadProduction
    entry: PassedManifestEntry


@dataclass(frozen=True, slots=True)
class GateDroppedOutputPlan:
    """Gate-owned dropped manifest row and optional debug payload plan."""

    payload: DatasetPayloadProduction | None
    entry: DroppedManifestEntry


def plan_unmatched_source_evaluation(
    *,
    match: SourceMatchResult,
    manifest_schema_version: str,
) -> GateSourceEvaluation:
    """Plan the manifest output for an unmatched source row."""
    return GateSourceEvaluation(
        match=match,
        candidate_viability_report=None,
        prepared_sample=None,
        gate=None,
        passed_payload=None,
        dropped_debug_payload=None,
        passed_entry=None,
        dropped_entry=plan_unmatched_source_drop(
            match=match,
            manifest_schema_version=manifest_schema_version,
        ),
    )


def plan_pose_dropped_evaluation(
    *,
    match: SourceMatchResult,
    candidate: SourceCandidate,
    viability_report: CandidateViabilityReport,
    manifest_schema_version: str,
) -> GateSourceEvaluation:
    """Plan the manifest output for a matched source without usable pose."""
    return GateSourceEvaluation(
        match=match,
        candidate_viability_report=viability_report,
        prepared_sample=None,
        gate=None,
        passed_payload=None,
        dropped_debug_payload=None,
        passed_entry=None,
        dropped_entry=plan_pose_dropped_entry(
            candidate=candidate,
            manifest_schema_version=manifest_schema_version,
        ),
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
    passed_production = plan_gate_passed_outputs(
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
        passed_payload=passed_production.payload,
        dropped_debug_payload=None,
        passed_entry=passed_production.entry,
        dropped_entry=None,
    )


def plan_gate_failed_evaluation(
    *,
    match: SourceMatchResult,
    viability_report: CandidateViabilityReport,
    sample: PreparedSample,
    gate: GateDecisionBundle,
    output_refs: GateSourceOutputRefs,
    manifest_schema_version: str,
    materialize_dropped_debug_payloads: bool,
) -> GateSourceEvaluation:
    """Plan payload and manifest outputs for a gate-failed prepared sample."""
    dropped_production = plan_gate_failed_sample_outputs(
        sample=sample,
        manifest_schema_version=manifest_schema_version,
        materialize_debug_payload=materialize_dropped_debug_payloads,
        debug_payload_path=output_refs.dropped_payload_path,
        debug_payload_ref=output_refs.dropped_payload_ref,
    )
    return GateSourceEvaluation(
        match=match,
        candidate_viability_report=viability_report,
        prepared_sample=sample,
        gate=gate,
        passed_payload=None,
        dropped_debug_payload=dropped_production.payload,
        passed_entry=None,
        dropped_entry=dropped_production.entry,
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


def plan_gate_dropped_outputs(
    *,
    manifest_schema_version: str,
    sample_id: str,
    split: SampleSplit | str,
    drop_stage: GateDropStage | str,
    issue_codes: Iterable[GateDropIssueCode | str],
    text: str | None = None,
    source_video_id: str | None = None,
    source_sentence_id: str | None = None,
    source_sentence_name: str | None = None,
    debug_ref: str | None = None,
    debug_sample: PreparedSample | None = None,
    debug_payload_path: str | Path | None = None,
    debug_payload_ref: str | None = None,
) -> GateDroppedOutputPlan:
    """Build gate-owned dropped manifest row and optional debug payload plan."""
    payload: DatasetPayloadProduction | None = None
    if debug_sample is not None:
        if debug_payload_path is None or debug_payload_ref is None:
            raise ValueError("debug_sample requires debug_payload_path and debug_payload_ref.")
        payload = plan_prepared_sample_payload(
            sample=debug_sample,
            status=SampleStatus.DROPPED,
            payload_path=debug_payload_path,
            payload_ref=debug_payload_ref,
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
        debug_ref=payload.payload_ref if payload is not None else debug_ref,
    )
    return GateDroppedOutputPlan(payload=payload, entry=entry)


def plan_gate_failed_sample_outputs(
    *,
    sample: PreparedSample,
    manifest_schema_version: str,
    materialize_debug_payload: bool,
    debug_payload_path: str | Path | None = None,
    debug_payload_ref: str | None = None,
) -> GateDroppedOutputPlan:
    """Build output plans for a prepared sample rejected by gate policy."""
    return plan_gate_dropped_outputs(
        manifest_schema_version=manifest_schema_version,
        sample_id=sample.source.sample_id,
        split=sample.source.split,
        drop_stage=GateDropStage.GATES,
        issue_codes=(GateDropIssueCode.GATE_FAILED,),
        text=sample.source.text,
        source_video_id=sample.source.source_video_id,
        source_sentence_id=sample.source.source_sentence_id,
        source_sentence_name=sample.source.source_sentence_name,
        debug_sample=sample if materialize_debug_payload else None,
        debug_payload_path=debug_payload_path if materialize_debug_payload else None,
        debug_payload_ref=debug_payload_ref if materialize_debug_payload else None,
    )


def plan_unmatched_source_drop(
    *,
    match: SourceMatchResult,
    manifest_schema_version: str,
) -> DroppedManifestEntry:
    """Build the dropped row for a translation without a matched source."""
    return plan_gate_dropped_outputs(
        manifest_schema_version=manifest_schema_version,
        sample_id=sample_id_from_translation(match.translation),
        split=match.split,
        drop_stage=GateDropStage.SOURCE,
        issue_codes=(GateDropIssueCode.SOURCE_TRUTH_INVALID,),
        text=match.translation.text,
        source_video_id=match.translation.video_id,
        source_sentence_id=match.translation.sentence_id,
        source_sentence_name=match.translation.sentence_name,
    ).entry


def plan_pose_dropped_entry(
    *,
    candidate: SourceCandidate,
    manifest_schema_version: str,
    debug_ref: str | None = None,
) -> DroppedManifestEntry:
    """Build the dropped row for a matched source without valid pose truth."""
    return plan_gate_dropped_outputs(
        manifest_schema_version=manifest_schema_version,
        sample_id=candidate.sample_id,
        split=candidate.split,
        drop_stage=GateDropStage.POSE,
        issue_codes=(GateDropIssueCode.POSE_TRUTH_INVALID,),
        text=candidate.text,
        source_video_id=candidate.video_id,
        source_sentence_id=candidate.sentence_id,
        source_sentence_name=candidate.sentence_name,
        debug_ref=debug_ref,
    ).entry


__all__ = [
    "GateDroppedOutputPlan",
    "GatePassedOutputPlan",
    "GateSourceEvaluation",
    "GateSourceOutputRefs",
    "plan_gate_dropped_outputs",
    "plan_gate_failed_evaluation",
    "plan_gate_failed_sample_outputs",
    "plan_gate_passed_evaluation",
    "plan_gate_passed_outputs",
    "plan_pose_dropped_entry",
    "plan_pose_dropped_evaluation",
    "plan_unmatched_source_drop",
    "plan_unmatched_source_evaluation",
]
