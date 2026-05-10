from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    GateDecisionBundle,
    PassedManifestEntry,
    PreparedSample,
    SamplesDropStage,
    SamplesIssueCode,
)
from text_to_sign_production.data.dataset.manifests import (
    build_dropped_entry,
    build_passed_entry,
    write_dropped_manifest_jsonl,
    write_passed_manifest_jsonl,
)
from text_to_sign_production.data.gate.sources import (
    SourceCandidate,
    SourceMatchResult,
    sample_id_from_translation,
)
from text_to_sign_production.workflows.samples.contracts import SamplesWorkflowInvariantError
from text_to_sign_production.workflows.samples.layout import SamplesLayout
from text_to_sign_production.workflows.samples.processing.models import (
    SamplesExecutionBundle,
    SamplesPayloadOutput,
    SamplesSplitProcessingResult,
)

MANIFEST_SCHEMA_VERSION = "samples.manifest.v1"


def build_passed_workflow_entry(
    *,
    sample: PreparedSample,
    gate: GateDecisionBundle,
    payload: SamplesPayloadOutput,
) -> PassedManifestEntry:
    return build_passed_entry(
        sample,
        gate,
        payload.payload_ref,
        schema_version=MANIFEST_SCHEMA_VERSION,
    )


def build_unmatched_dropped_entry(match: SourceMatchResult) -> DroppedManifestEntry:
    return build_dropped_entry(
        schema_version=MANIFEST_SCHEMA_VERSION,
        sample_id=sample_id_from_translation(match.translation),
        split=match.split,
        drop_stage=SamplesDropStage.SOURCE,
        issue_codes=(SamplesIssueCode.SOURCE_TRUTH_INVALID,),
        text=match.translation.text,
        canonical_normalized_text=match.translation.canonical_normalized_text,
        source_video_id=match.translation.video_id,
        source_sentence_id=match.translation.sentence_id,
        source_sentence_name=match.translation.sentence_name,
    )


def build_pose_dropped_entry(
    *,
    sample: PreparedSample,
    debug_ref: str | None = None,
) -> DroppedManifestEntry:
    return build_dropped_entry(
        schema_version=MANIFEST_SCHEMA_VERSION,
        sample_id=sample.source.sample_id,
        split=sample.source.split,
        drop_stage=SamplesDropStage.POSE,
        issue_codes=(SamplesIssueCode.POSE_TRUTH_INVALID,),
        text=sample.source.text,
        canonical_normalized_text=sample.source.canonical_normalized_text,
        source_video_id=sample.source.source_video_id,
        source_sentence_id=sample.source.source_sentence_id,
        source_sentence_name=sample.source.source_sentence_name,
        debug_ref=debug_ref,
    )


def build_candidate_pose_dropped_entry(
    *,
    candidate: SourceCandidate,
    debug_ref: str | None = None,
) -> DroppedManifestEntry:
    return build_dropped_entry(
        schema_version=MANIFEST_SCHEMA_VERSION,
        sample_id=candidate.sample_id,
        split=candidate.split,
        drop_stage=SamplesDropStage.POSE,
        issue_codes=(SamplesIssueCode.POSE_TRUTH_INVALID,),
        text=candidate.text,
        canonical_normalized_text=candidate.canonical_normalized_text,
        source_video_id=candidate.video_id,
        source_sentence_id=candidate.sentence_id,
        source_sentence_name=candidate.sentence_name,
        debug_ref=debug_ref,
    )


def build_gate_dropped_entry(
    *,
    sample: PreparedSample,
    debug_ref: str | None = None,
) -> DroppedManifestEntry:
    return build_dropped_entry(
        schema_version=MANIFEST_SCHEMA_VERSION,
        sample_id=sample.source.sample_id,
        split=sample.source.split,
        drop_stage=SamplesDropStage.GATES,
        issue_codes=(SamplesIssueCode.GATE_FAILED,),
        text=sample.source.text,
        canonical_normalized_text=sample.source.canonical_normalized_text,
        source_video_id=sample.source.source_video_id,
        source_sentence_id=sample.source.source_sentence_id,
        source_sentence_name=sample.source.source_sentence_name,
        debug_ref=debug_ref,
    )


def write_samples_manifests(
    *,
    layout: SamplesLayout,
    execution_bundle: SamplesExecutionBundle,
) -> None:
    for split_result in _split_results_by_config_order(execution_bundle):
        write_passed_manifest_jsonl(
            _manifest_path(layout, "passed", split_result.split),
            split_result.passed_entries,
        )
        write_dropped_manifest_jsonl(
            _manifest_path(layout, "dropped", split_result.split),
            split_result.dropped_entries,
        )


def _split_results_by_config_order(
    execution_bundle: SamplesExecutionBundle,
) -> tuple[SamplesSplitProcessingResult, ...]:
    by_split = {result.split: result for result in execution_bundle.split_results}
    ordered_results = []
    for split in execution_bundle.workflow_result.config.splits:
        result = by_split.get(split)
        if result is None:
            raise SamplesWorkflowInvariantError(f"Missing split processing result: {split}")
        ordered_results.append(result)
    if len(by_split) != len(execution_bundle.split_results):
        raise SamplesWorkflowInvariantError("Duplicate split processing results are not allowed")
    return tuple(ordered_results)


def _manifest_path(layout: SamplesLayout, partition: str, split: str) -> Path:
    for manifest in layout.outputs.manifest_outputs:
        if manifest.partition == partition and manifest.split == split:
            return manifest.path
    raise SamplesWorkflowInvariantError(
        f"Missing {partition} manifest output path for split: {split}"
    )


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "build_candidate_pose_dropped_entry",
    "build_gate_dropped_entry",
    "build_passed_workflow_entry",
    "build_pose_dropped_entry",
    "build_unmatched_dropped_entry",
    "write_samples_manifests",
]
